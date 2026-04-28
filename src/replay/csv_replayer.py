"""Kinematic CSV replay driver.

Drives the ALLEX articulation by directly overwriting joint positions
from one or two ``ShowcaseReader`` CSV streams (no PD, no gravity, no
collision response) while pushing torque + contact-force data into the
``ForceTorqueVisualizer`` ring + custom-force overlay channels.

Two readers can be loaded simultaneously (``main`` + ``secondary``):
- ``main`` drives kinematic joint positions and one viz source channel.
- ``secondary`` (optional) only drives the *other* viz source channel —
  joint positions stay on main.

The replayer is meant to be installed on the scenario via
``scenario.set_csv_replayer(replayer)`` and have ``advance()`` invoked
from the per-step update.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .showcase_reader import ShowcaseReader

logger = logging.getLogger("allex.replay.csv")


_ZERO_VEC_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────
# 디버그 export — replay 동안 매 sample 의 contact_pos / aggregate_origin 을
# (world, link-local) frame 둘 다로 기록해서 stop() 시 CSV 로 저장.
# 자주 쓰는 기능이 아니라서 module-level flag 로 on/off 만 토글한다.
# True → trajectory/<group>/contact_local_<csv_stem>.csv 에 저장.
EXPORT_CONTACT_LOCAL_CSV: bool = False
# CSV pair / aggregate 이름은 collision region 단위 (예: "R_Palm_Back") 라
# 실제 articulation link 이름과 다름. 매핑 못 하는 토큰은 여기서 link 로 보정.
EXPORT_LINK_ALIAS: dict[str, str] = {
    "R_Palm_Back": "R_Palm",
    "L_Palm_Back": "L_Palm",
    # "L_Elbow"/"R_Elbow"/"R_Palm" 는 그대로 사용 (실제 link 이름과 일치).
}
# ─────────────────────────────────────────────────────────────────────


def _other_source(src: str) -> str:
    return "real" if src == "sim" else "sim"


class CsvReplayer:
    """Per-physics-step kinematic replay + viz push.

    Parameters
    ----------
    reader_main, reader_secondary
        ``ShowcaseReader`` instances. ``reader_secondary`` may be None.
    articulation
        ALLEX SingleArticulation (post-initialize). Must expose
        ``dof_names`` and ``set_joint_positions`` / ``set_joint_velocities``.
    visualizer
        ``ForceTorqueVisualizer`` instance. Used for torque ring updates +
        custom force vectors.
    main_source
        ``"sim"`` or ``"real"``. Visualizer channel for main torque/force.
        Secondary uses the opposite channel. Both channels are always
        pushed when secondary is present — visibility is governed by the
        visualizer's own ``set_mode`` / ``set_torque_mode`` toggles.
    """

    # custom force vector 들의 parent path 는 force_torque_visualizer 의
    # CUSTOM_FORCE_ROOT 아래 source 별 sub-Xform 으로 자동 분기됨.

    def __init__(
        self,
        reader_main: ShowcaseReader,
        reader_secondary: Optional[ShowcaseReader],
        articulation,
        visualizer,
        *,
        main_source: str,
        plotters: Optional[list] = None,
    ):
        if reader_main is None:
            raise ValueError("reader_main is required")
        ms = str(main_source).lower()
        if ms not in ("sim", "real"):
            raise ValueError(f"main_source must be 'sim' or 'real', got {main_source!r}")

        self._main = reader_main
        self._sec = reader_secondary
        self._articulation = articulation
        self._viz = visualizer
        self._main_src = ms
        self._sec_src = _other_source(ms)
        # TorquePlotter 인스턴스(들). plot 도 CSV 값을 보여주려면 sim tau / real dict
        # override 를 매 step 흘려야 함.
        self._plotters = list(plotters) if plotters else []

        # main reader 기준 dof index 사전 빌드 — advance 핫패스에서 dict lookup 제거.
        self._dof_names: list[str] = list(getattr(articulation, "dof_names", []) or [])
        self._num_dof = len(self._dof_names)
        if self._num_dof == 0:
            logger.warning("[replay] articulation has no dof_names — pos overrides will be no-op")

        # (csv_joint, dof_idx, csv_pos_array) 튜플 — main 만.
        self._pos_plan: list[tuple[str, int, np.ndarray]] = []
        for csv_joint, arr in self._main.pos.items():
            try:
                idx = self._dof_names.index(csv_joint)
            except ValueError:
                continue
            self._pos_plan.append((csv_joint, idx, arr))

        # torque plan: (dof_idx, csv_arr) — main + secondary 둘 다. plot 갱신용.
        # plotter 는 sim 채널은 (num_dof,) array, real 채널은 dict[abbr -> tau].
        self._main_torque_dof_plan: list[tuple[int, np.ndarray]] = []
        for csv_joint, arr in self._main.torque.items():
            try:
                idx = self._dof_names.index(csv_joint)
            except ValueError:
                continue
            self._main_torque_dof_plan.append((idx, arr))
        self._sec_torque_dof_plan: list[tuple[int, np.ndarray]] = []
        if self._sec is not None:
            for csv_joint, arr in self._sec.torque.items():
                try:
                    idx = self._dof_names.index(csv_joint)
                except ValueError:
                    continue
                self._sec_torque_dof_plan.append((idx, arr))

        # plot override 버퍼 — replayer 가 mutate, plotter 가 reference 로 read.
        self._plot_sim_tau: Optional[np.ndarray] = (
            np.zeros(self._num_dof, dtype=np.float32) if self._num_dof else None
        )
        # real channel 은 abbr keyed dict. _torque_keys_to_abbr 로 변환.
        self._plot_real_by_abbr: dict = {}
        # joint_name → abbr (plotter LUT 활용; lazy import 로 의존성 경량화).
        self._joint_to_abbr: dict[str, str] = {}
        try:
            from ..ui_utils.torque_plotter import _build_dof_name_to_abbr_lut
            self._joint_to_abbr = _build_dof_name_to_abbr_lut() or {}
        except Exception as exc:
            logger.debug(f"[replay] joint→abbr LUT load warn: {exc}")

        # 매 step 같은 array 재할당을 피하기 위한 work buffer.
        self._q_buf: Optional[np.ndarray] = (
            np.zeros(self._num_dof, dtype=np.float32) if self._num_dof else None
        )
        # torch tensor work buffer — Newton set_joint_positions 가 torch tensor 요구
        # (입력에 .to(device) 호출). 첫 advance 에서 articulation 의 dtype/device 로 lazy init.
        self._q_buf_t = None
        self._zero_vel_t = None

        # custom force vector 등록 상태 — (sanitized_pair, source) 가 add 됐는지.
        self._registered_force_keys: set[tuple[str, str]] = set()

        # debug pause/resume 용 — wall-clock baseline 보정.
        self._pause_t: Optional[float] = None

        self._t0: Optional[float] = None
        self._finished: bool = False
        self._step_count: int = 0
        self._csv_t0_main = float(self._main.t[0]) if self._main.num_samples else 0.0
        self._csv_t0_sec = (
            float(self._sec.t[0]) if (self._sec is not None and self._sec.num_samples) else 0.0
        )

        # rendering tick subscription — kinematic replay 는 physics 와 무관하게
        # 매 rendering frame 마다 진행해야 함 (timeline pause / physics rate 변경에 영향 X).
        self._update_sub = None
        self._self_ticking: bool = False

        # contact-local CSV export 버퍼 (flag on 일 때만).
        self._export_records: Optional[list[dict]] = None
        self._export_pair_to_links: dict[str, list[str]] = {}
        self._export_agg_to_links: dict[str, list[str]] = {}
        if EXPORT_CONTACT_LOCAL_CSV:
            self._export_records = []
            # sanitized "L_Elbow__R_Palm_Back" → ["L_Elbow", "R_Palm_Back"]
            for sanitized in self._main.pair_force_vec.keys():
                parts = sanitized.split("__")
                if len(parts) == 2:
                    self._export_pair_to_links[sanitized] = parts
            # aggregate "R_Palm_Net_Force" → "R_Palm" (trailing _Net_Force strip).
            for tag in self._main.aggregate.keys():
                base = tag
                for suffix in ("_Net_Force", "_Force", "_Net"):
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                self._export_agg_to_links[tag] = [base]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_active(self) -> bool:
        return self._t0 is not None and not self._finished

    @property
    def duration_s(self) -> float:
        return self._main.duration_s

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._finished = False
        self._step_count = 0
        n_pairs = len(self._main.pair_force_vec)
        n_aggs = len(self._main.aggregate)
        sec_tag = "none" if self._sec is None else self._sec.path.name
        logger.info(
            f"[replay] start main={self._main.path.name} ({self._main_src}, "
            f"{self._main.duration_s:.2f}s, {n_pairs} pairs, {n_aggs} aggregates), "
            f"secondary={sec_tag}"
        )


        # rendering tick (= 매 frame) subscription. 성공하면 scenario.update() 의
        # physics-step driven advance 호출은 skip 됨 (is_self_ticking() 이 True).
        try:
            import omni.kit.app
            stream = omni.kit.app.get_app().get_update_event_stream()
            self._update_sub = stream.create_subscription_to_pop(
                self._on_render_tick, name="allex.replay.csv.tick"
            )
            self._self_ticking = True
            logger.info("[replay] subscribed to omni.kit.app update stream (rendering tick)")
        except Exception as exc:
            logger.warning(
                f"[replay] rendering tick subscribe failed: {exc} — "
                f"falling back to physics-step driven advance"
            )
            self._update_sub = None
            self._self_ticking = False

        # CSV replay 동안 viz.update() 가 자체 qfrc 로 sim/real ring 을 덮어쓰지 않도록
        # 채널 lock — kinematic replay 는 actuator force 가 0 이라 그대로 두면 push 한
        # 값이 매 step 0 으로 리셋됨.
        viz = self._viz
        sources = {self._main_src}
        if self._sec is not None:
            sources.add(self._sec_src)
        if viz is not None and hasattr(viz, "set_external_torque_sources"):
            try:
                viz.set_external_torque_sources(sources)
            except Exception as exc:
                logger.debug(f"[replay] set_external_torque_sources warn: {exc}")

        # replay 용 custom force prim 가시성 bypass — global force/torque mode 는 건드리지
        # 않고 (사용자 토글 존중), replayer 가 등록한 prim 만 mode 와 무관하게 항상 보이게.
        # 이 플래그가 없으면 default mode "off" 일 때 prim 은 만들어져도 invisible.
        if viz is not None and hasattr(viz, "set_replay_force_visible"):
            try:
                viz.set_replay_force_visible(True)
            except Exception as exc:
                logger.debug(f"[replay] set_replay_force_visible warn: {exc}")

        # TorquePlotter override — plot 도 articulation qfrc 대신 CSV 값을 그리게.
        # 첫 frame 으로 1회 채워두고 reference 등록 (이후 advance 에서 매 step 갱신).
        self._refresh_plot_buffers(idx_main=0, idx_sec=0 if self._sec is not None else None)
        for plotter in self._plotters:
            try:
                if hasattr(plotter, "set_external_sim_tau"):
                    if self._main_src == "sim":
                        plotter.set_external_sim_tau(self._plot_sim_tau)
                    elif self._sec is not None and self._sec_src == "sim":
                        plotter.set_external_sim_tau(self._plot_sim_tau)
                if hasattr(plotter, "set_external_real_by_abbr"):
                    if self._main_src == "real" or (self._sec is not None and self._sec_src == "real"):
                        plotter.set_external_real_by_abbr(self._plot_real_by_abbr)
            except Exception as exc:
                logger.debug(f"[replay] plotter override register warn: {exc}")

    def stop(self) -> None:
        # rendering tick subscription 먼저 해제 — 더 이상 advance 안 불리도록.
        if self._update_sub is not None:
            try:
                self._update_sub.unsubscribe()
            except Exception:
                try:
                    # 일부 버전은 .unsubscribe() 대신 del/None 이면 GC.
                    self._update_sub = None
                except Exception:
                    pass
            self._update_sub = None
        self._self_ticking = False

        if self._t0 is None and not self._registered_force_keys:
            return
        self._t0 = None
        self._finished = True

        # 모든 custom force vector 제거 — main + secondary 둘 다.
        viz = self._viz
        if viz is not None:
            for name, src in list(self._registered_force_keys):
                try:
                    viz.remove_custom_force_vector(name, source=src)
                except Exception as exc:
                    logger.debug(f"[replay] remove_custom_force_vector warn: {exc}")
            # torque ring lock 해제 — 정상 sim 측정이 다시 ring 을 갱신하도록.
            if hasattr(viz, "set_external_torque_sources"):
                try:
                    viz.set_external_torque_sources(set())
                except Exception as exc:
                    logger.debug(f"[replay] release torque lock warn: {exc}")
            # replay 용 visibility bypass 해제 — 다시 global mode 에 따라 가려지도록.
            if hasattr(viz, "set_replay_force_visible"):
                try:
                    viz.set_replay_force_visible(False)
                except Exception as exc:
                    logger.debug(f"[replay] clear_replay_force_visible warn: {exc}")
        # plotter override 해제 — articulation 폴링 경로 복귀.
        for plotter in self._plotters:
            try:
                if hasattr(plotter, "set_external_sim_tau"):
                    plotter.set_external_sim_tau(None)
                if hasattr(plotter, "set_external_real_by_abbr"):
                    plotter.set_external_real_by_abbr(None)
            except Exception as exc:
                logger.debug(f"[replay] plotter override clear warn: {exc}")
        self._registered_force_keys.clear()
        # contact-local CSV flush (flag on 일 때만).
        if self._export_records is not None:
            try:
                self._write_export_csv()
            except Exception as exc:
                logger.warning(f"[replay] export CSV flush warn: {exc}")
        logger.info("[replay] stopped")

    def is_self_ticking(self) -> bool:
        """True 이면 rendering tick 으로 자체 진행 — scenario.update() 는 advance 호출 skip."""
        return self._self_ticking

    def _on_render_tick(self, _event) -> None:
        """omni.kit.app update stream callback — rendering frame 마다 호출."""
        if not self.is_active():
            return
        try:
            self.advance()
        except Exception as exc:
            logger.warning(f"[replay] render-tick advance warn: {exc}")

    # ------------------------------------------------------------------
    # Per-step
    # ------------------------------------------------------------------
    # 디버깅용 슬로우모션 — contact frame 을 Property panel 로 inspect 하기 좋게.
    # 1.0 = realtime, 0.1 = 10배 느리게, 0.0 < x < 1 권장. 운영시 1.0 으로 둘 것.
    TIME_SCALE: float = 1.0

    # 외부에서 토글 가능한 pause flag — True 면 advance 가 idx 진행 멈춤 (현재 frame 유지).
    _paused: bool = False

    def set_time_scale(self, scale: float) -> None:
        """0.1 = 10x 느리게. 1.0 = 정속. <0 또는 0 은 무시."""
        if scale > 0:
            self.TIME_SCALE = float(scale)
            logger.warning(f"[replay] TIME_SCALE set to {self.TIME_SCALE}")

    def pause(self) -> None:
        self._paused = True
        logger.warning("[replay] paused")

    def resume(self) -> None:
        # 일시정지 해제 시 t0 를 보정해 그동안 흐른 시간을 빼줌 — idx 가 점프하지 않게.
        if self._paused and self._t0 is not None and self._pause_t is not None:
            self._t0 += time.monotonic() - self._pause_t
        self._paused = False
        self._pause_t = None
        logger.warning("[replay] resumed")

    def advance(self) -> None:
        """Per render tick (or physics step in fallback) — kinematic write + viz/plot push."""
        if not self.is_active():
            return

        if self._paused:
            # pause 시작 시점 기록 (resume 에서 t0 보정용).
            if getattr(self, "_pause_t", None) is None:
                self._pause_t = time.monotonic()
            return

        # TIME_SCALE 적용 — wall-clock 경과 × scale 만큼만 CSV 진행.
        elapsed = (time.monotonic() - (self._t0 or 0.0)) * self.TIME_SCALE
        if elapsed >= self._main.duration_s:
            # 마지막 프레임을 한 번 적용한 뒤 자동 stop.
            self._apply_at(self._main.num_samples - 1, sec_idx=None)
            self.stop()
            return

        # CSV t축이 0에서 시작하지 않을 수도 있으므로 t0 offset 가산.
        idx_main = self._main.index_at(self._csv_t0_main + elapsed)
        idx_sec: Optional[int] = None
        if self._sec is not None:
            idx_sec = self._sec.index_at(self._csv_t0_sec + elapsed)
        self._apply_at(idx_main, idx_sec)
        self._step_count += 1

    # ------------------------------------------------------------------
    # Per-step internals
    # ------------------------------------------------------------------
    def _apply_at(self, idx_main: int, sec_idx: Optional[int]) -> None:
        # 1) Kinematic position write.
        self._kinematic_write(idx_main)

        # 1.5) plot 버퍼 매 step 갱신 (plotter 가 reference 로 read).
        self._refresh_plot_buffers(idx_main, sec_idx)

        viz = self._viz
        if viz is None:
            return

        # 2) main torque → main_src ring channel.
        for csv_joint, arr in self._main.torque.items():
            try:
                viz.push_external_torque(csv_joint, float(arr[idx_main]),
                                         source=self._main_src)
            except Exception as exc:
                if self._step_count % 200 == 0:
                    logger.debug(f"[replay] push_external_torque main warn: {exc}")
                break

        # 3) main force vectors (per-pair + aggregates) → main_src.
        self._push_force_frame(self._main, idx_main, self._main_src)

        # 3.5) contact-local CSV export (debug flag on 일 때만).
        if self._export_records is not None:
            self._sample_local_contacts(idx_main)

        # 4) secondary (optional) — visibility 는 visualizer set_mode 가 결정.
        if self._sec is not None and sec_idx is not None:
            for csv_joint, arr in self._sec.torque.items():
                try:
                    viz.push_external_torque(csv_joint, float(arr[sec_idx]),
                                             source=self._sec_src)
                except Exception as exc:
                    if self._step_count % 200 == 0:
                        logger.debug(f"[replay] push_external_torque sec warn: {exc}")
                    break
            self._push_force_frame(self._sec, sec_idx, self._sec_src)

    def _kinematic_write(self, idx: int) -> None:
        if self._num_dof == 0 or self._q_buf is None:
            return
        articulation = self._articulation
        if articulation is None:
            return

        # 1) 현재 articulation 상태로 init — 매핑 안 된 joint 는 그대로 유지.
        cur_t = None
        try:
            cur_t = articulation.get_joint_positions()
            if cur_t is not None:
                if hasattr(cur_t, "detach"):
                    cur_np = cur_t.detach().cpu().numpy()
                else:
                    cur_np = np.asarray(cur_t)
                cur_np = np.asarray(cur_np, dtype=np.float32).reshape(-1)
                n = min(cur_np.size, self._num_dof)
                self._q_buf[:n] = cur_np[:n]
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] get_joint_positions warn: {exc}")

        # 2) CSV 로 매핑된 joint 만 덮어쓰기.
        for _name, dof_idx, arr in self._pos_plan:
            self._q_buf[dof_idx] = float(arr[idx])

        # 3) Newton set_joint_positions 는 torch tensor 요구 (.to(device) 호출).
        try:
            import torch
        except ImportError:
            if self._step_count % 200 == 0:
                logger.warning("[replay] torch unavailable — kinematic write skipped")
            return

        # SingleArticulation 의 _backend_utils 가 numpy 인데 inner Articulation
        # 의 _backend_utils 가 torch 인 mismatch 가 있어서 wrapper 를 통하면
        # expand_dims 단계에서 torch→numpy 로 변환 후 inner move_data(.to()) 가
        # numpy 에서 fail 함. → inner view 를 직접 호출하면서 (1, K) shape +
        # inner 의 device 에 맞춘 torch tensor 를 넘긴다.
        view = getattr(articulation, "_articulation_view", None)
        if view is None:
            if self._step_count == 0:
                logger.warning("[replay] articulation has no _articulation_view — skip kinematic write")
            return
        view_device = getattr(view, "_device", None) or "cpu"

        # (1, num_dof) shape, inner view 의 device 로 lazy alloc.
        if self._q_buf_t is None:
            self._q_buf_t = torch.zeros(1, self._num_dof, dtype=torch.float32, device=view_device)
            self._zero_vel_t = torch.zeros(1, self._num_dof, dtype=torch.float32, device=view_device)

        # numpy → torch 동기. host→device copy 는 .copy_ 로 처리.
        cpu_tensor = torch.from_numpy(self._q_buf)
        self._q_buf_t[0].copy_(cpu_tensor)

        # 4) Kinematic write via inner Articulation view (bypass wrapper).
        try:
            view.set_joint_positions(self._q_buf_t)
        except Exception as exc:
            if self._step_count % 200 == 0:
                import traceback
                tb = traceback.format_exc()
                logger.warning(f"[replay] set_joint_positions warn: {exc}\n{tb}")
        try:
            view.set_joint_velocities(self._zero_vel_t)
        except Exception as exc:
            if self._step_count % 200 == 0:
                import traceback
                tb = traceback.format_exc()
                logger.warning(f"[replay] set_joint_velocities warn: {exc}\n{tb}")

    # ------------------------------------------------------------------
    # Contact-local CSV export (debug)
    # ------------------------------------------------------------------
    def _sample_local_contacts(self, idx: int) -> None:
        """non-zero contact_pos / aggregate_origin 을 link local frame 으로 변환해 buffer 에 적재."""
        try:
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
        except Exception:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        def _to_local(link_name: str, wx: float, wy: float, wz: float):
            # CSV pair 이름은 collision region 일 수 있음 → 실제 link 로 alias.
            resolved = EXPORT_LINK_ALIAS.get(link_name, link_name)
            prim_path = f"/ALLEX/{resolved}_Link"
            prim = stage.GetPrimAtPath(prim_path)
            if not (prim and prim.IsValid()):
                # 첫 sample 에 한 번 경고 — prim path convention 이 다르면 lookup 전부 fail.
                if not getattr(self, "_export_link_warned", False):
                    logger.warning(
                        f"[replay][export] link prim missing: {prim_path} — "
                        f"local-frame conversion will be skipped for '{link_name}'. "
                        f"Stage 의 articulation root 가 다른 경로면 csv_replayer.py 의 "
                        f"_to_local prim_path 를 수정해야 합니다."
                    )
                    self._export_link_warned = True
                return None
            try:
                xf = UsdGeom.Xformable(prim)
                mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                inv = mat.GetInverse()
                lp = inv.Transform(Gf.Vec3d(wx, wy, wz))
                # 첫 valid 변환 1회 진단 — link world pose 와 변환 결과 dump.
                if not getattr(self, "_export_dump_done", False):
                    t = mat.ExtractTranslation()
                    logger.warning(
                        f"[replay][export] first convert: link={link_name} "
                        f"link_world_translate=({float(t[0]):.4f},{float(t[1]):.4f},{float(t[2]):.4f}) "
                        f"world_pt=({wx:.4f},{wy:.4f},{wz:.4f}) "
                        f"→ local=({float(lp[0]):.4f},{float(lp[1]):.4f},{float(lp[2]):.4f})"
                    )
                    self._export_dump_done = True
                return (float(lp[0]), float(lp[1]), float(lp[2]))
            except Exception as e:
                if not getattr(self, "_export_xform_warned", False):
                    logger.warning(f"[replay][export] xform compute warn ({link_name}): {e}")
                    self._export_xform_warned = True
                return None

        # pair contacts
        for sanitized, _ in self._main.pair_force_vec.items():
            origin_arr = self._main.pair_contact_pos.get(sanitized)
            if origin_arr is None:
                continue
            wx = float(origin_arr[idx][0])
            wy = float(origin_arr[idx][1])
            wz = float(origin_arr[idx][2])
            if wx == 0.0 and wy == 0.0 and wz == 0.0:
                continue
            if not (np.isfinite(wx) and np.isfinite(wy) and np.isfinite(wz)):
                continue
            for link_name in self._export_pair_to_links.get(sanitized, []):
                local = _to_local(link_name, wx, wy, wz)
                if local is None:
                    continue
                self._export_records.append({
                    "group": sanitized, "link": link_name, "frame_idx": idx,
                    "world_x": wx, "world_y": wy, "world_z": wz,
                    "local_x": local[0], "local_y": local[1], "local_z": local[2],
                })

        # aggregates
        for tag, agg in self._main.aggregate.items():
            wx = float(agg["origin"][idx][0])
            wy = float(agg["origin"][idx][1])
            wz = float(agg["origin"][idx][2])
            if wx == 0.0 and wy == 0.0 and wz == 0.0:
                continue
            if not (np.isfinite(wx) and np.isfinite(wy) and np.isfinite(wz)):
                continue
            for link_name in self._export_agg_to_links.get(tag, []):
                local = _to_local(link_name, wx, wy, wz)
                if local is None:
                    continue
                self._export_records.append({
                    "group": tag, "link": link_name, "frame_idx": idx,
                    "world_x": wx, "world_y": wy, "world_z": wz,
                    "local_x": local[0], "local_y": local[1], "local_z": local[2],
                })

    def _write_export_csv(self) -> None:
        if not self._export_records:
            logger.info("[replay] export csv: no records to write")
            return
        out_path = self._main.path.parent / f"contact_local_{self._main.path.stem}.csv"
        fieldnames = ["group", "link", "frame_idx",
                      "world_x", "world_y", "world_z",
                      "local_x", "local_y", "local_z"]
        try:
            import csv
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for rec in self._export_records:
                    w.writerow(rec)
                # per (group, link) average row
                agg = defaultdict(lambda: [0.0] * 6 + [0])
                for rec in self._export_records:
                    key = (rec["group"], rec["link"])
                    agg[key][0] += rec["world_x"]; agg[key][1] += rec["world_y"]; agg[key][2] += rec["world_z"]
                    agg[key][3] += rec["local_x"]; agg[key][4] += rec["local_y"]; agg[key][5] += rec["local_z"]
                    agg[key][6] += 1
                for (group, link), s in agg.items():
                    n = s[6]
                    if n == 0:
                        continue
                    w.writerow({
                        "group": group, "link": link, "frame_idx": f"AVG (n={n})",
                        "world_x": s[0] / n, "world_y": s[1] / n, "world_z": s[2] / n,
                        "local_x": s[3] / n, "local_y": s[4] / n, "local_z": s[5] / n,
                    })
            logger.info(
                f"[replay] contact-local CSV written: {out_path}  "
                f"records={len(self._export_records)}, groups={len(agg)}"
            )
        except Exception as exc:
            logger.warning(f"[replay] export CSV write failed: {exc}")

    def _refresh_plot_buffers(self, idx_main: int, idx_sec: Optional[int]) -> None:
        """plotter override 버퍼를 현재 frame 기준으로 다시 채운다.

        sim 채널 (`_plot_sim_tau`) 와 real 채널 (`_plot_real_by_abbr`) 둘 다
        main / secondary 중 해당 source 값으로 갱신. plotter 는 reference 로 잡고
        있어서 별도 push 호출 없이 매 step 최신 값을 본다.
        """
        # sim 채널 array (num_dof,) — sim 인 source 의 torque 만 dof index 로 채움.
        if self._plot_sim_tau is not None:
            self._plot_sim_tau.fill(0.0)
        # real 채널 dict — 매 step rebuild (key 셋이 같아도 안전).
        self._plot_real_by_abbr.clear()

        def _fill_sim_from(plan: list, idx: int) -> None:
            if self._plot_sim_tau is None or idx is None:
                return
            for dof_idx, arr in plan:
                self._plot_sim_tau[dof_idx] = float(arr[idx])

        def _fill_real_from(reader: ShowcaseReader, idx: int) -> None:
            if idx is None:
                return
            for csv_joint, arr in reader.torque.items():
                abbr = self._joint_to_abbr.get(csv_joint)
                if abbr is None:
                    continue
                self._plot_real_by_abbr[abbr] = float(arr[idx])

        # main → main_src 채널.
        if self._main_src == "sim":
            _fill_sim_from(self._main_torque_dof_plan, idx_main)
        else:  # real
            _fill_real_from(self._main, idx_main)

        # secondary → sec_src 채널.
        if self._sec is not None and idx_sec is not None:
            if self._sec_src == "sim":
                _fill_sim_from(self._sec_torque_dof_plan, idx_sec)
            else:
                _fill_real_from(self._sec, idx_sec)

    def _push_force_frame(self, reader: ShowcaseReader, idx: int, source: str) -> None:
        viz = self._viz
        if viz is None:
            return

        # ---- per-pair contact force vectors ----
        for sanitized_pair, vec_arr in reader.pair_force_vec.items():
            vec = vec_arr[idx]
            origin_arr = reader.pair_contact_pos.get(sanitized_pair)
            origin = origin_arr[idx] if origin_arr is not None else (0.0, 0.0, 0.0)
            self._set_or_add(viz, sanitized_pair, source, origin, vec)

        # ---- aggregate (R_Palm_Net_Force 등) ----
        # 부호 반전 — CSV 의 normal 은 contact 면이 받는 reaction 기준이라 palm 안쪽
        # (= 손바닥 아래) 으로 향함. 시각적으로 손등 위로 솟는 게 보기 좋아 -1 곱한다.
        # 양쪽 동시 표시가 필요하면 두 개 prim 으로 분리하면 됨.
        for tag, agg in reader.aggregate.items():
            mag = float(agg["mag"][idx])
            normal = agg["normal"][idx]
            origin = agg["origin"][idx]
            vec = (-float(normal[0]) * mag,
                   -float(normal[1]) * mag,
                   -float(normal[2]) * mag)
            self._set_or_add(viz, tag, source, origin, vec)

    def _set_or_add(self, viz, name: str, source: str, origin, vec) -> None:
        """``set_custom_force_vector`` 호출. 첫 호출이면 ``add_custom_force_vector``."""
        # NaN 방어 (CSV 빈 셀).
        try:
            ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
            fx, fy, fz = float(vec[0]), float(vec[1]), float(vec[2])
        except Exception:
            return
        if not (np.isfinite(ox) and np.isfinite(oy) and np.isfinite(oz)
                and np.isfinite(fx) and np.isfinite(fy) and np.isfinite(fz)):
            return

        key = (name, source)
        if key not in self._registered_force_keys:
            try:
                ret = viz.add_custom_force_vector(
                    name, position=(ox, oy, oz), vector=(fx, fy, fz), source=source,
                )
                logger.debug(
                    f"[replay] add_custom_force_vector(name={name!r}, src={source}) "
                    f"→ {ret!r}"
                )
                if ret is not None:
                    self._registered_force_keys.add(key)
                    # stage 에서 prim 실재 여부 검증 (1회).
                    try:
                        import omni.usd
                        st = omni.usd.get_context().get_stage()
                        if st is not None:
                            p = st.GetPrimAtPath(ret)
                            if not (p and p.IsValid()):
                                logger.warning(
                                    f"[replay] add_custom_force_vector created prim "
                                    f"but stage lookup invalid: {ret}"
                                )
                    except Exception as e:
                        logger.debug(f"[replay] stage check warn: {e}")
            except Exception as exc:
                import traceback
                logger.warning(f"[replay] add_custom_force_vector raised: {exc}\n{traceback.format_exc()}")
            return

        # force ≈ 0 이면 prim hide — set_custom_force_vector 가 _apply_force_vector(mag=0)
        # 로 shaft scale Z=0 (collapse) + head translate Z=-L_base (mesh 망가짐) 만들어서
        # collapsed 화살표가 visible 인 채 남는 걸 방지.
        mag = (fx * fx + fy * fy + fz * fz) ** 0.5
        zero_force = mag < _ZERO_VEC_EPS
        try:
            if zero_force:
                self._set_force_prim_visible(viz, name, source, False)
            else:
                self._set_force_prim_visible(viz, name, source, True)
                viz.set_custom_force_vector(
                    name, position=(ox, oy, oz), vector=(fx, fy, fz), source=source,
                )
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] set_custom_force_vector warn: {exc}")

    def _set_force_prim_visible(self, viz, name: str, source: str, visible: bool) -> None:
        """custom force prim 한 개의 visibility 만 토글 — global mode 와 무관.
        visualizer 의 _custom_force_prims 에서 prim 직접 가져와 MakeVisible/Invisible.
        """
        try:
            key = f"{source}::{name}"
            rec = getattr(viz, "_custom_force_prims", {}).get(key)
            if rec is None:
                return
            prim = rec.get("prim")
            if prim is None:
                return
            from pxr import UsdGeom
            if not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        except Exception as exc:
            logger.debug(f"[replay] per-prim visibility toggle warn: {exc}")

