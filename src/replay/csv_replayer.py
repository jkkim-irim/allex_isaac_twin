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
from typing import Optional

import numpy as np

from .showcase_reader import ShowcaseReader

logger = logging.getLogger("allex.replay.csv")


_ZERO_VEC_EPS = 1e-6


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

        self._t0: Optional[float] = None
        self._finished: bool = False
        self._step_count: int = 0
        self._csv_t0_main = float(self._main.t[0]) if self._main.num_samples else 0.0
        self._csv_t0_sec = (
            float(self._sec.t[0]) if (self._sec is not None and self._sec.num_samples) else 0.0
        )

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

        # CSV replay 동안 viz.update() 가 자체 qfrc 로 sim/real ring 을 덮어쓰지 않도록
        # 채널 lock — kinematic replay 는 actuator force 가 0 이라 그대로 두면 push 한
        # 값이 매 step 0 으로 리셋됨.
        viz = self._viz
        if viz is not None and hasattr(viz, "set_external_torque_sources"):
            sources = {self._main_src}
            if self._sec is not None:
                sources.add(self._sec_src)
            try:
                viz.set_external_torque_sources(sources)
            except Exception as exc:
                logger.debug(f"[replay] set_external_torque_sources warn: {exc}")

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
        logger.info("[replay] stopped")

    # ------------------------------------------------------------------
    # Per-step
    # ------------------------------------------------------------------
    def advance(self) -> None:
        """Per physics step — call from scenario.update() before viz.update."""
        if not self.is_active():
            return

        elapsed = time.monotonic() - (self._t0 or 0.0)
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

        # 진단: 첫 step 에서 실제로 push 되는 값의 magnitude 확인.
        if self._step_count == 0:
            self._diag_push_values(idx_main)

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

        # 3) 첫 호출 시 backend 진단 — 어떤 형태가 통하는지 1회 로깅.
        if self._step_count == 0:
            try:
                from isaacsim.core.simulation_manager import SimulationManager
                be = SimulationManager.get_backend()
                dev = SimulationManager.get_physics_sim_device()
            except Exception as e:
                be, dev = f"<err:{e}>", "?"
            cur_kind = type(cur_t).__name__ if cur_t is not None else "None"
            cur_dtype = getattr(cur_t, "dtype", None)
            cur_device = getattr(cur_t, "device", None)
            logger.warning(
                f"[replay][diag] backend={be} sim_device={dev} "
                f"articulation={type(articulation).__name__} "
                f"get_joint_positions: kind={cur_kind} dtype={cur_dtype} device={cur_device} "
                f"num_dof={self._num_dof}"
            )

        # 4) Newton set_joint_positions 는 torch tensor 요구 (.to(device) 호출).
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

        # 5) Kinematic write via inner Articulation view (bypass wrapper).
        if self._step_count == 0:
            logger.warning(
                f"[replay][diag] inner view={type(view).__name__} view._device={view_device} "
                f"q_buf_t dtype={self._q_buf_t.dtype} device={self._q_buf_t.device} "
                f"shape={tuple(self._q_buf_t.shape)}"
            )
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

    def _diag_push_values(self, idx: int) -> None:
        """첫 step 진단 — main reader 의 torque/force 값 magnitude 와 channel 보고."""
        viz = self._viz
        # torque: 처음 3개 joint 의 값 + 최대 abs.
        torque_items = list(self._main.torque.items())
        torque_max = max((abs(float(arr[idx])) for _, arr in torque_items), default=0.0)
        torque_sample = [(n, float(arr[idx])) for n, arr in torque_items[:3]]
        # pair force: vector norm 의 max.
        pair_max = 0.0
        pair_sample = None
        for pair, vec_arr in self._main.pair_force_vec.items():
            v = vec_arr[idx]
            n = float((v[0]**2 + v[1]**2 + v[2]**2) ** 0.5)
            if n > pair_max:
                pair_max = n
                pair_sample = (pair, v.tolist())
        # aggregate: mag.
        agg_max = 0.0
        agg_sample = None
        for tag, agg in self._main.aggregate.items():
            m = float(agg["mag"][idx])
            if abs(m) > abs(agg_max):
                agg_max = m
                agg_sample = (tag, m, agg["normal"][idx].tolist())
        logger.warning(
            f"[replay][diag] push values @idx={idx} src={self._main_src} "
            f"torque_max={torque_max:.3f} sample={torque_sample} "
            f"pair_max={pair_max:.3f} sample={pair_sample} "
            f"agg_max={agg_max:.3f} sample={agg_sample}"
        )
        # visualizer 채널 상태 — 메서드가 있다면 mode dump.
        for attr in ("get_mode", "get_torque_mode"):
            if hasattr(viz, attr):
                try:
                    logger.warning(f"[replay][diag] viz.{attr}() = {getattr(viz, attr)()}")
                except Exception as e:
                    logger.warning(f"[replay][diag] viz.{attr} err: {e}")

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
        for tag, agg in reader.aggregate.items():
            mag = float(agg["mag"][idx])
            normal = agg["normal"][idx]
            origin = agg["origin"][idx]
            vec = (float(normal[0]) * mag, float(normal[1]) * mag, float(normal[2]) * mag)
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
                viz.add_custom_force_vector(
                    name, position=(ox, oy, oz), vector=(fx, fy, fz), source=source,
                )
                self._registered_force_keys.add(key)
            except Exception as exc:
                logger.debug(f"[replay] add_custom_force_vector warn: {exc}")
            return

        # 0 vector 면 update 만 — visualizer 가 length 0 으로 그려서 거의 안 보임.
        try:
            viz.set_custom_force_vector(
                name, position=(ox, oy, oz), vector=(fx, fy, fz), source=source,
            )
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] set_custom_force_vector warn: {exc}")
