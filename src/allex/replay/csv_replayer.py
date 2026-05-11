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
from ..config.viz_config import (
    HAND_JOINT_TORQUE_RING_MAP, EXT_JOINT_TORQUE_GAIN_MULT,
)

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


def _validate_ranges(label: str, raw: object) -> list:
    """Validate list-of-[t_on, t_off] payload. 실패 entry 는 WARNING + skip.
    리턴: sorted list[tuple[float, float]] (정상 entry 만)."""
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            f"[replay] {label}: value must be list of [t_on, t_off] pairs; "
            f"got {type(raw).__name__}"
        )
        return []
    out: list[tuple[float, float]] = []
    for r in raw:
        if (not isinstance(r, (list, tuple))) or len(r) != 2:
            logger.warning(f"[replay] {label}: bad range {r!r}, skip")
            continue
        try:
            a, b = float(r[0]), float(r[1])
        except (TypeError, ValueError):
            logger.warning(f"[replay] {label}: non-numeric range {r!r}, skip")
            continue
        if not (a < b):
            logger.warning(f"[replay] {label}: t_on>=t_off in {r!r}, skip")
            continue
        out.append((a, b))
    out.sort()
    return out


def _t_in_ranges(t: float, ranges) -> bool:
    """ranges 중 하나라도 [t_on, t_off] 안에 t 가 들어가면 True."""
    for t_on, t_off in ranges:
        if t_on <= t <= t_off:
            return True
    return False


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
        viz_scenario: Optional[dict] = None,
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

        # viz_scenario_config.json 에서 로드된 시나리오 dict. 키 (전부 optional):
        #   "force_triggers":       {"real.<topic_id>"/"sim.<channel>": [[t_on, t_off], ...]}
        #   "ext_torque_triggers":  {"real.<topic_id>":                  [[t_on, t_off], ...]}
        #   "torque_ring_triggers": {"real": [...], "sim": [...]}
        #   "graph_plots":          [ {plot_spec}, ... ]
        # 각 섹션은 _parse_*_triggers 헬퍼로 검증·정규화 후 내부 dict 에 저장.
        viz_scenario = viz_scenario or {}

        # --- force_triggers 파싱 ---
        # JSON key form: "real.<topic_id>" (real CSV ext_force topic) /
        #                "sim.<channel>"   (sim CSV pair_force_vec 또는 aggregate key).
        # 내부 dict 키 = (source, channel) tuple. source ∈ {"real", "sim"}.
        # backward-compat: "ext_force_<topic_id>" 도 "real.<topic_id>" 로 해석 (warning 1회).
        self._force_triggers: dict = self._parse_force_triggers(
            viz_scenario.get("force_triggers") or {}
        )

        # --- ext_torque_triggers 파싱 ---
        # Wrench 의 torque (Nm) ring 게이팅. real CSV 만 — sim 쪽엔 Wrench torque 없음.
        # JSON key form: "real.<topic_id>". backward-compat: "ext_torque_<topic_id>".
        # 내부 dict 키 = (source, topic_id) tuple (source 는 항상 "real" 이지만 force 와
        # 통일성 위해 tuple).
        self._ext_torque_triggers: dict = self._parse_ext_torque_triggers(
            viz_scenario.get("ext_torque_triggers") or {}
        )

        # --- torque_ring_triggers 파싱 ---
        # 키 = "real" / "sim", 값 = list of [t_on, t_off]. 매 tick 시간 보고 visualizer 의
        # 채널별 gate 를 동적 토글. 없으면 visualizer 의 user 모드 그대로.
        self._torque_ring_triggers: dict = self._parse_channel_triggers(
            viz_scenario.get("torque_ring_triggers") or {}, "torque_ring"
        )
        # 직전 적용된 gate 상태 — transition 만 visualizer 에 push (불필요한 호출 방지).
        self._torque_gate_prev: dict = {}

        # --- ext_joint_torque_triggers 파싱 ---
        # Key = CSV column name 그대로 (e.g. ``ext_joint_torque_hand_l_index_abad``).
        # ext_joint_torque CSV 는 real 만 존재 (sim 없음) 라 source/region 시멘틱 불필요
        # — 그냥 joint 별 trigger window 만 다룬다.
        # Internal dict: joint_full_name → ranges. parser 가 short→full reverse-map.
        self._ext_jtq_triggers: dict = self._parse_ext_joint_torque_triggers(
            viz_scenario.get("ext_joint_torque_triggers") or {}
        )

        # --- graph_plots: Phase 3 에서 처리 — 현재는 보관만, 미사용. ---
        self._graph_plot_specs = list(viz_scenario.get("graph_plots") or [])

        if (self._force_triggers or self._ext_torque_triggers
                or self._torque_ring_triggers or self._ext_jtq_triggers
                or self._graph_plot_specs):
            logger.info(
                f"[replay] viz_scenario loaded: "
                f"force={len(self._force_triggers)}ch, "
                f"ext_torque={len(self._ext_torque_triggers)}ch, "
                f"torque_ring={len(self._torque_ring_triggers)}ch, "
                f"ext_joint_torque={len(self._ext_jtq_triggers)}ch, "
                f"plots={len(self._graph_plot_specs)}"
            )
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

        # follower plan — kinematic write 는 set_joint_positions 가 equality constraint
        # 를 무시하고 즉시 teleport 한다. CSV 에 logged 된 한 쪽만 쓰고 다른 쪽을
        # stale 값으로 두면 매 frame equality 위반 → 솔버가 큰 보정력으로 흔들림
        # (waist Lower/Upper, finger PIP/DIP). joint_config.json::equality_constraints
        # 의 polycoef 로 빠진 쪽 q 를 직접 도출해 같이 써준다.
        #
        # bidirectional: regen tool 의 active master 분류가 actuator 기준이라 시간/버전에
        # 따라 master/follower 가 swap 될 수 있고, 옛 showcase CSV 와 새 CSV 가 logged
        # 한 쪽이 다를 수 있음 (예: 옛 CSV = Upper_Pitch logged, 새 CSV = Lower_Pitch
        # logged). 둘 다 지원하기 위해 CSV 에 어느 쪽이 들어있든 빠진 쪽을 도출.
        # forward (master→follower) 는 polycoef 그대로, reverse (follower→master) 는
        # 선형 polycoef [c0,c1,0,0,0] 의 inverse [-c0/c1, 1/c1, 0, 0, 0] 사용.
        # plan entry: (write_idx, source_idx, polycoef_to_apply)
        self._follower_plan: list[tuple[int, int, list[float]]] = []
        try:
            from ..config import load_joint_config_json
            jc = load_joint_config_json()
            csv_pos_dofs = {idx for _, idx, _ in self._pos_plan}
            for e in jc.get("equality_constraints", []):
                if not e.get("active", True):
                    continue
                f_name = e.get("follower")
                m_name = e.get("master")
                coef = e.get("polycoef", [])
                if not (f_name and m_name and coef):
                    continue
                try:
                    f_idx = self._dof_names.index(f_name)
                    m_idx = self._dof_names.index(m_name)
                except ValueError:
                    continue
                has_f = f_idx in csv_pos_dofs
                has_m = m_idx in csv_pos_dofs
                if has_m and not has_f:
                    # forward: f = poly(m). CSV 에 master 만 있음 → follower 도출.
                    self._follower_plan.append((f_idx, m_idx, list(coef)))
                elif has_f and not has_m:
                    # reverse: m = inv_poly(f). 선형일 때만 inverse 가능.
                    c = list(coef)
                    if (abs(c[2]) > 1e-12 or abs(c[3]) > 1e-12 or abs(c[4]) > 1e-12
                            or abs(c[1]) < 1e-12):
                        logger.warning(
                            f"[replay] non-linear polycoef cannot reverse-derive "
                            f"{m_name} from {f_name}; skipping"
                        )
                        continue
                    inv = [-c[0] / c[1], 1.0 / c[1], 0.0, 0.0, 0.0]
                    self._follower_plan.append((m_idx, f_idx, inv))
                # both / neither in CSV: derivation 불필요/불가
        except Exception as exc:
            logger.debug(f"[replay] follower plan build warn: {exc}")

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

        # CSV 에 torque 컬럼이 없는 ring joint (DIP/IP follower 등) — replay 동안
        # viz.update() 가 _external_torque_sources lock 으로 skip 되어 push 도 안 되면
        # ring 이 base_scale=(1,1,1) 초기값 그대로 거대하게 고정된다. 매 frame 0 push
        # 해서 MIN_SCALE 로 floor 시킴.
        self._unpushed_torque_joints: list[str] = []
        try:
            from ..config.viz_config import HAND_JOINT_TORQUE_RING_MAP
            csv_torque_joints = set(self._main.torque.keys())
            if self._sec is not None:
                csv_torque_joints |= set(self._sec.torque.keys())
            for e in HAND_JOINT_TORQUE_RING_MAP:
                if e["usd_joint_name"] not in csv_torque_joints:
                    self._unpushed_torque_joints.append(e["usd_joint_name"])
        except Exception as exc:
            logger.debug(f"[replay] enumerate unpushed torque joints warn: {exc}")

        # plot override 버퍼 — replayer 가 mutate, plotter 가 reference 로 read.
        self._plot_sim_tau: Optional[np.ndarray] = (
            np.zeros(self._num_dof, dtype=np.float32) if self._num_dof else None
        )
        # real channel 은 abbr keyed dict. _torque_keys_to_abbr 로 변환.
        self._plot_real_by_abbr: dict = {}
        # joint_name → abbr (plotter LUT 활용; lazy import 로 의존성 경량화).
        self._joint_to_abbr: dict[str, str] = {}
        try:
            from ..utils.torque_plotter import _build_dof_name_to_abbr_lut
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

        # ext_joint_torque substitution 상태 추적 — transition 시점에 log.
        # ``_ext_jtq_active`` 는 이번 step 에서 override 된 (joint, source) set,
        # ``_ext_jtq_prev_active`` 는 직전 step 의 동일 set. 매 step 끝에 diff 해
        # 새로 진입한 joint 는 warning, 빠져나간 joint 는 info 로 1회 emit.
        self._ext_jtq_active: set[tuple[str, str]] = set()
        self._ext_jtq_prev_active: set[tuple[str, str]] = set()

        # joint_name → frozenset[region] cache. ext_joint_torque_triggers 의
        # (source, region) key 매칭에 사용. visualizer 의 _dof_abbr_to_regions 와
        # 동일 룰을 csv_replayer 측에 1회 빌드 — 핫패스에서 viz private 멤버 안 만짐.
        self._joint_to_regions: dict[str, frozenset] = {}
        try:
            from ..core.force_torque_visualizer import ForceTorqueVisualizer
            for e in HAND_JOINT_TORQUE_RING_MAP:
                jn = e["usd_joint_name"]
                abbr = e["dof_abbr"]
                self._joint_to_regions[jn] = (
                    ForceTorqueVisualizer._dof_abbr_to_regions(abbr)
                )
        except Exception as exc:
            logger.debug(f"[replay] build joint→regions cache warn: {exc}")

        # custom force vector 등록 상태 — (sanitized_pair, source) 가 add 됐는지.
        self._registered_force_keys: set[tuple[str, str]] = set()
        # custom torque ring 등록 상태 (ext_torque ring 전용).
        self._registered_torque_ring_keys: set[tuple[str, str]] = set()

        # debug pause/resume 용 — wall-clock baseline 보정.
        self._pause_t: Optional[float] = None

        self._t0: Optional[float] = None
        self._finished: bool = False
        self._step_count: int = 0
        self._csv_t0_main = float(self._main.t[0]) if self._main.num_samples else 0.0
        self._csv_t0_sec = (
            float(self._sec.t[0]) if (self._sec is not None and self._sec.num_samples) else 0.0
        )

        # Plot push range tracking — 매 _apply_at 에서 prev_idx_main+1 ~ idx_main 사이의
        # 모든 CSV 샘플을 plotter 로 push 한다. -1 sentinel = 첫 호출에서 0 부터 시작.
        self._prev_idx_main_for_plot: int = -1
        self._dof_name_to_idx: dict = {n: i for i, n in enumerate(self._dof_names)}
        # 매 sample 마다 np.zeros 재할당 피하기 — 같은 buffer 재사용.
        self._replay_sim_tau_buf: Optional[np.ndarray] = (
            np.zeros(self._num_dof, dtype=np.float32) if self._num_dof else None
        )
        self._replay_real_dict_buf: dict = {}

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
        # 이미 ongoing 이면 깔끔히 stop 후 진행 — start 두 번 누름/exception 후 재시도 안전.
        if self._update_sub is not None or self._t0 is not None:
            try:
                self.stop()
            except Exception as exc:
                logger.debug(f"[replay] start() reset-stop warn: {exc}")

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

        # 재시작 안전화: 이전 run 이 남긴 stale state 명시 reset.
        self._registered_force_keys.clear()
        self._torque_gate_prev.clear()
        self._paused = False
        self._pause_t = None
        self._prev_idx_main_for_plot = -1
        # contact-local export 캐시 — flag on 일 때만 records 재생성.
        if EXPORT_CONTACT_LOCAL_CSV:
            self._export_records = []
        # 1회성 warning/dump flag 와 USDRT stage 캐시도 clear.
        for _attr in (
            "_export_rt_warned",
            "_export_link_warned",
            "_export_no_world_warned",
            "_export_dump_done",
            "_export_xform_warned",
        ):
            if hasattr(self, _attr):
                try:
                    delattr(self, _attr)
                except Exception:
                    setattr(self, _attr, False)
        self._rt_stage = None

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

        # NOTE: 이전엔 set_replay_force_visible(True) 로 사용자 모드 토글을 bypass 해서
        # 'off' 일 때도 replay force prim 을 항상 보여줬는데, 그 결과 force vector 를
        # OFF 해도 replay 중엔 안 꺼지는 UX 이슈가 발생. 사용자가 viz 패널에서 직접
        # 모드를 정하도록 — 강제 표시 제거 (기본 비활성). replay 중에 force/torque 가
        # 보이게 하려면 Visualizer 패널에서 mode 를 real/sim/both 로 설정.
        if viz is not None and hasattr(viz, "set_replay_force_visible"):
            try:
                viz.set_replay_force_visible(False)
            except Exception as exc:
                logger.debug(f"[replay] set_replay_force_visible warn: {exc}")

        # TorquePlotter replay 모드 — apply_step (physics tick) 의 자동 push 차단,
        # _apply_at 매 호출 시 prev_idx_main+1 ~ idx_main 사이의 모든 CSV 샘플을
        # push_replay_frame 으로 직접 push. CSV 의 native 시간축 (1 kHz 등) 을 그대로
        # plot. real CSV rate 가 다르더라도 매 main sample 의 t 에 맞춰 nearest sec
        # sample 로 paired frame 생성.
        self._prev_idx_main_for_plot = -1   # 첫 _apply_at 에서 0 부터 시작
        for plotter in self._plotters:
            try:
                # 기존 external override 는 비우고 (push_replay_frame 으로 직접 push).
                if hasattr(plotter, "set_external_sim_tau"):
                    plotter.set_external_sim_tau(None)
                if hasattr(plotter, "set_external_real_by_abbr"):
                    plotter.set_external_real_by_abbr(None)
                if hasattr(plotter, "set_replay_mode"):
                    plotter.set_replay_mode(True)
                # Clear subprocess deques — without this, a new run's [0..T']
                # samples stack on top of the previous run's [0..T] and the
                # resulting non-monotonic t makes one legend entry look like
                # multiple traces. No-op when the plotter isn't running yet.
                if hasattr(plotter, "reset_buffer"):
                    plotter.reset_buffer()
            except Exception as exc:
                logger.debug(f"[replay] plotter replay mode set warn: {exc}")

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

        # NOTE: early-return guard removed — plotter cleanup, replay-force visibility
        # 해제, external_torque_sources 해제는 t0/registered keys 와 무관하게
        # 항상 실행되어야 (이전 run 이 어떤 단계에서 중단됐든) stale state 남지 않음.
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
        # plotter replay 모드는 끄지 않는다 — set_replay_mode(False) 로 풀면
        # apply_step 이 articulation 폴링을 재개하면서, stop 직후 PD 가 stale
        # target (zero pose 등) 을 쫓느라 발생하는 미세 떨림이 plot 으로 계속
        # 흘러들어와 사용자가 "이전 값이 이어서 들어오는 것처럼" 보였음.
        # replay_mode True 유지하면 apply_step 이 early-return 하여 plot freeze.
        # 외부 override 만 해제 (다음 run 에서 push_replay_frame 이 깨끗하게 시작).
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
        # USDRT stage 캐시 해제 — 다음 start() 가 새 stage 로 attach 하도록.
        self._rt_stage = None

        # torque ring trigger gate 전체 해제 — replay stop 후엔 user 토글이 다시 mode
        # 만 보고 동작하도록.
        if hasattr(viz, "clear_torque_region_gates"):
            try:
                viz.clear_torque_region_gates()
            except Exception as exc:
                logger.debug(f"[replay] clear_torque_region_gates warn: {exc}")
        self._torque_gate_prev.clear()

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

        # torque_ring_triggers 평가 (real CSV time 기준 상대시간 t_rel).
        if self._torque_ring_triggers:
            self._update_torque_ring_gates(idx_main, idx_sec)

        self._step_count += 1

    def _maybe_substitute_ext_torque(
        self,
        reader: ShowcaseReader,
        joint_name: str,
        torque_arr,
        idx: int,
        source: str,
        t_rel: float,
    ) -> float:
        """ext_joint_torque_triggers[joint_name] window 안이고
        ``reader.ext_joint_torque[joint_name]`` 데이터가 있으면 ext_torque ×
        EXT_JOINT_TORQUE_GAIN_MULT 리턴. 아니면 internal torque (torque_arr[idx]) 그대로.

        Trigger key 가 joint_full 단위라 source/region 매칭 불필요. ext_joint_torque CSV
        는 real 만 존재하므로 sim reader 면 reader.ext_joint_torque 가 비어있어 자동 fallback.
        """
        if not self._ext_jtq_triggers:
            return float(torque_arr[idx])
        ranges = self._ext_jtq_triggers.get(joint_name)
        if not ranges:
            return float(torque_arr[idx])
        ext_arr = reader.ext_joint_torque.get(joint_name)
        if ext_arr is None or idx >= len(ext_arr):
            return float(torque_arr[idx])
        if not _t_in_ranges(t_rel, ranges):
            return float(torque_arr[idx])
        # substitution 발생 — transition log 용으로 active set 기록.
        self._ext_jtq_active.add((joint_name, source))
        return float(ext_arr[idx]) * EXT_JOINT_TORQUE_GAIN_MULT

    def _flush_ext_jtq_diff(self, t_rel_main: float) -> None:
        """매 step 말 호출. 이번 step 의 substitution active set 과 직전 step set 을
        diff 해서:
          (a) transition 만 로그로 emit (spam 방지)
          (b) visualizer 의 _torque_ring_force_visible_dofs 동기화 — entered joint
              는 mode/gate override 해서 강제 visible, exited 는 override 해제 →
              torque_ring_triggers 가 "off" 여도 substitution 활성 joint 의 ring 만
              표시 가능.
        """
        viz = self._viz
        curr = self._ext_jtq_active
        prev = self._ext_jtq_prev_active
        if curr != prev:
            entered = curr - prev
            exited = prev - curr
            # visualizer override set 동기화 — joint_name → dof_abbr.
            j2abbr = self._joint_name_to_abbr_cache()
            for jn, src in entered:
                abbr = j2abbr.get(jn)
                if abbr and viz is not None and hasattr(viz, "set_torque_ring_force_visible"):
                    try:
                        viz.set_torque_ring_force_visible(src, abbr, True)
                    except Exception as e:
                        logger.debug(f"[replay] force_visible set warn: {e}")
            for jn, src in exited:
                abbr = j2abbr.get(jn)
                if abbr and viz is not None and hasattr(viz, "set_torque_ring_force_visible"):
                    try:
                        viz.set_torque_ring_force_visible(src, abbr, False)
                    except Exception as e:
                        logger.debug(f"[replay] force_visible clear warn: {e}")
            for jn, src in sorted(entered):
                logger.warning(
                    f"[replay] ext_torque substitution active @ t={t_rel_main:.3f}s: "
                    f"joint={jn}, source={src} — ring 에 internal torque 대신 "
                    f"external torque 가 표시됨 (torque_ring 모드/gate override)"
                )
            for jn, src in sorted(exited):
                logger.info(
                    f"[replay] ext_torque substitution ended @ t={t_rel_main:.3f}s: "
                    f"joint={jn}, source={src} — ring 이 internal torque 로 복귀"
                )
            self._ext_jtq_prev_active = set(curr)
        self._ext_jtq_active.clear()

    def _joint_name_to_abbr_cache(self) -> dict:
        """joint_full → dof_abbr 매핑 (HAND_JOINT_TORQUE_RING_MAP 기반). lazy 빌드."""
        cache = getattr(self, "_jn2abbr", None)
        if cache is None:
            cache = {e["usd_joint_name"]: e["dof_abbr"]
                     for e in HAND_JOINT_TORQUE_RING_MAP}
            self._jn2abbr = cache
        return cache

    def _update_torque_ring_gates(self, idx_main: int, idx_sec: Optional[int]) -> None:
        """현재 t 가 (source, region) 별 active range 안에 있는지 검사 → visualizer
        region gate 토글.

        시간 기준 = real reader 의 'time' 컬럼 (real CSV 첫 sample 0 으로 한 상대시간).
        sim only group 이면 sim CSV t 사용. Transition 만 visualizer 에 push.
        """
        viz = self._viz
        if viz is None or not hasattr(viz, "set_torque_region_gate"):
            return
        ref_reader, ref_idx = self._reader_idx_for("real", idx_main, idx_sec)
        if ref_reader is None:
            ref_reader, ref_idx = self._reader_idx_for("sim", idx_main, idx_sec)
        if ref_reader is None or ref_idx is None:
            return
        try:
            t_rel = float(ref_reader.t[ref_idx] - ref_reader.t[0])
        except Exception:
            return

        # 각 (source, region) entry 별 active 평가 + transition push.
        for key, ranges in self._torque_ring_triggers.items():
            new_v = _t_in_ranges(t_rel, ranges)
            if self._torque_gate_prev.get(key) == new_v:
                continue
            self._torque_gate_prev[key] = new_v
            src, region = key
            try:
                viz.set_torque_region_gate(src, region, new_v)
            except Exception as exc:
                if self._step_count % 200 == 0:
                    logger.debug(f"[replay] set_torque_region_gate warn: {exc}")

    # ------------------------------------------------------------------
    # Per-step internals
    # ------------------------------------------------------------------
    def _apply_at(self, idx_main: int, sec_idx: Optional[int]) -> None:
        # 1) Kinematic position write.
        self._kinematic_write(idx_main)

        # 1.5) Plot push — prev_idx_main+1 ~ idx_main 의 모든 CSV 샘플을 plotter 로
        # 직접 push. CSV native rate (1 kHz) 그대로 plot 됨. real CSV 가 있으면
        # 매 main sample 의 t 에 맞춰 nearest real sample 페어링.
        self._push_replay_plot_range(idx_main)

        viz = self._viz
        if viz is None:
            return

        # 2) main torque → main_src ring channel.
        # ext_joint_torque_triggers 가 (main_src, region) 으로 현재 t 에서 활성이면
        # 그 region 의 joint 는 main.ext_joint_torque[joint] 로 substitute (per-joint).
        main_t_rel = float(self._main.t[idx_main] - self._main.t[0])
        for csv_joint, arr in self._main.torque.items():
            value = self._maybe_substitute_ext_torque(
                self._main, csv_joint, arr, idx_main, self._main_src, main_t_rel,
            )
            try:
                viz.push_external_torque(csv_joint, float(value),
                                         source=self._main_src)
            except Exception as exc:
                if self._step_count % 200 == 0:
                    logger.debug(f"[replay] push_external_torque main warn: {exc}")
                break

        # 3) Sim force vectors (pair + aggregate) → "sim" channel. Sim CSV 가
        # main 이든 secondary 든 sim source 인 reader 만 골라 그린다.
        sim_reader, sim_idx = self._reader_idx_for("sim", idx_main, sec_idx)
        if sim_reader is not None and sim_idx is not None:
            sim_t_rel = float(sim_reader.t[sim_idx] - sim_reader.t[0])
            self._push_force_frame_sim(sim_reader, sim_idx, sim_t_rel)

        # 3.5) contact-local CSV export (debug flag on 일 때만).
        if self._export_records is not None:
            self._sample_local_contacts(idx_main)

        # 4) secondary torque push (force 는 위 sim path / 아래 routed real path 가 처리).
        if self._sec is not None and sec_idx is not None:
            sec_t_rel = float(self._sec.t[sec_idx] - self._sec.t[0])
            for csv_joint, arr in self._sec.torque.items():
                value = self._maybe_substitute_ext_torque(
                    self._sec, csv_joint, arr, sec_idx, self._sec_src, sec_t_rel,
                )
                try:
                    viz.push_external_torque(csv_joint, float(value),
                                             source=self._sec_src)
                except Exception as exc:
                    if self._step_count % 200 == 0:
                        logger.debug(f"[replay] push_external_torque sec warn: {exc}")
                    break

        # 4.5) Real force vectors → "real" channel. real_reader 만 있으면 항상 routing
        # 호출, 내부에서 topic 별 게이팅. (이전엔 trigger·sim 둘 다 없으면 skip 했지만
        # real-only 에서 force 가 안 보이는 문제로 수정 — torque 와 동일 시멘틱)
        real_reader, real_idx = self._reader_idx_for("real", idx_main, sec_idx)
        if real_reader is not None and real_idx is not None:
            self._push_real_force_routed(real_reader, real_idx)
            # ext_torque arrow (Wrench.torque). force 와 독립 게이팅, 같은 chest 변환.
            self._push_real_torque_routed(real_reader, real_idx)

        # 5) CSV 에 없는 follower (DIP/IP) ring 은 0 push → base_scale 거대값 → floor.
        if self._unpushed_torque_joints:
            for jn in self._unpushed_torque_joints:
                try:
                    viz.push_external_torque(jn, 0.0, source=self._main_src)
                    if self._sec is not None:
                        viz.push_external_torque(jn, 0.0, source=self._sec_src)
                except Exception:
                    pass

        # 6) ext_torque substitution transition log (warning/info, transition 만 emit).
        if self._ext_jtq_triggers:
            self._flush_ext_jtq_diff(main_t_rel)

    def _kinematic_write(self, idx: int) -> None:
        if self._num_dof == 0 or self._q_buf is None:
            return
        articulation = self._articulation
        if articulation is None:
            return

        # Physics sim view 가 만들어진 뒤에만 set_* 호출. timeline 이 stop 이거나
        # 첫 physics step 이전에 render-tick 이 fire 되면 view 의 wrapper 가
        # carb.log_warn("Physics Simulation View is not created yet ...") 를
        # 찍는다. 같은 가드를 미리 점검해서 wasted CPU + 노이즈 방지.
        view = getattr(articulation, "_articulation_view", None)
        if view is None:
            if self._step_count == 0:
                logger.warning("[replay] articulation has no _articulation_view — skip kinematic write")
            return
        try:
            if not view.is_physics_handle_valid():
                return
        except Exception:
            pass

        # 1) 현재 articulation 상태로 init — 매핑 안 된 joint 는 그대로 유지.
        # NaN guard: 이전 physics step 이 explode 해서 joint_q 에 NaN 이 들어오면 여기서
        # cur_np 가 NaN. pos_plan + follower_plan 이 모든 dof 를 덮으면 다음 단계에서
        # 전부 overwrite 되지만, 안 그런 경우엔 NaN 이 q_buf 에 그대로 남고 set_joint_positions
        # 통해 다시 NaN feedback. 진단용 카운트 + 0 fallback 으로 끊어 둔다.
        cur_t = None
        try:
            cur_t = articulation.get_joint_positions()
            if cur_t is not None:
                if hasattr(cur_t, "detach"):
                    cur_np = cur_t.detach().cpu().numpy()
                else:
                    cur_np = np.asarray(cur_t)
                cur_np = np.asarray(cur_np, dtype=np.float32).reshape(-1)
                if not np.all(np.isfinite(cur_np)):
                    n_bad = int(np.sum(~np.isfinite(cur_np)))
                    if self._step_count % 60 == 0:
                        logger.warning(
                            f"[replay] get_joint_positions returned {n_bad} non-finite "
                            f"value(s) — physics likely exploded; sanitizing to 0.0"
                        )
                    cur_np = np.nan_to_num(cur_np, nan=0.0, posinf=0.0, neginf=0.0)
                n = min(cur_np.size, self._num_dof)
                self._q_buf[:n] = cur_np[:n]
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] get_joint_positions warn: {exc}")

        # 2) CSV 로 매핑된 joint 만 덮어쓰기.
        for _name, dof_idx, arr in self._pos_plan:
            self._q_buf[dof_idx] = float(arr[idx])

        # 2b) Equality follower 는 master q 에서 polycoef 로 도출해 같이 쓴다.
        # set_joint_positions 가 equality 무시하고 teleport 하므로 follower 를
        # 직접 일관된 값으로 안 채우면 다음 step 에서 큰 위반력 → 흔들림.
        for f_idx, m_idx, coef in self._follower_plan:
            m = float(self._q_buf[m_idx])
            v = 0.0
            mp = 1.0
            for c in coef:
                v += float(c) * mp
                mp *= m
            self._q_buf[f_idx] = v

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

        # Newton 의 set_dof_positions 는 state.joint_q 만 갱신하고 state.body_q
        # (link world transform — Fabric worldMatrix 통해 viewport 가 읽음) 는 다음
        # physics step 까지 stale. clasp 등 mesh 자가-간섭이 큰 자세에서 contact
        # 솔버가 한 step body_q 를 NaN/extreme 으로 망가뜨리면, 그 후 set_joint_positions
        # 호출이 아무리 와도 joint_q 만 깨끗할 뿐 body_q 는 망가진 채 — viewport 에는
        # base_link 외 전 링크가 사라진 것처럼 보인다.
        # eval_fk 로 매 frame joint_q → body_q 재계산해서 viewport 가 항상 깨끗한
        # kinematic 결과를 보도록 강제. 다음 physics step 이 다시 덮어쓰지만, 그 출력이
        # 망가져도 다음 render tick 의 eval_fk 가 회복한다.
        try:
            inner = getattr(view, "_physics_view", None)
            if inner is not None:
                from newton import eval_fk
                state = inner._newton_stage.state_0
                eval_fk(inner._model, state.joint_q, state.joint_qd, state)
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] eval_fk after kinematic write warn: {exc}")

    # ------------------------------------------------------------------
    # Contact-local CSV export (debug)
    # ------------------------------------------------------------------
    def _sample_local_contacts(self, idx: int) -> None:
        """non-zero contact_pos / aggregate_origin 을 link local frame 으로 변환해 buffer 에 적재.

        link world transform 은 USDRT (Fabric) 에서 읽어 *replay 시점의 실제 자세*를 반영한다.
        USD `Usd.TimeCode.Default()` 는 authored rest pose 만 보기 때문에 non-runtime
        """
        try:
            import omni.usd
            import usdrt
        except Exception:
            return

        # USDRT stage 한 번만 attach 해서 캐시. Newton 이 매 step update_fabric 으로
        # link prim 의 worldXform 어트리뷰트를 갱신하므로 여기서 읽으면 live pose.
        if getattr(self, "_rt_stage", None) is None:
            try:
                stage_id = omni.usd.get_context().get_stage_id()
                self._rt_stage = usdrt.Usd.Stage.Attach(stage_id)
            except Exception as e:
                if not getattr(self, "_export_rt_warned", False):
                    logger.warning(f"[replay][export] usdrt stage attach failed: {e}")
                    self._export_rt_warned = True
                return
        rt_stage = self._rt_stage

        def _to_local(link_name: str, wx: float, wy: float, wz: float):
            # CSV pair 이름은 collision region 일 수 있음 → 실제 link 로 alias.
            resolved = EXPORT_LINK_ALIAS.get(link_name, link_name)
            prim_path = f"/ALLEX/{resolved}_Link"
            rt_prim = rt_stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))
            if not rt_prim:
                if not getattr(self, "_export_link_warned", False):
                    logger.warning(
                        f"[replay][export] fabric prim missing: {prim_path} — "
                        f"local-frame conversion will be skipped for '{link_name}'. "
                        f"Stage 의 articulation root 가 다른 경로면 csv_replayer.py 의 "
                        f"_to_local prim_path 를 수정해야 합니다."
                    )
                    self._export_link_warned = True
                return None
            try:
                # Newton FabricManager 는 매 step `omni:fabric:worldMatrix` (Matrix4d)
                # 한 곳에만 live transform 을 쓴다. WorldPosition/WorldOrientation
                # 어트리뷰트는 init 시 SetWorldXformFromUsd() 로 한 번 채워진 뒤
                # 갱신되지 않으므로 여기서 읽으면 회전 부분이 stale identity 됨.
                # → 매트릭스 어트리뷰트를 직접 읽는다.
                attr = rt_prim.GetAttribute("omni:fabric:worldMatrix")
                if not attr or not attr.IsValid():
                    if not getattr(self, "_export_no_world_warned", False):
                        logger.warning(
                            f"[replay][export] {prim_path} has no omni:fabric:worldMatrix — "
                            f"Newton update_fabric 가 비활성이거나 prim 이 sim graph 밖일 수 있음"
                        )
                        self._export_no_world_warned = True
                    return None
                mat = attr.Get()                              # usdrt.Gf.Matrix4d (row-major, translation in last row)
                inv = mat.GetInverse()
                lp = inv.Transform(usdrt.Gf.Vec3d(wx, wy, wz))
                # 첫 valid 변환 1회 진단 — link world pose 와 변환 결과 dump.
                if not getattr(self, "_export_dump_done", False):
                    # Matrix4d 의 마지막 행이 translation (USD/Fabric 컨벤션).
                    t = mat.ExtractTranslation()
                    logger.warning(
                        f"[replay][export] first convert (Fabric runtime): link={link_name} "
                        f"link_world_pos=({float(t[0]):.4f},{float(t[1]):.4f},{float(t[2]):.4f}) "
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

    def _push_replay_plot_range(self, idx_main: int) -> None:
        """``self._prev_idx_main_for_plot+1 ~ idx_main`` 사이의 모든 main CSV 샘플을
        plotter 로 직접 push.

        sim/real 두 CSV 의 sampling rate 가 달라도 무관 — 매 main sample 의 t 로
        nearest secondary sample 을 찾아 paired frame 으로 push. plot 의 X 축은
        ``frame['t']`` (sample 의 절대 시간) 이라 두 stream 이 같은 시간축에 정렬됨.

        한 번의 호출에서 push 되는 frame 수 = idx_main - prev + 1 (보통 16~17 개,
        rendering tick 사이의 1 kHz CSV 샘플들).
        """
        if not self._plotters:
            self._prev_idx_main_for_plot = idx_main
            return
        start = self._prev_idx_main_for_plot + 1
        if start > idx_main:
            return

        main = self._main
        sec = self._sec
        main_src = self._main_src
        sim_buf = self._replay_sim_tau_buf
        real_buf = self._replay_real_dict_buf

        # main_src 에 따라 sim/real 채널을 어디서 읽을지 한 번만 결정.
        # main_src=="sim": main reader → sim, sec reader → real (있을 때).
        # main_src=="real": main reader → real, sec reader → sim (있을 때).
        sim_plan_main = self._main_torque_dof_plan if main_src == "sim" else None
        real_reader_main = main if main_src == "real" else None
        sim_plan_sec = self._sec_torque_dof_plan if (sec is not None and self._sec_src == "sim") else None
        real_reader_sec = sec if (sec is not None and self._sec_src == "real") else None

        # main CSV 의 모든 새 sample 마다 push.
        for i in range(start, idx_main + 1):
            t = float(main.t[i])

            # sim 채널 채우기.
            sim_tau: Optional[np.ndarray] = None
            if sim_plan_main is not None and sim_buf is not None:
                sim_buf.fill(0.0)
                for dof_idx, arr in sim_plan_main:
                    sim_buf[dof_idx] = float(arr[i])
                sim_tau = sim_buf
            elif sim_plan_sec is not None and sim_buf is not None:
                # main 이 real 인 경우 sec 에서 sim. 시간 매칭.
                s_idx = sec.index_at(t)
                sim_buf.fill(0.0)
                for dof_idx, arr in sim_plan_sec:
                    sim_buf[dof_idx] = float(arr[s_idx])
                sim_tau = sim_buf

            # real 채널 채우기.
            real_dict: Optional[dict] = None
            if real_reader_main is not None:
                real_buf.clear()
                for csv_joint, arr in real_reader_main.torque.items():
                    abbr = self._joint_to_abbr.get(csv_joint)
                    if abbr is None:
                        continue
                    real_buf[abbr] = float(arr[i])
                real_dict = real_buf
            elif real_reader_sec is not None:
                # main 이 sim 인 경우 sec 에서 real. 시간 매칭.
                r_idx = sec.index_at(t)
                real_buf.clear()
                for csv_joint, arr in real_reader_sec.torque.items():
                    abbr = self._joint_to_abbr.get(csv_joint)
                    if abbr is None:
                        continue
                    real_buf[abbr] = float(arr[r_idx])
                real_dict = real_buf

            for plotter in self._plotters:
                try:
                    if hasattr(plotter, "push_replay_frame"):
                        plotter.push_replay_frame(
                            t, sim_tau, real_dict, self._dof_name_to_idx,
                        )
                except Exception as exc:
                    if self._step_count % 200 == 0:
                        logger.debug(f"[replay] push_replay_frame warn: {exc}")
                    break

        self._prev_idx_main_for_plot = idx_main

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

    # ──────────────────────────────────────────────────────────────────
    # Chest_Origin_Link runtime transform (Fabric API).
    # rosbag_to_csv.py 의 ext_force / contact_pos pair 들 (이름에 "_to_" 포함) 은
    # CSV 에 chest_origin frame 의 RAW 값 그대로 박혀 있다 — sim runtime 의
    # 실제 chest world transform 으로 매 frame 변환해서 viz 에 push.
    # parallel 4-bar linkage 의 pitch-induced 평행이동까지 정확히 반영됨.
    # ──────────────────────────────────────────────────────────────────
    _CHEST_PRIM_PATH = "/ALLEX/Chest_Origin_Link"

    def _get_chest_world_matrix(self):
        """Chest_Origin_Link 의 omni:fabric:worldMatrix (usdrt Matrix4d) — 또는 None."""
        try:
            import omni.usd
            import usdrt
        except Exception:
            return None
        if getattr(self, "_rt_stage", None) is None:
            try:
                stage_id = omni.usd.get_context().get_stage_id()
                self._rt_stage = usdrt.Usd.Stage.Attach(stage_id)
            except Exception as e:
                if not getattr(self, "_chest_xform_warned", False):
                    logger.warning(f"[replay] usdrt stage attach failed for chest xform: {e}")
                    self._chest_xform_warned = True
                return None
        try:
            rt_prim = self._rt_stage.GetPrimAtPath(usdrt.Sdf.Path(self._CHEST_PRIM_PATH))
            if not rt_prim:
                if not getattr(self, "_chest_prim_warned", False):
                    logger.warning(f"[replay] chest prim missing: {self._CHEST_PRIM_PATH} — "
                                   f"ext_force/contact_pos 변환 불가, raw chest-frame 값 그대로 push.")
                    self._chest_prim_warned = True
                return None
            attr = rt_prim.GetAttribute("omni:fabric:worldMatrix")
            if not attr or not attr.IsValid():
                return None
            return attr.Get()
        except Exception as e:
            if not getattr(self, "_chest_xform_runtime_warned", False):
                logger.warning(f"[replay] chest xform query warn: {e}")
                self._chest_xform_runtime_warned = True
            return None

    @staticmethod
    def _xform_vec_chest_to_world(vec_chest, mat) -> tuple:
        """Force vector — rotation only (translation 무시).
        usdrt Matrix4d 는 row-major (translation 이 row 3): rotation 은 0..2 행."""
        x, y, z = float(vec_chest[0]), float(vec_chest[1]), float(vec_chest[2])
        return (x * mat[0][0] + y * mat[1][0] + z * mat[2][0],
                x * mat[0][1] + y * mat[1][1] + z * mat[2][1],
                x * mat[0][2] + y * mat[1][2] + z * mat[2][2])

    @staticmethod
    def _xform_pt_chest_to_world(pt_chest, mat) -> tuple:
        """Contact point — full SE3 (rotation + translation)."""
        x, y, z = float(pt_chest[0]), float(pt_chest[1]), float(pt_chest[2])
        return (x * mat[0][0] + y * mat[1][0] + z * mat[2][0] + mat[3][0],
                x * mat[0][1] + y * mat[1][1] + z * mat[2][1] + mat[3][1],
                x * mat[0][2] + y * mat[1][2] + z * mat[2][2] + mat[3][2])

    # ------------------------------------------------------------------
    # viz_scenario 파싱 헬퍼 (init 시 1회)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_force_triggers(raw: dict) -> dict:
        """JSON 의 force_triggers section 을 internal dict 로 변환.

        JSON key form:
          - ``"real.<topic_id>"`` : real CSV ext_force topic_id
            (showcase_reader.topic_force_vec key 와 매칭)
          - ``"sim.<channel>"``   : sim CSV pair_force_vec sanitized key 또는
            aggregate key (예: ``L_Elbow__R_Palm_Back``, ``R_Palm_Net_Force``)

        Backward-compat: ``"ext_force_<topic_id>"`` → ``"real.<topic_id>"`` 로 해석
        (legacy form, section 단위로 WARNING 1회).

        Internal key: ``(source, channel)`` tuple.
        """
        return CsvReplayer._parse_force_triggers_v2(
            raw, "force_trigger",
            legacy_prefix="ext_force_", allow_sim=True,
        )

    @staticmethod
    def _parse_ext_torque_triggers(raw: dict) -> dict:
        """ext_torque_triggers — Wrench torque (Nm) ring 게이팅.

        JSON key form: ``"real.<topic_id>"``. sim 쪽은 Wrench torque 데이터가 없어
        ``"sim.*"`` 키가 들어오면 WARNING 후 skip.

        Backward-compat: ``"ext_torque_<topic_id>"`` → ``"real.<topic_id>"`` 로 해석.

        Internal key: ``(source, topic_id)`` tuple (source 는 항상 ``"real"``).
        """
        return CsvReplayer._parse_force_triggers_v2(
            raw, "ext_torque_trigger",
            legacy_prefix="ext_torque_", allow_sim=False,
        )

    @staticmethod
    def _parse_ext_joint_torque_triggers(raw: dict) -> dict:
        """ext_joint_torque_triggers — CSV column key 그대로 받는 per-joint trigger.

        JSON key form:    ``ext_joint_torque_<short_key>`` (CSV 컬럼명과 동일).
        Internal key:     ``<joint_full_name>`` (short → full reverse-map).
        Value:            normal range list 또는 explicit OFF (``"off"`` / ``false`` / ``[]``).
        """
        if not isinstance(raw, dict):
            return {}
        # short_key → joint_full reverse-map (lazy load).
        try:
            from ..trajectory_generate.joint_name_map import EXT_TORQUE_JOINT_CSV_KEYS
            short_to_full = {v: k for k, v in EXT_TORQUE_JOINT_CSV_KEYS.items()}
        except Exception:
            short_to_full = {}

        out: dict = {}
        prefix = "ext_joint_torque_"
        for raw_key, ranges in raw.items():
            if not isinstance(raw_key, str):
                continue
            if not raw_key.startswith(prefix):
                logger.warning(
                    f"[replay] ext_joint_torque_trigger key {raw_key!r} 은 "
                    f"'{prefix}<short>' 형태여야 합니다 (CSV 컬럼명 그대로). skip."
                )
                continue
            short = raw_key[len(prefix):]
            joint_full = short_to_full.get(short)
            if joint_full is None:
                logger.warning(
                    f"[replay] ext_joint_torque_trigger {raw_key!r}: 알 수 없는 short_key "
                    f"'{short}'. EXT_TORQUE_JOINT_CSV_KEYS 에 없습니다. skip."
                )
                continue
            is_off = (ranges is False
                      or (isinstance(ranges, str) and ranges.strip().lower() == "off")
                      or (isinstance(ranges, list) and len(ranges) == 0))
            if is_off:
                out[joint_full] = []
                continue
            clean = _validate_ranges(f"ext_joint_torque_trigger '{raw_key}'", ranges)
            if clean:
                out[joint_full] = clean
        return out

    @staticmethod
    def _parse_force_triggers_v2(
        raw: dict, label: str, *,
        legacy_prefix: str, allow_sim: bool = True,
    ) -> dict:
        """force_triggers / ext_torque_triggers 공용 파서 (dotted form).

        Key form:
          - ``"real.<channel>"`` / ``"sim.<channel>"`` (allow_sim=False 면 sim 거부)
          - legacy: ``"<legacy_prefix><channel>"`` → ``"real.<channel>"`` 로 해석
            (section 단위로 WARNING 1회)

        Value 규칙:
          - ``[[t_on, t_off], ...]`` : 정상 trigger window
          - ``"off"`` / ``false`` / ``[]`` : explicit OFF — 영구 hide

        Returns: dict[(source, channel) -> ranges]. 빈 ranges list = always-off sentinel.
        """
        if not isinstance(raw, dict):
            return {}
        out: dict = {}
        legacy_warned = False
        for raw_key, ranges in raw.items():
            if not isinstance(raw_key, str):
                continue
            # dotted form 우선 검사.
            source: Optional[str] = None
            channel: Optional[str] = None
            if "." in raw_key:
                src_part, ch_part = raw_key.split(".", 1)
                src_part = src_part.strip()
                ch_part = ch_part.strip()
                if src_part in ("real", "sim") and ch_part:
                    source, channel = src_part, ch_part
            if source is None and raw_key.startswith(legacy_prefix):
                # legacy form — real 로 매핑. section 단위로 한 번만 warning.
                ch_part = raw_key[len(legacy_prefix):].strip()
                if ch_part:
                    if not legacy_warned:
                        logger.warning(
                            f"[replay] {label}: legacy key form '{legacy_prefix}<id>' "
                            f"detected (e.g. {raw_key!r}); please migrate to "
                            f"'real.<id>' dotted form. legacy keys still accepted."
                        )
                        legacy_warned = True
                    source, channel = "real", ch_part
            if source is None or channel is None:
                logger.warning(
                    f"[replay] {label} key {raw_key!r} 은 "
                    f"'real.<id>' / 'sim.<id>' (또는 legacy '{legacy_prefix}<id>') "
                    f"형태여야 합니다. skip."
                )
                continue
            if source == "sim" and not allow_sim:
                logger.warning(
                    f"[replay] {label} key {raw_key!r}: sim source 미지원 — skip."
                )
                continue
            is_off = (ranges is False
                      or (isinstance(ranges, str) and ranges.strip().lower() == "off")
                      or (isinstance(ranges, list) and len(ranges) == 0))
            if is_off:
                out[(source, channel)] = []
                continue
            clean = _validate_ranges(f"{label} '{raw_key}'", ranges)
            if clean:
                out[(source, channel)] = clean
        return out

    @staticmethod
    def _parse_channel_triggers(raw: dict, label: str) -> dict:
        """``{'real': [...], 'sim': [...]}`` 또는 ``{'real.<region>': [...], ...}`` 형태 trigger 파싱.

        Key form:
          - ``"real"`` / ``"sim"`` : 그 source 의 모든 ring (region=None 으로 저장)
          - ``"real.<region>"`` / ``"sim.<region>"`` : 특정 region 만
            (예: "real.Arm_L", "real.Hand_L_thumb")

        Value form:
          - ``[[t_on, t_off], ...]`` : 정상 trigger window
          - ``"off"`` / ``false`` / ``[]`` : explicit OFF — 그 (source, region) 의
            ring 을 replay 동안 항상 hide. 빈 dict ``{}`` 와 다름:
              · ``{}`` (전체) → trigger 없음 → mode 기본값 (visible)
              · ``{"real": "off"}`` → real source 의 모든 ring 강제 hide

        Returns: dict[(source, region|None) -> ranges]. 빈 ranges list 면 "always off"
        sentinel — _update_*_gates 가 첫 step 에 False 로 push.
        """
        if not isinstance(raw, dict):
            return {}
        out: dict = {}
        for ch_raw, ranges in raw.items():
            if not isinstance(ch_raw, str):
                continue
            if "." in ch_raw:
                ch, region = ch_raw.split(".", 1)
                region = region.strip() or None
            else:
                ch, region = ch_raw, None
            if ch not in ("real", "sim"):
                logger.warning(f"[replay] {label}_trigger: invalid channel {ch_raw!r}, skip "
                               f"(expected 'real'/'sim' or 'real.<region>'/'sim.<region>')")
                continue
            # explicit OFF: "off" / false / [] → 항상 hide.
            is_off = (ranges is False
                      or (isinstance(ranges, str) and ranges.strip().lower() == "off")
                      or (isinstance(ranges, list) and len(ranges) == 0))
            if is_off:
                out[(ch, region)] = []  # 빈 ranges = always-off sentinel
                continue
            clean = _validate_ranges(f"{label}_trigger.{ch_raw}", ranges)
            if clean:
                out[(ch, region)] = clean
        return out

    def _reader_idx_for(self, source: str, idx_main: int,
                        sec_idx: Optional[int]) -> tuple:
        """source ('sim'/'real') 에 해당하는 (reader, idx) 리턴. 없으면 (None, None)."""
        if self._main_src == source:
            return self._main, idx_main
        if self._sec_src == source and self._sec is not None and sec_idx is not None:
            return self._sec, sec_idx
        return None, None

    def _push_force_frame_sim(self, reader: ShowcaseReader, idx: int,
                              t_rel: float) -> None:
        """Sim CSV 의 force vector 채널들 push (source='sim'). World frame 그대로 — chest 변환 없음.

        Channel:
          - pair_force_vec[<pair>] : showcase_logger 의 contact pair (allowlist).
          - aggregate[<tag>]       : R_Palm_Net_Force 등 그룹 합산.

        Gate 결정 순서 (sim source 별 독립, torque_ring_triggers 와 동일 시멘틱):
          1) self._force_triggers[("sim", channel)] 있음 → 시간 범위 in/out
          2) sim source 의 다른 채널에 trigger 있음 + 이 채널만 미정의 → exclusive hide
          3) sim source 의 trigger 통째로 없음 → default visible
        """
        viz = self._viz
        if viz is None:
            return

        sim_section_present = any(s == "sim" for (s, _) in self._force_triggers)

        def _sim_active(channel: str) -> bool:
            triggers = self._force_triggers.get(("sim", channel))
            if triggers is not None:
                if not triggers:
                    return False
                return _t_in_ranges(t_rel, triggers)
            if sim_section_present:
                return False
            return True

        for sanitized_pair, vec_arr in reader.pair_force_vec.items():
            if not _sim_active(sanitized_pair):
                self._set_force_prim_visible(viz, sanitized_pair, "sim", False)
                continue
            vec = vec_arr[idx]
            origin_arr = reader.pair_contact_pos.get(sanitized_pair)
            origin = origin_arr[idx] if origin_arr is not None else (0.0, 0.0, 0.0)
            self._set_or_add(viz, sanitized_pair, "sim", origin, vec)

        # aggregate — CSV 의 normal 은 reaction 기준이라 -1 곱해 손등 위로 향하게.
        for tag, agg in reader.aggregate.items():
            if not _sim_active(tag):
                self._set_force_prim_visible(viz, tag, "sim", False)
                continue
            mag = float(agg["mag"][idx])
            normal = agg["normal"][idx]
            origin = agg["origin"][idx]
            vec = (-float(normal[0]) * mag,
                   -float(normal[1]) * mag,
                   -float(normal[2]) * mag)
            self._set_or_add(viz, tag, "sim", origin, vec)

    def _push_real_force_routed(self, real_reader: ShowcaseReader,
                                real_idx: int) -> None:
        """real CSV 의 모든 ext_force topic 을 'real' 채널에 push (topic_id 1 prim).

        Iteration: ``real_reader.topic_force_vec.keys()`` — CSV 에 실제 존재하는 모든
        topic_id 순회. 임의로 추가된 새 ext_force_<id>_* 컬럼도 자동 인식.

        Gate 결정 순서 (real source 별 독립, torque_ring_triggers 와 동일 시멘틱):
          1) self._force_triggers[("real", topic_id)] 있음 → 시간 범위 in/out
          2) real source 의 다른 topic 에 trigger 있는데 이 topic 만 미정의
             → exclusive hide
          3) real source 의 trigger 가 통째로 없음 → default visible
             (mode-only, threshold-hide 가 알아서 처리)

        Prim name = topic_id (예 "arm_r"). source dir 가 sim/real 로 분리되므로
        sim 쪽 prim (sim_channel 이름) 과 충돌 없음.
        """
        viz = self._viz
        if viz is None:
            return

        chest_mat = self._get_chest_world_matrix()
        if chest_mat is None:
            return

        # trigger 시간 비교용 — real CSV 첫 sample 을 0 으로 한 상대시간 (사용자가
        # CSV/JSON 에서 직접 보는 값과 동일).
        current_t_rel = float(real_reader.t[real_idx] - real_reader.t[0])
        real_section_present = any(s == "real" for (s, _) in self._force_triggers)

        for topic_id, force_arr in real_reader.topic_force_vec.items():
            # ----- gate 결정 -----
            triggers = self._force_triggers.get(("real", topic_id))
            if triggers is not None:
                if not triggers:
                    active = False
                else:
                    active = _t_in_ranges(current_t_rel, triggers)
            elif real_section_present:
                # 다른 real topic 에 trigger 있음, 이 topic 은 미정의 → exclusive hide
                active = False
            else:
                # real source 의 trigger 가 통째로 없음 → default visible.
                active = True

            if not active:
                self._set_force_prim_visible(viz, topic_id, "real", False)
                continue

            f_chest = force_arr[real_idx]
            f_world = self._xform_vec_chest_to_world(f_chest, chest_mat)

            cpos_arr = real_reader.topic_contact_pos.get(topic_id)
            if cpos_arr is not None:
                p_chest = cpos_arr[real_idx]
                p_world = self._xform_pt_chest_to_world(p_chest, chest_mat)
            else:
                p_world = (float(chest_mat[3][0]),
                           float(chest_mat[3][1]),
                           float(chest_mat[3][2]))

            self._set_or_add(viz, topic_id, "real", p_world, f_world)

    def _push_real_torque_routed(self, real_reader: ShowcaseReader,
                                 real_idx: int) -> None:
        """real CSV 의 모든 ext_torque topic 을 contact_pos 중심 **torque ring**
        (arrow 아님!) 으로 push.

        Iteration: ``real_reader.topic_torque_vec.keys()``. 각 topic_id 에 대해
        chest→world 변환 후 ``torque_<topic_id>`` 이름으로 free-floating ring 생성.
        기존 joint torque ring 과는 prim namespace 분리 (``/World/AllexTorqueRing/real/...``).

        Gate 결정 순서 (ext_torque_triggers 만 보고 결정 — ext_force gate 와 독립):
          1) self._ext_torque_triggers[("real", topic_id)] 있음 → 시간 범위 in/out
          2) real source 의 다른 topic 에 trigger 있는데 이 topic 만 미정의
             → exclusive hide
          3) real source 의 trigger 가 통째로 없음 → default visible
             (hide_below 임계가 알아서 작은 |τ| 자동 hide)

        Ring 의 +Z 축 = torque vector 방향 (world frame). 위치 = contact_pos
        (topic_contact_pos 가 있으면 chest→world, 없으면 chest_origin link 위치).
        스케일 = clip(|τ|*EXT_TORQUE_GAIN, EXT_TORQUE_MIN_SCALE, EXT_TORQUE_MAX_SCALE).
        """
        viz = self._viz
        if viz is None:
            return
        if not real_reader.topic_torque_vec:
            return

        chest_mat = self._get_chest_world_matrix()
        if chest_mat is None:
            return

        current_t_rel = float(real_reader.t[real_idx] - real_reader.t[0])
        real_section_present = any(s == "real" for (s, _) in self._ext_torque_triggers)

        for topic_id, torque_arr in real_reader.topic_torque_vec.items():
            triggers = self._ext_torque_triggers.get(("real", topic_id))
            if triggers is not None:
                if not triggers:
                    active = False
                else:
                    active = _t_in_ranges(current_t_rel, triggers)
            elif real_section_present:
                active = False
            else:
                active = True

            prim_name = f"torque_{topic_id}"
            if not active:
                self._set_torque_ring_prim_visible(viz, prim_name, "real", False)
                continue

            t_chest = torque_arr[real_idx]
            t_world = self._xform_vec_chest_to_world(t_chest, chest_mat)
            magnitude = float((t_world[0] * t_world[0]
                               + t_world[1] * t_world[1]
                               + t_world[2] * t_world[2]) ** 0.5)

            cpos_arr = real_reader.topic_contact_pos.get(topic_id)
            if cpos_arr is not None:
                p_chest = cpos_arr[real_idx]
                p_world = self._xform_pt_chest_to_world(p_chest, chest_mat)
            else:
                p_world = (float(chest_mat[3][0]),
                           float(chest_mat[3][1]),
                           float(chest_mat[3][2]))

            self._set_or_add_torque_ring(viz, prim_name, "real",
                                         p_world, t_world, magnitude)

    def _set_or_add_torque_ring(self, viz, name: str, source: str,
                                position, axis_vec, magnitude: float) -> None:
        """add_custom_torque_ring 첫 호출 / 이후 set_custom_torque_ring 으로 갱신."""
        try:
            ox, oy, oz = float(position[0]), float(position[1]), float(position[2])
            ax, ay, az = float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])
            mag = float(magnitude)
        except Exception:
            return
        if not (np.isfinite(ox) and np.isfinite(oy) and np.isfinite(oz)
                and np.isfinite(ax) and np.isfinite(ay) and np.isfinite(az)
                and np.isfinite(mag)):
            return

        key = (name, source)
        if key not in self._registered_torque_ring_keys:
            try:
                ret = viz.add_custom_torque_ring(
                    name, position=(ox, oy, oz), axis=(ax, ay, az),
                    magnitude=mag, source=source,
                )
                if ret is not None:
                    self._registered_torque_ring_keys.add(key)
            except Exception as exc:
                import traceback
                logger.warning(
                    f"[replay] add_custom_torque_ring raised: {exc}\n"
                    f"{traceback.format_exc()}"
                )
            return

        try:
            viz.set_custom_torque_ring(
                name, source=source,
                position=(ox, oy, oz), axis=(ax, ay, az), magnitude=mag,
            )
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] set_custom_torque_ring warn: {exc}")

    def _set_torque_ring_prim_visible(self, viz, name: str, source: str,
                                       visible: bool) -> None:
        """custom torque ring prim visibility 토글. 충돌 방지를 위해 session layer 통해."""
        try:
            key = f"{source}::{name}"
            rec = getattr(viz, "_custom_torque_ring_prims", {}).get(key)
            if rec is None:
                return
            prim = rec.get("prim")
            if prim is None:
                return
            if visible:
                # torque_mode 검사 — replay bypass 없으면 mode 따라 hide.
                bypass = getattr(viz, "_replay_force_visible", False)
                if not bypass:
                    mode = getattr(viz, "_torque_mode", "off")
                    src_lc = (source or "").lower()
                    if src_lc == "real" and mode not in ("real", "both"):
                        return
                    if src_lc == "sim" and mode not in ("sim", "both"):
                        return
            set_vis = getattr(viz, "_set_visible", None)
            if set_vis is not None:
                set_vis(prim, visible)
            try:
                prim_path = prim.GetPath().pathString
                cache = getattr(viz, "_threshold_hide_state", None)
                if cache is not None:
                    cache[prim_path] = (not visible)
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"[replay] torque ring visibility toggle warn: {exc}")

    def _set_or_add(self, viz, name: str, source: str, origin, vec,
                    kind: str = "force") -> None:
        """``set_custom_force_vector`` 호출. 첫 호출이면 ``add_custom_force_vector``.

        kind: 'force' (default) 또는 'torque'. 'torque' 면 ext_torque 스케일/색이
        적용된 화살표를 만든다. set_custom_force_vector 는 prim 의 kind 를 보고
        자동으로 같은 스케일을 사용.
        """
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
                    kind=kind,
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

        # Visibility / scale 결정은 visualizer 에 일임 — visualizer 의 _apply_force_scale 가
        # FORCE_VEC_HIDE_BELOW 임계 기반으로 visibility 토글 + 캐시 동기화. 여기서
        # 직접 _set_force_prim_visible 박으면 그 임계 hide 를 덮어쓰거나 (forced True),
        # 캐시를 stale 로 만들어서 (forced False) hide_below 가 무력화됨.
        # mag=0 이라도 set_custom_force_vector(vector=(0,0,0)) 호출하면 s=0 < hide_below
        # 으로 자동 hide 됨 (collapsed mesh 가 visible 인 채로 남지 않음).
        try:
            viz.set_custom_force_vector(
                name, position=(ox, oy, oz), vector=(fx, fy, fz), source=source,
            )
        except Exception as exc:
            if self._step_count % 200 == 0:
                logger.debug(f"[replay] set_custom_force_vector warn: {exc}")

    def _set_force_prim_visible(self, viz, name: str, source: str, visible: bool) -> None:
        """custom force prim 한 개의 visibility 토글.

        Hide (visible=False) 는 mode 와 무관하게 항상 적용 — force=0 이면 화살표
        무조건 안 보여야 함.

        Show (visible=True) 는 viz 의 force mode 가 해당 source 를 허용하는 경우만
        적용. user 가 Force Visualizer 를 OFF 했으면 force 가 들어와도 화살표 안
        나타나야 함.

        ★ visualizer 의 `_set_visible` 을 호출해야 session_edit 컨텍스트로 visibility
        가 session layer 에 authored 됨. 직접 MakeVisible/MakeInvisible 호출하면
        root layer 에 쓰여서, prim 자체가 session 에 authored 된 visibility 가
        있으면 layer composition strength 상 session 이 이겨서 hide/show 가 안 먹힘.
        """
        try:
            key = f"{source}::{name}"
            rec = getattr(viz, "_custom_force_prims", {}).get(key)
            if rec is None:
                return
            prim = rec.get("prim")
            if prim is None:
                return
            # Show 는 mode 검사 — replay bypass 가 켜져 있으면 무조건 OK.
            if visible:
                bypass = getattr(viz, "_replay_force_visible", False)
                if not bypass:
                    mode = getattr(viz, "_mode", "off")
                    src_lc = (source or "").lower()
                    if src_lc == "real" and mode not in ("real", "both"):
                        return
                    if src_lc == "sim" and mode not in ("sim", "both"):
                        return
                    # source="user" 또는 미지: 통과 (user 는 항상 visible 약속).
            set_vis = getattr(viz, "_set_visible", None)
            if set_vis is None:
                from pxr import UsdGeom
                if not prim.IsValid():
                    return
                imageable = UsdGeom.Imageable(prim)
                ctx = getattr(viz, "_session_edit", None)
                if ctx is not None:
                    with ctx():
                        if visible:
                            imageable.MakeVisible()
                        else:
                            imageable.MakeInvisible()
                else:
                    if visible:
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()
            else:
                set_vis(prim, visible)

            # _apply_force_scale 가 관리하는 threshold-hide cache 와 sync.
            # 이 helper 가 외부에서 visibility 를 강제로 토글하면 그 cache 가
            # stale 이 되어, 다음 _apply_force_scale 호출이 transition 을 못
            # 잡아 USD 가 직전 외부 write 상태에 stuck 될 수 있다 (hide_below
            # 무력화의 원인). visible=True → cache=hidden_False, False → True.
            try:
                prim_path = prim.GetPath().pathString
                cache = getattr(viz, "_threshold_hide_state", None)
                if cache is not None:
                    cache[prim_path] = (not visible)
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"[replay] per-prim visibility toggle warn: {exc}")

