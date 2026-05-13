"""Motor-domain PD gain mirror — orchestrator for per-step jk_kernel launches.

매 physics step 에 호출되어 모터-도메인 K_m / Kv_m / τ_m_max 와 현재 q 로
joint-domain K_j / D_j / τ_j_max 를 계산해 Newton MuJoCo 의 PD 게인 view
(joint_target_ke / joint_target_kd / joint_effort_limit) 에 직접 write.

데이터 흐름:
    nominal_motor_gains (joint_config.json)
        ↓ __init__ load
    K_m_current / Kv_m_current / trq_m_current  (host numpy + device warp)
        ↓ (set_target 시 ramp_target 갱신, update() 에서 motor_step 단위 step)
        ↓ wp.launch(jk_kernel.compute_*)
    joint_target_ke / joint_target_kd / joint_effort_limit  (Newton model views)
        ↓ solver._update_joint_dof_properties()
    actuator gainprm / biasprm  (mjw_model)
        ↓ mj_step
    sim 동작

이 클래스가 trajectory_player 와 독립 — sim loop 에서 항상 호출되어 nominal
모터 게인이 매 step Newton view 에 강제 mirror.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from . import jk_kernel
from .motor_joint_transform import (
    NECK_RATIOS,
    SHOULDER_RATIO,
    WAIST_RATIOS,
)


# ============================================================
# Static kernel layout — 어느 joint name 이 어느 kernel slot 에 들어가는지.
# jk_kernel.py 의 인덱싱 규약과 1:1 매칭되어야 함.
# ============================================================
SCALAR_MOTORS: tuple[str, ...] = (
    "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint",
    "Neck_Pitch_Joint", "Neck_Yaw_Joint",
)
SCALAR_RATIOS: tuple[float, ...] = (
    SHOULDER_RATIO, SHOULDER_RATIO, SHOULDER_RATIO,
    SHOULDER_RATIO, SHOULDER_RATIO, SHOULDER_RATIO,
    float(WAIST_RATIOS[0]), float(WAIST_RATIOS[1]),
    float(NECK_RATIOS[0]), float(NECK_RATIOS[1]),
)

# Elbow kernel: instance 0=L (Elbow + Wrist_Yaw), instance 1=R.
ELBOW_MOTORS: tuple[str, ...] = (
    "L_Elbow_Joint", "L_Wrist_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint",
)

# Wrist kernel: instance 0=L (Roll + Pitch), instance 1=R.
WRIST_MOTORS: tuple[str, ...] = (
    "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
)

# Finger kernel: 8 instances (L_Index/Middle/Ring/Little + R_*) × 3 motors each.
def _build_finger_motors() -> tuple[str, ...]:
    motors: list[str] = []
    for hand in ("L", "R"):
        for finger in ("Index", "Middle", "Ring", "Little"):
            for part in ("ABAD", "MCP", "PIP"):
                motors.append(f"{hand}_{finger}_{part}_Joint")
    return tuple(motors)


FINGER_MOTORS: tuple[str, ...] = _build_finger_motors()  # 24 motors

# Thumb kernel: 2 instances (L, R) × 3 motors (Yaw, CMC, MCP).
THUMB_MOTORS: tuple[str, ...] = (
    "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
)

# Motor-side Coulomb friction per actuator — sourced from allex_control's
# robot_model.json (v1).  Mirrors real controller: τ_m_actual = τ_m_clip
# - friction_coulomb_actu · tanh(qd_m · tanh_scale).  Applied inside the
# finger / thumb warp kernels (NOT via mjw_model.dof_frictionloss, which is
# joint-side).
# Order matches FINGER_MOTORS / THUMB_MOTORS.
FINGER_FRICTION_MOTOR: tuple[float, ...] = (
    # L_Index   ABAD,        MCP,         PIP
    0.0004,     0.00018,     0.00033,
    # L_Middle
    0.0005,     0.00015,     0.0004477,
    # L_Ring
    0.0005,     0.000101,    0.000333,
    # L_Little
    0.0005,     0.0001784,   0.0001681,
    # R_Index
    0.0006,     0.0001949,   0.0004424,
    # R_Middle
    0.0005236,  0.0001097,   0.0004,
    # R_Ring
    0.00035,    0.0001016,   0.000361,
    # R_Little
    0.0006,     0.0001167,   0.0003386,
)
FINGER_FRICTION_TANH_SCALE: tuple[float, ...] = (
    0.02, 0.02, 0.02,   0.02, 0.02, 0.02,   0.02, 0.02, 0.02,   0.02, 0.02, 0.02,   # L
    0.02, 0.02, 0.02,   0.02, 0.02, 0.02,   0.02, 0.02, 0.02,   0.02, 0.02, 0.02,   # R
)
THUMB_FRICTION_MOTOR: tuple[float, ...] = (
    # L: Yaw, CMC, MCP
    0.0002123, 0.0004, 0.0004,
    # R: Yaw, CMC, MCP
    0.00018,   0.0005, 0.0005,
)
THUMB_FRICTION_TANH_SCALE: tuple[float, ...] = (
    0.02, 0.02, 0.02,
    0.02, 0.02, 0.02,
)
assert len(FINGER_FRICTION_MOTOR) == len(FINGER_MOTORS) == 24
assert len(FINGER_FRICTION_TANH_SCALE) == 24
assert len(THUMB_FRICTION_MOTOR) == len(THUMB_MOTORS) == 6
assert len(THUMB_FRICTION_TANH_SCALE) == 6


# Total motor count sanity check: 10 + 4 + 4 + 24 + 6 = 48 (active_joints).
assert (
    len(SCALAR_MOTORS) + len(ELBOW_MOTORS) + len(WRIST_MOTORS)
    + len(FINGER_MOTORS) + len(THUMB_MOTORS) == 48
), "kernel layout motor count must equal active joint count"


@dataclass
class _KernelGroup:
    """Per-kernel state: dof_idx + K_m/Kv_m/trq_m device arrays + host shadows."""
    name: str                      # "scalar" / "elbow" / "wrist" / "finger" / "thumb"
    n_motors: int                  # total motors across all instances
    n_instances: int               # kernel launch dim
    dof_idx: wp.array              # int32 device, size = n_motors
    K_m: wp.array                  # float device
    Kv_m: wp.array                 # float device
    trq_m: wp.array                # float device
    # Host shadows: latest values that have been (or will be) uploaded.
    K_m_host: np.ndarray
    Kv_m_host: np.ndarray
    trq_m_host: np.ndarray
    motor_names: tuple[str, ...]   # joint name per slot (for _motor_index lookup)
    # Phase 2 ramp state.
    #   _target = where set_target() / reset_to_nominal() asks host to go;
    #   _nominal = immutable boot-time value used by reset_to_nominal().
    K_m_target: np.ndarray
    Kv_m_target: np.ndarray
    trq_m_target: np.ndarray
    K_m_nominal: np.ndarray
    Kv_m_nominal: np.ndarray
    trq_m_nominal: np.ndarray
    # Time-bounded ramp state (per-slot start value + start time at last
    # set_target). Host = lerp(start, target, alpha) where alpha = clip(
    # (t_now - start_t) / ramp_window_s, 0, 1).
    K_m_ramp_start:   np.ndarray = None  # type: ignore[assignment]
    Kv_m_ramp_start:  np.ndarray = None  # type: ignore[assignment]
    trq_m_ramp_start: np.ndarray = None  # type: ignore[assignment]
    K_m_ramp_t0:   np.ndarray = None  # type: ignore[assignment]
    Kv_m_ramp_t0:  np.ndarray = None  # type: ignore[assignment]
    trq_m_ramp_t0: np.ndarray = None  # type: ignore[assignment]
    # Per-slot ramp END time. K_m/Kv_m: t_end = t0 + ramp_window_s (time-bounded,
    # 'stretching' style). trq_m: t_end = t0 + |delta|/step_rate (step-based,
    # allex_control 0.001/ms style — fast for small delta, slow for large).
    K_m_ramp_t_end:   np.ndarray = None  # type: ignore[assignment]
    Kv_m_ramp_t_end:  np.ndarray = None  # type: ignore[assignment]
    trq_m_ramp_t_end: np.ndarray = None  # type: ignore[assignment]
    ratio: wp.array | None = None         # scalar group only — per-motor reduction ratio
    is_mech_comp: wp.array | None = None  # scalar group only — int per motor (1=mech, 0=actuator)


class MotorStateMirror:
    """Real-time motor→joint PD mirror via warp kernels.

    Args:
        articulation: ALLEX articulation (provides ``dof_names``).
        model: Newton Model object (with ``joint_q``, ``joint_target_ke`` etc).
        solver: Newton MuJoCo solver (for ``_update_joint_dof_properties``).
        device: warp device id, default 'cuda:0'.

    Lifecycle:
        - ``__init__``: layout 빌드 + nominal_motor_gains 로드 + device 할당
        - ``update()``: 매 step 호출 — host ramp + dirty 시 H2D + 모든 kernel launch + solver sync
        - ``set_target(joint_names, kps, kds, max_efforts)``: 모터-도메인 ramp target 갱신
          (host-only, NaN 슬롯은 skip). 다음 update() 가 host shadow 를 motor_step
          속도로 target 까지 끌어감.
        - ``reset_to_nominal()`` / ``reset()``: target → nominal. host 가 ramp 로 복귀.
    """

    def __init__(self, articulation, model, solver, stage=None, control=None,
                 get_target_vel_fn=None, device: str = "cuda:0"):
        self._articulation = articulation
        self._model = model
        self._solver = solver
        self._stage = stage  # Newton stage — for live state_0.joint_q / joint_qd
        self._control = control  # Newton Control — for joint_target_pos / joint_f
        # ArticulationAction 이 joint_velocities 를 Isaac Sim 의 articulation_controller
        # 로 보내면 CUDA tensor → np.isnan 변환 버그 때문에 터짐. 그래서 q̇_target 은
        # 외부 callback (scenario 가 trajectory_player.get_current_velocity_target 을
        # tolist 로 래핑) 으로 받아 매 step 우리가 직접 device buffer 에 업로드.
        self._get_target_vel_fn = get_target_vel_fn
        self._device = device

        dof_names = list(getattr(articulation, "dof_names", []) or [])
        if not dof_names:
            raise RuntimeError("articulation.dof_names empty — articulation not ready")
        self._dof_names = dof_names
        self._name_to_dof = {n: i for i, n in enumerate(dof_names)}
        self._num_dof = len(dof_names)

        # Load nominal motor gains (joint_config.json::nominal_motor_gains).
        from ..utils.sim_settings_utils import (
            get_motor_step_per_ms,
            get_nominal_motor_gains,
            _load as _load_physics_config,
        )
        self._nominal = get_nominal_motor_gains()
        if not self._nominal:
            raise RuntimeError("nominal_motor_gains not found in joint_config.json")

        # pd_mode from physics_config.json::newton.pd_mode.
        #   'motor_space'        → 0 (Y) — q_m=J·q 선형화 후 motor-space PD + motor clip
        #   'joint_space_diag'   → 1 (X) — joint diag K_j PD + J⁻ᵀ·motor-clip·Jᵀ
        #   'joint_nominal'      → 2     — USD-loaded K_j / Kv_j / effort_limit 그대로
        #                                  사용 (자세/이벤트 무관 고정 baseline), joint-space clip
        # Default 0 (motor_space).
        _newton_cfg = _load_physics_config().get("newton", {})
        pd_mode_str = str(_newton_cfg.get("pd_mode", "motor_space"))
        _pd_mode_map = {"motor_space": 0, "joint_space_diag": 1, "joint_nominal": 2}
        self._pd_mode = _pd_mode_map.get(pd_mode_str, 0)
        # 디버그 토글들:
        #  - disable_torque_clip: True (모든 group) / False (없음) /
        #    list[str] (해당 group 만). group ∈ {"scalar","elbow","wrist","finger","thumb"}.
        #    예: ["finger","thumb"] → 손가락/엄지 motor clip 만 무효화, 나머지는 정상.
        #  - joint_nominal_torque_source: "usd" (기본) 또는 "mjcf" — joint_nominal mode 만
        #  - pd_gain_scale: K_m / Kv_m / K_j_nom / Kv_j_nom 에 곱하는 scale.
        #    0.0 → PD 사실상 off (gravcomp only 진단용). 1.0 → 정상.
        _ALL_GROUPS = {"scalar", "elbow", "wrist", "finger", "thumb"}
        _cfg_clip = _newton_cfg.get("disable_torque_clip", False)
        if isinstance(_cfg_clip, bool):
            self._disable_clip_groups = set(_ALL_GROUPS) if _cfg_clip else set()
        elif isinstance(_cfg_clip, (list, tuple)):
            self._disable_clip_groups = {str(g) for g in _cfg_clip} & _ALL_GROUPS
            _unknown = {str(g) for g in _cfg_clip} - _ALL_GROUPS
            if _unknown:
                print(f"[ALLEX][Mirror] disable_torque_clip: unknown groups ignored: {_unknown}")
        else:
            raise RuntimeError(
                f"disable_torque_clip must be bool or list of group names, "
                f"got {type(_cfg_clip).__name__}"
            )
        # Backward-compat alias used by other passes (joint_nominal etc).
        self._disable_clip = bool(self._disable_clip_groups)
        self._jn_torque_source = str(
            _newton_cfg.get("joint_nominal_torque_source", "usd")
        ).lower()
        # pd_gain_scale (legacy) = single scalar applied to both kp & kv.
        # kp_scale / kv_scale (preferred) = independent scales. If either is set,
        # they override pd_gain_scale for that category.
        _pd_legacy = float(_newton_cfg.get("pd_gain_scale", 1.0))
        self._kp_scale = float(_newton_cfg.get("kp_scale", _pd_legacy))
        self._kv_scale = float(_newton_cfg.get("kv_scale", _pd_legacy))
        # Keep _pd_gain_scale name for callers that still read it (informational).
        self._pd_gain_scale = _pd_legacy
        # Per-joint scale overrides: dict {substring: scale}. Longest substring
        # match wins. Falls back to global kp_scale/kv_scale if no key in name.
        # Example: {"Wrist": 2.0, "Waist_Yaw": 1.5} — boosts wrists 2x, waist_yaw 1.5x.
        self._kp_overrides = dict(_newton_cfg.get("kp_scale_overrides", {}))
        self._kv_overrides = dict(_newton_cfg.get("kv_scale_overrides", {}))
        # Ramp mode per category — matches real allex_control:
        #   K_m, Kv_m: time-bounded over `ramp_window_s` (stretching-section behavior).
        #   trq_m:     step-based at `motor_step_per_ms` per ms (allex_control
        #              style, time = |delta| / (step_per_ms·1000) seconds).
        # ramp_window_s=0 → snap K/Kv. motor_step_per_ms=0 → snap trq.
        self._ramp_window_s     = float(_newton_cfg.get("ramp_window_s", 1.0))
        self._motor_step_per_ms = float(_newton_cfg.get("motor_step_per_ms", 0.001))
        # Step time tracking (used for ramp_t0). dt from physics_hz (world cfg).
        try:
            from ..utils.sim_settings_utils import get_physics_hz
            _hz = float(get_physics_hz() or 1000.0)
        except Exception:
            _hz = 1000.0
        self._step_dt = 1.0 / _hz
        self._step_t = 0.0
        # friction / armature scaling — dynamics 검증용 (default 1.0 = MJCF 그대로).
        # 0.0 → 해당 효과 완전 제거 (마찰 / 모터 회전관성).
        self._friction_scale = float(_newton_cfg.get("friction_scale", 1.0))
        # Per-joint absolute frictionloss override (applied after scale).
        # Used when MJCF/USD lacks friction values (e.g., Waist_Yaw) but real
        # joint has friction. dict {joint_name: frictionloss [N·m]}.
        self._friction_overrides = dict(_newton_cfg.get("friction_overrides", {}))
        self._armature_scale = float(_newton_cfg.get("armature_scale", 1.0))
        # Per-joint absolute joint-level damping (MuJoCo dof_damping). Applies
        # `-d·qd` purely from velocity — independent of PD's Kv·(vt-v) tracking.
        # MJCF default damping=0 for all ALLEX joints; set here to add passive
        # natural damping (e.g. wrist). dict {joint_name: damping [N·m·s/rad]}.
        self._damping_overrides = dict(_newton_cfg.get("damping_overrides", {}))
        flags = []
        if self._disable_clip_groups:
            if self._disable_clip_groups == _ALL_GROUPS:
                flags.append("disable_torque_clip=ALL")
            else:
                flags.append(f"disable_torque_clip={sorted(self._disable_clip_groups)}")
        if self._jn_torque_source != "usd":
            flags.append(f"torque_source={self._jn_torque_source}")
        if self._kp_scale != 1.0 or self._kv_scale != 1.0:
            flags.append(f"kp_scale={self._kp_scale} kv_scale={self._kv_scale}")
        if self._kp_overrides:
            flags.append(f"kp_overrides={self._kp_overrides}")
        if self._kv_overrides:
            flags.append(f"kv_overrides={self._kv_overrides}")
        if self._friction_scale != 1.0:
            flags.append(f"friction_scale={self._friction_scale}")
        if self._friction_overrides:
            flags.append(f"friction_overrides={self._friction_overrides}")
        if self._damping_overrides:
            flags.append(f"damping_overrides={self._damping_overrides}")
        if self._armature_scale != 1.0:
            flags.append(f"armature_scale={self._armature_scale}")
        flag_str = (" ⚠ " + " | ".join(flags)) if flags else ""
        print(f"[ALLEX][Mirror] pd_mode='{pd_mode_str}' ({self._pd_mode}){flag_str}")

        # Cache motor-domain ramp step (per 1ms). 0.0 → snap mode (no ramp).
        self._motor_step = float(get_motor_step_per_ms() or 0.0)
        if self._motor_step <= 0.0:
            print("[ALLEX][Mirror] motor_step_per_ms <= 0 → snap mode (host=target instantly)")

        # Build per-kernel groups.
        # is_mech_comp 의 의미: 1 = 기구적 중력보상 (motor clip 시 gravcomp skip)
        #                     0 = actuator 중력보상 (motor clip 시 gravcomp 합산)
        # Waist_Lower_Pitch / Neck_Pitch 만 1, 나머지 8 scalar motor 는 0.
        scalar_mech_comp = tuple(
            1 if name in ("Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint") else 0
            for name in SCALAR_MOTORS
        )
        self._scalar = self._build_group("scalar", SCALAR_MOTORS, n_per_inst=1,
                                         ratios=SCALAR_RATIOS,
                                         is_mech_comp=scalar_mech_comp)
        self._elbow  = self._build_group("elbow", ELBOW_MOTORS, n_per_inst=2)
        self._wrist  = self._build_group("wrist", WRIST_MOTORS, n_per_inst=2)
        self._finger = self._build_group("finger", FINGER_MOTORS, n_per_inst=3)
        self._thumb  = self._build_group("thumb", THUMB_MOTORS, n_per_inst=3)
        self._groups: tuple[_KernelGroup, ...] = (
            self._scalar, self._elbow, self._wrist, self._finger, self._thumb,
        )
        # Motor-side Coulomb friction device buffers (finger + thumb only).
        # Values from allex_control/config/asset/model/v1/robot_model.json.
        self._finger_friction_motor = wp.array(
            np.asarray(FINGER_FRICTION_MOTOR, dtype=np.float32), dtype=float, device=device)
        self._finger_friction_tanh_scl = wp.array(
            np.asarray(FINGER_FRICTION_TANH_SCALE, dtype=np.float32), dtype=float, device=device)
        self._thumb_friction_motor = wp.array(
            np.asarray(THUMB_FRICTION_MOTOR, dtype=np.float32), dtype=float, device=device)
        self._thumb_friction_tanh_scl = wp.array(
            np.asarray(THUMB_FRICTION_TANH_SCALE, dtype=np.float32), dtype=float, device=device)

        # disable_torque_clip: 선택된 group 의 motor τ_m_max 를 1e6 으로 override.
        # motor_space / joint_space_diag 의 motor-space clip 이 해당 group 에서 사실상 무효화.
        # (joint_nominal mode 는 별도로 _build_joint_nominal_buffers 에서 슬롯 범위 적용.)
        # via event (set_target) 가 host_target 을 줄이려 해도 motor_step_per_ms=0.001 기준
        # 1e6 → 작은 값까지 ramp 시간이 trajectory 길이를 훨씬 초과해 사실상 영구.
        if self._disable_clip_groups:
            for grp in self._groups:
                if grp.name not in self._disable_clip_groups:
                    continue
                grp.trq_m_host[:] = 1.0e6
                grp.trq_m_target[:] = 1.0e6
                grp.trq_m_nominal[:] = 1.0e6
                grp.trq_m.assign(grp.trq_m_host)
            print(f"[ALLEX][Mirror] disable_torque_clip groups={sorted(self._disable_clip_groups)} "
                  f"— motor trq_m overridden to 1e6 for listed groups only")

        # kp_scale / kv_scale (global + per-joint overrides). 0.0 → off.
        # joint_nominal mode 의 K_j_nom / Kv_j_nom 은 _build_joint_nominal_buffers 에서
        # 별도로 곱함.
        for grp in self._groups:
            kp_scales = np.array([self._kp_scale_for(nm) for nm in grp.motor_names],
                                 dtype=np.float32)
            kv_scales = np.array([self._kv_scale_for(nm) for nm in grp.motor_names],
                                 dtype=np.float32)
            grp.K_m_host    *= kp_scales
            grp.K_m_target  *= kp_scales
            grp.K_m_nominal *= kp_scales
            grp.Kv_m_host   *= kv_scales
            grp.Kv_m_target *= kv_scales
            grp.Kv_m_nominal*= kv_scales
            self._upload_group(grp)
        if (self._kp_scale != 1.0 or self._kv_scale != 1.0
                or self._kp_overrides or self._kv_overrides):
            print(f"[ALLEX][Mirror] kp_scale={self._kp_scale}, kv_scale={self._kv_scale}, "
                  f"kp_overrides={self._kp_overrides}, kv_overrides={self._kv_overrides}")

        # q̇_target device buffer — callback 으로 받은 tolist() 값을 매 step assign.
        # control.joint_target_vel 을 fallback 으로 둠 (대개 0 이라 idle 시 안전).
        self._qdt_buf = wp.zeros(self._num_dof, dtype=float, device=device)

        # joint_nominal mode count (실제 buffer 빌드는 model views 획득 후 아래에서).
        self._jn_count = 0

        # Singular monitoring — |det(J)| device buffers per coupled group.
        # scalar 은 J = ratio 라 항상 비-singular, det monitoring 불필요.
        self._elbow_det  = wp.zeros(self._elbow.n_instances,  dtype=float, device=device)
        self._wrist_det  = wp.zeros(self._wrist.n_instances,  dtype=float, device=device)
        self._finger_det = wp.zeros(self._finger.n_instances, dtype=float, device=device)
        self._thumb_det  = wp.zeros(self._thumb.n_instances,  dtype=float, device=device)
        # |det| threshold per group: typical magnitude × 1e-3. 측정값 기반으로 조정 가능.
        # 일단 ad-hoc: 매우 작은 값 (1e-6) 만 singular 로 카운트.
        self._det_threshold = 1e-6
        # Cumulative singular event counts (host-side).
        self._singular_count = {"elbow": 0, "wrist": 0, "finger": 0, "thumb": 0}
        # Throttle singular log (avoid spam at 1kHz).
        self._singular_check_interval = 100   # check every N steps
        self._step_counter = 0

        # Flat name → (group, slot) lookup for set_target hot-path. O(1).
        self._motor_index: dict[str, tuple[_KernelGroup, int]] = {}
        for grp in self._groups:
            for slot, jname in enumerate(grp.motor_names):
                self._motor_index[jname] = (grp, slot)

        # Diagnostics.
        self._set_target_logged = False
        self._set_target_unmatched = 0

        # Acquire zero-copy views of Newton model arrays.
        self._joint_q = getattr(model, "joint_q", None)
        self._out_ke = getattr(model, "joint_target_ke", None)
        self._out_kd = getattr(model, "joint_target_kd", None)
        self._out_eff = getattr(model, "joint_effort_limit", None)
        for arr_name, arr in [("joint_q", self._joint_q),
                              ("joint_target_ke", self._out_ke),
                              ("joint_target_kd", self._out_kd),
                              ("joint_effort_limit", self._out_eff)]:
            if arr is None:
                raise RuntimeError(f"Newton model.{arr_name} not available")

        # joint_nominal mode 용 flat buffer — USD-loaded K_j / Kv_j / effort_limit 그대로.
        # 반드시 view acquire 이후, 첫 update() 호출 (== 첫 kernel launch) 이전에 실행.
        # 우리 kernel 이 ke/kd 를 0 으로 덮어쓰기 때문에 init 시점의 USD 값만 유효.
        self._build_joint_nominal_buffers()

        # FK polynomial buffers — joint→motor 변환 (allex_control 매퍼와 동일).
        # Y(motor_space) 모드의 position err 계산에만 사용 (velocity / torque 는 J(q) 유지).
        self._build_fk_buffers()

        # friction / armature scaling (dynamics 검증용) — MuJoCo Warp 의 dof_frictionloss /
        # dof_armature 배열에 직접 곱해 모델 변경. 0.0 = 효과 제거.
        # 항상 호출해 캐시된 원본 기준으로 복원-혹은-스케일 (1.0 도 idempotent 복원 효과).
        self._apply_dynamics_scale()

        # Cache solver sync handle.
        self._sync_dof_props = getattr(solver, "_update_joint_dof_properties", None)
        self._notify_solver = None
        self._notify_flag = 0
        if self._sync_dof_props is None:
            try:
                from newton.solvers import SolverNotifyFlags
                self._notify_solver = solver.notify_model_changed
                self._notify_flag = int(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
            except Exception as exc:
                print(f"[ALLEX][Mirror] no solver sync available: {exc}")

        sync_mode = ("_update_joint_dof_properties"
                     if self._sync_dof_props is not None
                     else f"notify_model_changed(flag={self._notify_flag})")
        print(
            f"[ALLEX][Mirror] initialized — "
            f"scalar={self._scalar.n_motors} elbow={self._elbow.n_motors} "
            f"wrist={self._wrist.n_motors} finger={self._finger.n_motors} "
            f"thumb={self._thumb.n_motors}; "
            f"device={device}, sync={sync_mode}"
        )

    # --------------------------------------------------------
    # Layout build
    # --------------------------------------------------------
    def _build_group(
        self,
        name: str,
        motor_names: tuple[str, ...],
        *,
        n_per_inst: int,
        ratios: tuple[float, ...] | None = None,
        is_mech_comp: tuple[int, ...] | None = None,
    ) -> _KernelGroup:
        """Build per-kernel state from static motor name layout."""
        n_motors = len(motor_names)
        if n_motors % n_per_inst != 0:
            raise RuntimeError(
                f"{name} motor count {n_motors} not divisible by n_per_inst={n_per_inst}"
            )
        n_instances = n_motors // n_per_inst

        # Resolve articulation DOF index for each motor slot.
        dof_idx_np = np.empty(n_motors, dtype=np.int32)
        kp_host = np.empty(n_motors, dtype=np.float32)
        kv_host = np.empty(n_motors, dtype=np.float32)
        trq_host = np.empty(n_motors, dtype=np.float32)
        missing: list[str] = []
        for slot, jname in enumerate(motor_names):
            dof = self._name_to_dof.get(jname)
            if dof is None:
                missing.append(jname)
                dof_idx_np[slot] = -1
            else:
                dof_idx_np[slot] = int(dof)
            entry = self._nominal.get(jname)
            if entry is None:
                kp_host[slot] = 0.0
                kv_host[slot] = 0.0
                trq_host[slot] = 0.0
            else:
                kp_host[slot] = float(entry.get("kp", 0.0))
                kv_host[slot] = float(entry.get("kv", 0.0))
                trq_host[slot] = float(entry.get("trq_lim", 0.0))
        if missing:
            raise RuntimeError(
                f"{name} kernel: joint names not found in articulation.dof_names: {missing}"
            )

        # Allocate device arrays.
        dof_idx_dev = wp.array(dof_idx_np, dtype=int, device=self._device)
        K_m_dev = wp.array(kp_host, dtype=float, device=self._device)
        Kv_m_dev = wp.array(kv_host, dtype=float, device=self._device)
        trq_m_dev = wp.array(trq_host, dtype=float, device=self._device)

        ratio_dev: wp.array | None = None
        if ratios is not None:
            if len(ratios) != n_motors:
                raise RuntimeError(
                    f"{name} kernel: ratio size {len(ratios)} != n_motors {n_motors}"
                )
            ratio_dev = wp.array(np.asarray(ratios, dtype=np.float32),
                                 dtype=float, device=self._device)

        is_mech_dev: wp.array | None = None
        if is_mech_comp is not None:
            if len(is_mech_comp) != n_motors:
                raise RuntimeError(
                    f"{name} kernel: is_mech_comp size {len(is_mech_comp)} != n_motors {n_motors}"
                )
            is_mech_dev = wp.array(np.asarray(is_mech_comp, dtype=np.int32),
                                   dtype=int, device=self._device)

        return _KernelGroup(
            name=name,
            n_motors=n_motors,
            n_instances=n_instances,
            dof_idx=dof_idx_dev,
            K_m=K_m_dev,
            Kv_m=Kv_m_dev,
            trq_m=trq_m_dev,
            K_m_host=kp_host,
            Kv_m_host=kv_host,
            trq_m_host=trq_host,
            motor_names=tuple(motor_names),
            K_m_target=kp_host.copy(),
            Kv_m_target=kv_host.copy(),
            trq_m_target=trq_host.copy(),
            K_m_nominal=kp_host.copy(),
            Kv_m_nominal=kv_host.copy(),
            trq_m_nominal=trq_host.copy(),
            # Ramp state: start = host (so first ramp from boot is no-op until target changes)
            K_m_ramp_start=kp_host.copy(),
            Kv_m_ramp_start=kv_host.copy(),
            trq_m_ramp_start=trq_host.copy(),
            K_m_ramp_t0=np.zeros_like(kp_host),
            Kv_m_ramp_t0=np.zeros_like(kv_host),
            trq_m_ramp_t0=np.zeros_like(trq_host),
            K_m_ramp_t_end=np.zeros_like(kp_host),
            Kv_m_ramp_t_end=np.zeros_like(kv_host),
            trq_m_ramp_t_end=np.zeros_like(trq_host),
            ratio=ratio_dev,
            is_mech_comp=is_mech_dev,
        )

    # --------------------------------------------------------
    # joint_nominal mode pre-compute (USD-loaded values)
    # --------------------------------------------------------
    @staticmethod
    def _load_mjcf_actuator_frcrange() -> "dict[str, float] | None":
        """asset/ALLEX/mjcf/ALLEX.xml 에서 <joint actuatorfrcrange="-X X"> 추출.

        Returns: {joint_name: |peak_torque|} or None if file missing / parse 실패.
        """
        from pathlib import Path
        import xml.etree.ElementTree as ET
        # motor_state_mirror.py 위치: <repo>/src/allex/trajectory_generate/...
        mjcf_path = Path(__file__).resolve().parents[3] / "asset/ALLEX/mjcf/ALLEX.xml"
        if not mjcf_path.is_file():
            print(f"[ALLEX][Mirror] MJCF not found at {mjcf_path}")
            return None
        try:
            tree = ET.parse(mjcf_path)
        except Exception as exc:
            print(f"[ALLEX][Mirror] MJCF parse failed: {exc}")
            return None
        out: dict[str, float] = {}
        for joint in tree.iter("joint"):
            name = joint.get("name")
            frcrange = joint.get("actuatorfrcrange")
            if not name or not frcrange:
                continue
            parts = frcrange.strip().split()
            if len(parts) == 2:
                try:
                    out[name] = abs(float(parts[1]))
                except ValueError:
                    pass
        print(f"[ALLEX][Mirror] MJCF actuatorfrcrange loaded: {len(out)} joints")
        return out

    def _build_joint_nominal_buffers(self) -> None:
        """USD-loaded K_j / Kv_j / effort_limit 을 그대로 nominal 값으로 채택.

        제조사가 "적당한 자세" 에서 stiffness/effort_limit 을 보정해서 USD 에
        박아둔 값. 우리가 J(q=0) 기반으로 재계산할 수도 있으나, scalar / elbow /
        wrist 는 USD 와 거의 일치하고 finger / thumb 는 USD 가 q≠0 기준이라 더
        정확. 그래서 USD 값을 그대로 사용.

        Flat layout (kernel pd_clip_joint_nominal 의 tid 순서):
            scalar (10) + elbow (4) + wrist (4) + finger (24) + thumb (6) = 48
        """
        try:
            usd_ke  = self._out_ke.numpy()     # (n_dof,) USD-loaded joint_target_ke
            usd_kd  = self._out_kd.numpy()     # joint_target_kd
            usd_eff = self._out_eff.numpy()    # joint_effort_limit
        except Exception as exc:
            print(f"[ALLEX][Mirror] USD value readback failed: {exc}; joint_nominal disabled")
            return

        MECH_JOINTS = ("Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint")
        order = (
            SCALAR_MOTORS + ELBOW_MOTORS + WRIST_MOTORS
            + FINGER_MOTORS + THUMB_MOTORS
        )

        n = len(order)
        dof_idx_h = np.zeros(n, dtype=np.int32)
        is_mech_h = np.zeros(n, dtype=np.int32)
        K_j_h     = np.zeros(n, dtype=np.float32)
        Kv_j_h    = np.zeros(n, dtype=np.float32)
        tau_j_h   = np.zeros(n, dtype=np.float32)

        for slot, name in enumerate(order):
            dof = self._name_to_dof.get(name)
            if dof is None:
                print(f"[ALLEX][Mirror] joint_nominal: '{name}' dof missing; skip")
                continue
            dof_idx_h[slot] = int(dof)
            is_mech_h[slot] = 1 if name in MECH_JOINTS else 0
            K_j_h[slot]     = float(usd_ke[dof])
            Kv_j_h[slot]    = float(usd_kd[dof])
            tau_j_h[slot]   = float(usd_eff[dof])

        # kp_scale / kv_scale 적용 (joint_nominal mode 용)
        if self._kp_scale != 1.0:
            K_j_h  *= self._kp_scale
        if self._kv_scale != 1.0:
            Kv_j_h *= self._kv_scale
        if self._kp_scale != 1.0 or self._kv_scale != 1.0:
            print(f"[ALLEX][Mirror] joint_nominal: K_j_nom×{self._kp_scale} / Kv_j_nom×{self._kv_scale}")

        # MJCF peak override (joint_nominal_torque_source ∈ {"mjcf_peak", "mjcf", "peak"})
        if self._jn_torque_source in ("mjcf_peak", "mjcf", "peak"):
            mjcf_limits = self._load_mjcf_actuator_frcrange()
            if mjcf_limits is None:
                print("[ALLEX][Mirror] mjcf_peak requested but MJCF unavailable; using USD values")
            else:
                overridden = 0
                not_found: list[str] = []
                for slot, name in enumerate(order):
                    if name in mjcf_limits:
                        tau_j_h[slot] = mjcf_limits[name]
                        overridden += 1
                    else:
                        not_found.append(name)
                print(f"[ALLEX][Mirror] τ_j_max_nom overridden from MJCF actuatorfrcrange: "
                      f"{overridden}/{len(order)}")
                if not_found:
                    print(f"  USD value retained for missing-from-MJCF: {not_found}")

        # disable_torque_clip 은 모든 override 의 마지막 적용 (debug 최우선).
        # Flat slot 순서는 `order` 와 동일 (scalar→elbow→wrist→finger→thumb).
        if self._disable_clip_groups:
            _slot_ranges = (
                ("scalar", 0,                              len(SCALAR_MOTORS)),
                ("elbow",  len(SCALAR_MOTORS),             len(SCALAR_MOTORS) + len(ELBOW_MOTORS)),
                ("wrist",  len(SCALAR_MOTORS) + len(ELBOW_MOTORS),
                            len(SCALAR_MOTORS) + len(ELBOW_MOTORS) + len(WRIST_MOTORS)),
                ("finger", len(SCALAR_MOTORS) + len(ELBOW_MOTORS) + len(WRIST_MOTORS),
                            len(SCALAR_MOTORS) + len(ELBOW_MOTORS) + len(WRIST_MOTORS) + len(FINGER_MOTORS)),
                ("thumb",  len(SCALAR_MOTORS) + len(ELBOW_MOTORS) + len(WRIST_MOTORS) + len(FINGER_MOTORS),
                            len(order)),
            )
            for grp_name, lo, hi in _slot_ranges:
                if grp_name in self._disable_clip_groups:
                    tau_j_h[lo:hi] = 1.0e6
            print(f"[ALLEX][Mirror] joint_nominal: τ_j_max_nom overridden to 1e6 for groups "
                  f"{sorted(self._disable_clip_groups)}")

        self._jn_dof_idx   = wp.array(dof_idx_h, dtype=int,   device=self._device)
        self._jn_is_mech   = wp.array(is_mech_h, dtype=int,   device=self._device)
        self._jn_K_j_nom   = wp.array(K_j_h,     dtype=float, device=self._device)
        self._jn_Kv_j_nom  = wp.array(Kv_j_h,    dtype=float, device=self._device)
        self._jn_tau_j_nom = wp.array(tau_j_h,   dtype=float, device=self._device)
        self._jn_count     = n

        # 진단 print — USD 값이 우리가 알고 있는 대표 joint 값들과 일치하는지 user 가
        # 확인할 수 있도록 sample dump.
        print(f"[ALLEX][Mirror] joint_nominal buffers built (n={n}, USD-loaded)")
        sample_names = (
            "L_Shoulder_Pitch_Joint", "L_Elbow_Joint", "L_Wrist_Roll_Joint",
            "L_Index_MCP_Joint", "L_Index_PIP_Joint",
            "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
            "Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint",
        )
        for nm in sample_names:
            try:
                idx = order.index(nm)
                print(f"  {nm:30s}: K_j={K_j_h[idx]:9.3f} Kv_j={Kv_j_h[idx]:8.4f} "
                      f"τ_j_max={tau_j_h[idx]:8.4f} mech={is_mech_h[idx]}")
            except ValueError:
                pass

    # --------------------------------------------------------
    # FK polynomial buffers
    # --------------------------------------------------------
    def _build_fk_buffers(self) -> None:
        """fk_coeffs.py 의 (exp, coef) 테이블을 warp 배열로 빌드.

        wrist (qr, qp → 2 motors): FK_WRIST_R 만 사용 (L/R 동일 검증됨).
        finger (q1, q2, q3 → 3 motors): FK_FINGER_JANGHWAN (8 finger 공유).
        thumb (q1, q2, q3 → 3 motors): FK_THUMB_R (L/R 동일 검증됨).

        Layout: offsets[n_out+1] 가 flat 배열을 출력별로 slice.
        """
        import warp as wp
        import numpy as np

        try:
            from . import fk_coeffs
        except Exception as exc:
            raise RuntimeError(
                "fk_coeffs.py not found — run tools/extract_fk_polynomials.py first"
            ) from exc

        device = self._device

        def _flatten_2d(table):
            offsets = [0]
            exp_qr, exp_qp, coef = [], [], []
            for terms in table["outputs"]:
                for (e1, e2), c in terms:
                    exp_qr.append(e1); exp_qp.append(e2); coef.append(c)
                offsets.append(len(coef))
            return (
                wp.array(np.asarray(offsets,  dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(exp_qr,   dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(exp_qp,   dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(coef,     dtype=np.float32), dtype=float, device=device),
            )

        def _flatten_3d(table):
            offsets = [0]
            e1, e2, e3, coef = [], [], [], []
            for terms in table["outputs"]:
                for (a1, a2, a3), c in terms:
                    e1.append(a1); e2.append(a2); e3.append(a3); coef.append(c)
                offsets.append(len(coef))
            return (
                wp.array(np.asarray(offsets, dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(e1,      dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(e2,      dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(e3,      dtype=np.int32),   dtype=int,   device=device),
                wp.array(np.asarray(coef,    dtype=np.float32), dtype=float, device=device),
            )

        (self._wrist_poly_off,
         self._wrist_poly_exp_qr,
         self._wrist_poly_exp_qp,
         self._wrist_poly_coef) = _flatten_2d(fk_coeffs.FK_WRIST_R)

        (self._finger_poly_off,
         self._finger_poly_exp_q1,
         self._finger_poly_exp_q2,
         self._finger_poly_exp_q3,
         self._finger_poly_coef) = _flatten_3d(fk_coeffs.FK_FINGER_JANGHWAN)

        (self._thumb_poly_off,
         self._thumb_poly_exp_q1,
         self._thumb_poly_exp_q2,
         self._thumb_poly_exp_q3,
         self._thumb_poly_coef) = _flatten_3d(fk_coeffs.FK_THUMB_R)

        print(
            f"[ALLEX][Mirror] FK polynomial loaded — "
            f"wrist={self._wrist_poly_coef.shape[0]} terms, "
            f"finger={self._finger_poly_coef.shape[0]} terms, "
            f"thumb={self._thumb_poly_coef.shape[0]} terms"
        )

    # --------------------------------------------------------
    # Dynamics scaling (debug)
    # --------------------------------------------------------
    def _apply_dynamics_scale(self) -> None:
        """solver.mjw_model.dof_frictionloss / dof_armature 에 scale 곱하기 (idempotent).

        MJCF 의 frictionloss / armature 값을 그대로 두되 sim runtime 에서 scale.
        friction_scale=0.0 → 마찰 완전 제거. armature_scale=0.0 → motor inertia 제거.

        Idempotent: 원본을 mjw_model 인스턴스에 캐싱해두고 매번 원본 × scale 로 덮어쓰기.
        scenario.reset() 후 mirror 재빌드 시 compound 되지 않음. LOAD 로 mjw_model 자체가
        새로 만들어지면 캐시가 없으므로 새 원본 기준으로 다시 캐싱.
        """
        mjw_model = getattr(self._solver, "mjw_model", None)
        if mjw_model is None:
            print("[ALLEX][Mirror] mjw_model not available; cannot scale dynamics")
            return
        for attr_name, scale in (("dof_frictionloss", self._friction_scale),
                                  ("dof_armature", self._armature_scale)):
            arr = getattr(mjw_model, attr_name, None)
            if arr is None:
                print(f"[ALLEX][Mirror] mjw_model.{attr_name} missing; skip")
                continue
            cache_attr = f"_allex_orig_{attr_name}"
            try:
                if not hasattr(mjw_model, cache_attr):
                    # 첫 접근 — 원본 캐싱
                    setattr(mjw_model, cache_attr, arr.numpy().copy())
                orig = getattr(mjw_model, cache_attr)
                new_arr = orig * scale
                arr.assign(new_arr)
                print(f"[ALLEX][Mirror] mjw_model.{attr_name} × {scale} applied "
                      f"(max={float(np.abs(new_arr).max()):.4f}, "
                      f"orig_max={float(np.abs(orig).max()):.4f})")
            except Exception as exc:
                print(f"[ALLEX][Mirror] dynamics scale failed for {attr_name}: {exc}")

        # Per-joint friction override — set absolute frictionloss for named joints
        # (e.g., Waist_Yaw not present in MJCF/USD). Applied AFTER scale, so override
        # wins. Cache scaled result first, then patch overrides.
        friction_overrides = dict(self._friction_overrides)
        if friction_overrides:
            arr = getattr(mjw_model, "dof_frictionloss", None)
            if arr is not None:
                try:
                    cur = arr.numpy().copy()
                    for name, val in friction_overrides.items():
                        dof = self._name_to_dof.get(name)
                        if dof is None:
                            print(f"[ALLEX][Mirror] friction_overrides: '{name}' dof missing; skip")
                            continue
                        cur[dof] = float(val)
                        print(f"[ALLEX][Mirror] friction_overrides: {name} → {val}")
                    arr.assign(cur)
                except Exception as exc:
                    print(f"[ALLEX][Mirror] friction_overrides failed: {exc}")

        # Per-joint damping override — MuJoCo applies `-d·qd` to each DOF as a
        # passive joint-level dissipative force (independent of PD).  MJCF
        # default `damping="0"` for all ALLEX joints, so without this override
        # there's no natural damping; PD's Kv only damps target-tracking error.
        damping_overrides = dict(self._damping_overrides)
        if damping_overrides:
            arr = getattr(mjw_model, "dof_damping", None)
            if arr is None:
                print("[ALLEX][Mirror] mjw_model.dof_damping missing; cannot apply damping_overrides")
            else:
                try:
                    cur = arr.numpy().copy()
                    for name, val in damping_overrides.items():
                        dof = self._name_to_dof.get(name)
                        if dof is None:
                            print(f"[ALLEX][Mirror] damping_overrides: '{name}' dof missing; skip")
                            continue
                        cur[dof] = float(val)
                        print(f"[ALLEX][Mirror] damping_overrides: {name} → {val} N·m·s/rad")
                    arr.assign(cur)
                except Exception as exc:
                    print(f"[ALLEX][Mirror] damping_overrides failed: {exc}")

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _get_joint_q(self):
        """현재 simulation q warp array 반환 — stage.state_0.joint_q 우선."""
        if self._stage is not None:
            state = getattr(self._stage, "state_0", None)
            if state is not None:
                jq = getattr(state, "joint_q", None)
                if jq is not None:
                    return jq
        return self._joint_q  # fallback: model.joint_q (initial pose)

    def _get_joint_qd(self):
        """현재 simulation q̇ warp array 반환 — stage.state_0.joint_qd 우선."""
        if self._stage is not None:
            state = getattr(self._stage, "state_0", None)
            if state is not None:
                jqd = getattr(state, "joint_qd", None)
                if jqd is not None:
                    return jqd
        # fallback: model.joint_qd (initial all-zero)
        return getattr(self._model, "joint_qd", None)

    def _get_target_pos(self):
        """ArticulationAction 이 채워둔 control.joint_target_pos 반환."""
        if self._control is not None:
            return getattr(self._control, "joint_target_pos", None)
        return None

    def _get_target_vel(self):
        """q̇_target 1D wp.array 반환.

        callback 이 있으면 (scenario 가 trajectory_player 값을 list 로 래핑해서 넘김)
        그것을 device buffer 에 assign. None 이면 0 으로 채워 hold-pose 동작.
        callback 자체가 없으면 control.joint_target_vel 을 그대로 사용 (대개 0).
        """
        if self._get_target_vel_fn is not None:
            try:
                vel = self._get_target_vel_fn()
            except Exception as exc:
                if not getattr(self, "_warned_vel_cb", False):
                    print(f"[ALLEX][Mirror] target_vel callback raised: {exc}; using zeros")
                    self._warned_vel_cb = True
                vel = None
            if vel is None:
                self._qdt_buf.zero_()
                return self._qdt_buf
            try:
                np_arr = np.asarray(vel, dtype=np.float32)
            except Exception:
                self._qdt_buf.zero_()
                return self._qdt_buf
            if np_arr.size != self._num_dof:
                if not getattr(self, "_warned_vel_size", False):
                    print(f"[ALLEX][Mirror] target_vel size {np_arr.size} != num_dof "
                          f"{self._num_dof}; ignoring")
                    self._warned_vel_size = True
                self._qdt_buf.zero_()
                return self._qdt_buf
            self._qdt_buf.assign(np_arr)
            return self._qdt_buf
        # Fallback: Newton control's velocity target (usually all zero now).
        if self._control is not None:
            return getattr(self._control, "joint_target_vel", None)
        return None

    def _get_joint_f(self):
        """우리 토크 출력 path — control.joint_f (Newton → mjw_data.qfrc_applied)."""
        if self._control is not None:
            return getattr(self._control, "joint_f", None)
        return None

    def _get_gravcomp_world0(self):
        """solver.mjw_data.qfrc_gravcomp 의 world 0 slice (1D wp.array) 반환.

        qfrc_gravcomp 는 shape (nworld, nv). nworld=1 환경에서 world 0 만 사용.
        매 step 같은 underlying 메모리이므로 첫 호출 시 view 만 caching.
        """
        if not hasattr(self, "_gravcomp_cache_src"):
            self._gravcomp_cache_src = None
            self._gravcomp_cache_view = None
        mjw = getattr(self._solver, "mjw_data", None)
        if mjw is None:
            return None
        g2d = getattr(mjw, "qfrc_gravcomp", None)
        if g2d is None:
            return None
        if g2d is not self._gravcomp_cache_src:
            try:
                t = wp.to_torch(g2d)            # shape (nworld, nv)
                self._gravcomp_cache_view = wp.from_torch(t[0].contiguous())
                self._gravcomp_cache_src = g2d
            except Exception as exc:
                print(f"[ALLEX][Mirror] qfrc_gravcomp slice failed: {exc}")
                return None
        return self._gravcomp_cache_view

    # --------------------------------------------------------
    # Phase 2: ramp targets / set_target / reset_to_nominal
    # --------------------------------------------------------
    def _kp_scale_for(self, joint_name: str) -> float:
        """Per-joint kp scale via substring match (longest key wins)."""
        if self._kp_overrides:
            best = max(((len(k), v) for k, v in self._kp_overrides.items() if k in joint_name),
                       default=None)
            if best is not None:
                return float(best[1])
        return self._kp_scale

    def _kv_scale_for(self, joint_name: str) -> float:
        """Per-joint kv scale via substring match (longest key wins)."""
        if self._kv_overrides:
            best = max(((len(k), v) for k, v in self._kv_overrides.items() if k in joint_name),
                       default=None)
            if best is not None:
                return float(best[1])
        return self._kv_scale

    def set_target(
        self,
        joint_names: list[str],
        kps: "np.ndarray | None" = None,
        kds: "np.ndarray | None" = None,
        max_efforts: "np.ndarray | None" = None,
    ) -> None:
        """Update motor-domain ramp targets for the listed motors.

        Per slot per category: NaN means "no change". Names not registered
        in any group are silently dropped (counter-bumped). Host-only —
        the next ``update()`` ramps host shadows toward the new targets at
        ``motor_step_per_ms``.

        Args:
            joint_names: list of DOF names (length M, CSV column order).
            kps / kds / max_efforts: 1D float32 arrays of length M, or None.
        """
        if not self._set_target_logged:
            cats = []
            if kps is not None: cats.append("kp")
            if kds is not None: cats.append("kv")
            if max_efforts is not None: cats.append("trq")
            print(f"[ALLEX][Mirror] set_target wired (categories={cats}, n={len(joint_names)})")
            self._set_target_logged = True

        # kp_scale / kv_scale 적용 (per-joint overrides). via event 입력값을 미리 scale.
        if kps is not None:
            kps = np.asarray(kps, dtype=np.float32).copy()
            for i, name in enumerate(joint_names):
                if i < len(kps) and not np.isnan(kps[i]):
                    kps[i] *= self._kp_scale_for(name)
        if kds is not None:
            kds = np.asarray(kds, dtype=np.float32).copy()
            for i, name in enumerate(joint_names):
                if i < len(kds) and not np.isnan(kds[i]):
                    kds[i] *= self._kv_scale_for(name)

        t_now = self._step_t
        # Time per category:
        #   K_m, Kv_m: window_s (time-bounded, stretching style)
        #   trq_m: step-based — t_window = |delta| / (motor_step_per_ms · 1000)
        win_s = self._ramp_window_s
        step_per_s = self._motor_step_per_ms * 1000.0  # units / second
        for i, name in enumerate(joint_names):
            slot_info = self._motor_index.get(name)
            if slot_info is None:
                self._set_target_unmatched += 1
                continue
            grp, slot = slot_info
            if kps is not None and i < len(kps):
                v = float(kps[i])
                if not np.isnan(v) and v != grp.K_m_target[slot]:
                    grp.K_m_ramp_start[slot] = grp.K_m_host[slot]
                    grp.K_m_ramp_t0[slot] = t_now
                    grp.K_m_ramp_t_end[slot] = t_now + win_s
                    grp.K_m_target[slot] = v
            if kds is not None and i < len(kds):
                v = float(kds[i])
                if not np.isnan(v) and v != grp.Kv_m_target[slot]:
                    grp.Kv_m_ramp_start[slot] = grp.Kv_m_host[slot]
                    grp.Kv_m_ramp_t0[slot] = t_now
                    grp.Kv_m_ramp_t_end[slot] = t_now + win_s
                    grp.Kv_m_target[slot] = v
            if max_efforts is not None and i < len(max_efforts):
                v = float(max_efforts[i])
                if not np.isnan(v) and v != grp.trq_m_target[slot]:
                    delta = abs(v - grp.trq_m_host[slot])
                    t_win = delta / max(step_per_s, 1e-9) if step_per_s > 0 else 0.0
                    grp.trq_m_ramp_start[slot] = grp.trq_m_host[slot]
                    grp.trq_m_ramp_t0[slot] = t_now
                    grp.trq_m_ramp_t_end[slot] = t_now + t_win
                    grp.trq_m_target[slot] = v

    def reset_to_nominal(self) -> None:
        """Restore ramp targets to boot-time nominal. host ramps back automatically."""
        for grp in self._groups:
            np.copyto(grp.K_m_target, grp.K_m_nominal)
            np.copyto(grp.Kv_m_target, grp.Kv_m_nominal)
            np.copyto(grp.trq_m_target, grp.trq_m_nominal)

    def _ramp_group(self, grp: _KernelGroup) -> bool:
        """Per-step host ramp toward target — per-slot t_end smoothstep interpolation.

        Each slot ramps from start to target between [ramp_t0, ramp_t_end].
        Interpolation uses cosine smoothstep `s = 0.5·(1 - cos(π·u))` so the
        curve has zero slope at both endpoints (smooth start, smooth stop) —
        matches user request for sin/tanh-like easing.

        ramp_t_end depends on category (K_m/Kv_m = time-bounded ramp_window_s;
        trq_m = step-based |Δ|/step_per_s, allex_control style).
        """
        t_now = self._step_t
        dirty = False
        for host, target, start, t0, t_end in (
            (grp.K_m_host,   grp.K_m_target,   grp.K_m_ramp_start,   grp.K_m_ramp_t0,   grp.K_m_ramp_t_end),
            (grp.Kv_m_host,  grp.Kv_m_target,  grp.Kv_m_ramp_start,  grp.Kv_m_ramp_t0,  grp.Kv_m_ramp_t_end),
            (grp.trq_m_host, grp.trq_m_target, grp.trq_m_ramp_start, grp.trq_m_ramp_t0, grp.trq_m_ramp_t_end),
        ):
            dur = t_end - t0
            # Per-slot linear progress u ∈ [0,1]. dur ≤ 0 (snap) → u = 1.
            u = np.where(dur > 1e-12,
                         np.clip((t_now - t0) / np.maximum(dur, 1e-12), 0.0, 1.0),
                         1.0).astype(np.float32, copy=False)
            # Cosine smoothstep: smooth start & stop (zero derivative at u=0,1).
            alpha = (0.5 * (1.0 - np.cos(np.pi * u))).astype(np.float32, copy=False)
            new_host = (start + alpha * (target - start)).astype(np.float32, copy=False)
            if not np.array_equal(host, new_host):
                np.copyto(host, new_host)
                dirty = True
        return dirty

    def _upload_group(self, grp: _KernelGroup) -> None:
        """In-place H2D copy of host shadows for a group."""
        grp.K_m.assign(grp.K_m_host)
        grp.Kv_m.assign(grp.Kv_m_host)
        grp.trq_m.assign(grp.trq_m_host)

    # --------------------------------------------------------
    # Per-step update
    # --------------------------------------------------------
    def update(self) -> None:
        """매 physics step 호출.

        Phase 1 (motor-space PD + clip pipeline):
          host ramp K_m/Kv_m/τ_m → state/control 읽기 → pd_clip_* kernel launch →
          control.joint_f 에 write → solver sync.

        이전 (compute_*_K_j) 구현은 deprecation — model.joint_target_ke/kd 에 K_j 를
        써서 MuJoCo PD 가 출력하게 했으나, 이제 ke/kd=0 으로 강제하고 우리가 PD 식을
        직접 풀어 motor-space clip 후 joint_f 로 적용.
        """
        # Host ramp: upload only dirty groups before launches.
        for grp in self._groups:
            if self._ramp_group(grp):
                self._upload_group(grp)

        # Acquire per-step state / control / gravcomp views.
        q   = self._get_joint_q()
        qd  = self._get_joint_qd()
        qt  = self._get_target_pos()
        qdt = self._get_target_vel()
        joint_f = self._get_joint_f()
        gravcomp_j = self._get_gravcomp_world0()

        # Defensive: bail early if any required view is missing — write nothing.
        # (Newton step would then either skip our torque path or use defaults.)
        if any(v is None for v in (q, qd, qt, qdt, joint_f, gravcomp_j)):
            if not getattr(self, "_warned_missing_views", False):
                missing = [n for n, v in zip(
                    ("q", "qd", "qt", "qdt", "joint_f", "gravcomp_j"),
                    (q, qd, qt, qdt, joint_f, gravcomp_j)
                ) if v is None]
                print(f"[ALLEX][Mirror] missing views, skipping pd_clip kernel: {missing}")
                self._warned_missing_views = True
            return
        self._warned_missing_views = False

        # pd_mode == 2 (joint_nominal): 단일 kernel, USD-loaded 고정 값 사용
        if self._pd_mode == 2 and self._jn_count > 0:
            wp.launch(
                kernel=jk_kernel.pd_clip_joint_nominal,
                dim=self._jn_count,
                inputs=[
                    self._jn_dof_idx, self._jn_is_mech,
                    self._jn_K_j_nom, self._jn_Kv_j_nom, self._jn_tau_j_nom,
                    q, qd, qt, qdt, gravcomp_j,
                    joint_f, self._out_ke, self._out_kd, self._out_eff,
                ],
                device=self._device,
            )
            # solver sync + skip 나머지 motor-space kernel launches
            if self._sync_dof_props is not None:
                self._sync_dof_props()
            elif self._notify_solver is not None:
                self._notify_solver(self._notify_flag)
            return

        # pd_mode 0/1 (motor_space / joint_space_diag): 5개 group-specific kernel
        # scalar (10 motors, X≡Y, mech_comp 분기)
        wp.launch(
            kernel=jk_kernel.pd_clip_scalar,
            dim=self._scalar.n_motors,
            inputs=[
                self._scalar.dof_idx, self._scalar.ratio, self._scalar.is_mech_comp,
                self._scalar.K_m, self._scalar.Kv_m, self._scalar.trq_m,
                q, qd, qt, qdt, gravcomp_j,
                joint_f, self._out_ke, self._out_kd, self._out_eff,
            ],
            device=self._device,
        )
        # elbow (2 instances, constant J, X/Y 분기)
        wp.launch(
            kernel=jk_kernel.pd_clip_elbow,
            dim=self._elbow.n_instances,
            inputs=[
                self._elbow.dof_idx,
                self._elbow.K_m, self._elbow.Kv_m, self._elbow.trq_m,
                q, qd, qt, qdt, gravcomp_j,
                self._pd_mode,
                joint_f, self._out_ke, self._out_kd, self._out_eff,
                self._elbow_det,
            ],
            device=self._device,
        )
        # wrist (2 instances, q-dependent J, X/Y 분기)
        wp.launch(
            kernel=jk_kernel.pd_clip_wrist,
            dim=self._wrist.n_instances,
            inputs=[
                self._wrist.dof_idx,
                self._wrist.K_m, self._wrist.Kv_m, self._wrist.trq_m,
                q, qd, qt, qdt, gravcomp_j,
                self._pd_mode,
                self._wrist_poly_off,
                self._wrist_poly_exp_qr, self._wrist_poly_exp_qp, self._wrist_poly_coef,
                joint_f, self._out_ke, self._out_kd, self._out_eff,
                self._wrist_det,
            ],
            device=self._device,
        )
        # finger (8 instances, q-dependent J 3×3, X/Y 분기)
        wp.launch(
            kernel=jk_kernel.pd_clip_finger,
            dim=self._finger.n_instances,
            inputs=[
                self._finger.dof_idx,
                self._finger.K_m, self._finger.Kv_m, self._finger.trq_m,
                q, qd, qt, qdt, gravcomp_j,
                self._pd_mode,
                self._finger_poly_off,
                self._finger_poly_exp_q1, self._finger_poly_exp_q2,
                self._finger_poly_exp_q3, self._finger_poly_coef,
                self._finger_friction_motor, self._finger_friction_tanh_scl,
                joint_f, self._out_ke, self._out_kd, self._out_eff,
                self._finger_det,
            ],
            device=self._device,
        )
        # thumb (2 instances, q-dependent J 3×3, X/Y 분기)
        wp.launch(
            kernel=jk_kernel.pd_clip_thumb,
            dim=self._thumb.n_instances,
            inputs=[
                self._thumb.dof_idx,
                self._thumb.K_m, self._thumb.Kv_m, self._thumb.trq_m,
                q, qd, qt, qdt, gravcomp_j,
                self._pd_mode,
                self._thumb_poly_off,
                self._thumb_poly_exp_q1, self._thumb_poly_exp_q2,
                self._thumb_poly_exp_q3, self._thumb_poly_coef,
                self._thumb_friction_motor, self._thumb_friction_tanh_scl,
                joint_f, self._out_ke, self._out_kd, self._out_eff,
                self._thumb_det,
            ],
            device=self._device,
        )

        # Sync to MuJoCo actuator gainprm/biasprm (ke/kd=0 이 actuator 에 반영되도록).
        if self._sync_dof_props is not None:
            self._sync_dof_props()
        elif self._notify_solver is not None:
            self._notify_solver(self._notify_flag)

        # Singular monitoring — throttled host readback.
        self._step_counter += 1
        self._step_t += self._step_dt
        if self._step_counter % self._singular_check_interval == 0:
            self._check_singular()

    def _check_singular(self) -> None:
        """Periodically read |det(J)| device buffers, count near-singular events."""
        for name, det_arr in (
            ("elbow", self._elbow_det),
            ("wrist", self._wrist_det),
            ("finger", self._finger_det),
            ("thumb", self._thumb_det),
        ):
            try:
                vals = det_arr.numpy()
            except Exception:
                continue
            below = int(np.sum(vals < self._det_threshold))
            if below > 0:
                self._singular_count[name] += below
                # Log first time and then every 1000 events.
                if self._singular_count[name] - below == 0 or self._singular_count[name] % 1000 == 0:
                    print(
                        f"[ALLEX][Mirror] SINGULAR {name} |det|<{self._det_threshold:.1e} "
                        f"this step={below}/{len(vals)}, cumulative={self._singular_count[name]}"
                    )

    # --------------------------------------------------------
    # Reset / introspection
    # --------------------------------------------------------
    def reset(self) -> None:
        """Alias for ``reset_to_nominal()`` — host ramps back via update()."""
        self.reset_to_nominal()

    def num_motors(self) -> int:
        """Total active motors managed by mirror."""
        return (self._scalar.n_motors + self._elbow.n_motors
                + self._wrist.n_motors + self._finger.n_motors
                + self._thumb.n_motors)
