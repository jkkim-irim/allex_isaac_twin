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
    ratio: wp.array | None = None  # scalar group only — per-motor reduction ratio


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

    def __init__(self, articulation, model, solver, stage=None, device: str = "cuda:0"):
        self._articulation = articulation
        self._model = model
        self._solver = solver
        self._stage = stage  # Newton stage — for live state_0.joint_q
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
        )
        self._nominal = get_nominal_motor_gains()
        if not self._nominal:
            raise RuntimeError("nominal_motor_gains not found in joint_config.json")

        # Cache motor-domain ramp step (per 1ms). 0.0 → snap mode (no ramp).
        self._motor_step = float(get_motor_step_per_ms() or 0.0)
        if self._motor_step <= 0.0:
            print("[ALLEX][Mirror] motor_step_per_ms <= 0 → snap mode (host=target instantly)")

        # Build per-kernel groups.
        self._scalar = self._build_group("scalar", SCALAR_MOTORS, n_per_inst=1,
                                         ratios=SCALAR_RATIOS)
        self._elbow  = self._build_group("elbow", ELBOW_MOTORS, n_per_inst=2)
        self._wrist  = self._build_group("wrist", WRIST_MOTORS, n_per_inst=2)
        self._finger = self._build_group("finger", FINGER_MOTORS, n_per_inst=3)
        self._thumb  = self._build_group("thumb", THUMB_MOTORS, n_per_inst=3)
        self._groups: tuple[_KernelGroup, ...] = (
            self._scalar, self._elbow, self._wrist, self._finger, self._thumb,
        )

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
            ratio=ratio_dev,
        )

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

    # --------------------------------------------------------
    # Phase 2: ramp targets / set_target / reset_to_nominal
    # --------------------------------------------------------
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

        for i, name in enumerate(joint_names):
            slot_info = self._motor_index.get(name)
            if slot_info is None:
                self._set_target_unmatched += 1
                continue
            grp, slot = slot_info
            if kps is not None and i < len(kps):
                v = float(kps[i])
                if not np.isnan(v):
                    grp.K_m_target[slot] = v
            if kds is not None and i < len(kds):
                v = float(kds[i])
                if not np.isnan(v):
                    grp.Kv_m_target[slot] = v
            if max_efforts is not None and i < len(max_efforts):
                v = float(max_efforts[i])
                if not np.isnan(v):
                    grp.trq_m_target[slot] = v

    def reset_to_nominal(self) -> None:
        """Restore ramp targets to boot-time nominal. host ramps back automatically."""
        for grp in self._groups:
            np.copyto(grp.K_m_target, grp.K_m_nominal)
            np.copyto(grp.Kv_m_target, grp.Kv_m_nominal)
            np.copyto(grp.trq_m_target, grp.trq_m_nominal)

    def _ramp_group(self, grp: _KernelGroup) -> bool:
        """Per-step host ramp toward target. Returns True if any element changed.

        Snap mode (motor_step <= 0): host := target in one shot.
        Otherwise: host += clip(target - host, ±step), then snap-to-target
        when within one step (absorbs float drift, clean steady-state).
        """
        step = self._motor_step
        if step <= 0.0:
            dirty = False
            for host, target in (
                (grp.K_m_host, grp.K_m_target),
                (grp.Kv_m_host, grp.Kv_m_target),
                (grp.trq_m_host, grp.trq_m_target),
            ):
                if not np.array_equal(host, target):
                    np.copyto(host, target)
                    dirty = True
            return dirty

        dirty = False
        for host, target in (
            (grp.K_m_host, grp.K_m_target),
            (grp.Kv_m_host, grp.Kv_m_target),
            (grp.trq_m_host, grp.trq_m_target),
        ):
            diff = target - host
            if not np.any(diff):
                continue
            delta = np.clip(diff, -step, step).astype(np.float32, copy=False)
            host += delta
            near = np.abs(target - host) < step
            if np.any(near):
                host[near] = target[near]
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
        """매 physics step 호출. host ramp → dirty 시 H2D → kernel → solver sync."""
        # Phase 2: ramp host shadow toward targets, upload only dirty groups.
        # Must run BEFORE kernel launches (kernels read device K_m / Kv_m / trq_m).
        for grp in self._groups:
            if self._ramp_group(grp):
                self._upload_group(grp)

        # q-independent kernels: scalar (shoulder/waist/neck) + elbow.
        wp.launch(
            kernel=jk_kernel.apply_scalar_groups,
            dim=self._scalar.n_motors,
            inputs=[
                self._scalar.dof_idx, self._scalar.ratio,
                self._scalar.K_m, self._scalar.Kv_m, self._scalar.trq_m,
                self._out_ke, self._out_kd, self._out_eff,
            ],
            device=self._device,
        )
        wp.launch(
            kernel=jk_kernel.compute_elbow_K_j,
            dim=self._elbow.n_instances,
            inputs=[
                self._elbow.dof_idx,
                self._elbow.K_m, self._elbow.Kv_m, self._elbow.trq_m,
                self._out_ke, self._out_kd, self._out_eff,
            ],
            device=self._device,
        )

        # q-dependent kernels: wrist / finger / thumb — use live state q.
        joint_q = self._get_joint_q()
        wp.launch(
            kernel=jk_kernel.compute_wrist_K_j,
            dim=self._wrist.n_instances,
            inputs=[
                self._wrist.dof_idx,
                self._wrist.K_m, self._wrist.Kv_m, self._wrist.trq_m,
                joint_q,
                self._out_ke, self._out_kd, self._out_eff,
            ],
            device=self._device,
        )
        wp.launch(
            kernel=jk_kernel.compute_finger_K_j,
            dim=self._finger.n_instances,
            inputs=[
                self._finger.dof_idx,
                self._finger.K_m, self._finger.Kv_m, self._finger.trq_m,
                joint_q,
                self._out_ke, self._out_kd, self._out_eff,
            ],
            device=self._device,
        )
        wp.launch(
            kernel=jk_kernel.compute_thumb_K_j,
            dim=self._thumb.n_instances,
            inputs=[
                self._thumb.dof_idx,
                self._thumb.K_m, self._thumb.Kv_m, self._thumb.trq_m,
                joint_q,
                self._out_ke, self._out_kd, self._out_eff,
            ],
            device=self._device,
        )

        # Sync to MuJoCo actuator gainprm/biasprm.
        if self._sync_dof_props is not None:
            self._sync_dof_props()
        elif self._notify_solver is not None:
            self._notify_solver(self._notify_flag)

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
