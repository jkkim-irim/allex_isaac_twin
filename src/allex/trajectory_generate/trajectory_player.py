"""Cubic Hermite trajectory playback against the live ALLEX articulation.

Usage:
    player = TrajectoryPlayer(csv_dir, articulation, hz=200)
    player.start()                       # when user clicks Run
    target = player.get_current_target() # called every physics step
    player.stop()                        # user clicks Stop

Once `start()` is called, each `get_current_target()` call advances one
sample (assumed to match the physics step rate). When the trajectory ends,
the player enters "finished" state and keeps returning the last frame —
equivalent to holding the final pose.

Via events (PD stiffness / damping / actuator torque limit changes parsed
from the CSV side columns) are dispatched in ``get_current_target()`` via
``MotorStateMirror.set_target``, which owns the per-step ramp of
``joint_target_ke/kd`` / effort limit. Player hz is sync'd to
physics_hz=1000 by the UI, so 1 call = 1 ms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hermite_spline import generate_trajectory, parse_via_csv
from .joint_name_map import ALLEX_CSV_JOINT_NAMES


@dataclass
class _ViaEvent:
    """CSV via-point에서 파생된 sim 런타임 이벤트.

    값은 CSV 원본 단위 그대로 (스케일링 없음).
    """
    t: float                      # via 절대 시간 [s] (dense_base 기준)
    joint_names: list[str]        # 대상 DOF 이름 (CSV column 순서)
    kps: np.ndarray | None = None
    kds: np.ndarray | None = None
    max_efforts: np.ndarray | None = None
    sample_idx: int = -1          # start() 시점에 ramp offset 반영하여 계산


def _row_to_array(row: np.ndarray | None) -> np.ndarray | None:
    """NaN-only row는 None, 부분 NaN이면 해당 원소만 NaN으로 유지한 array 반환."""
    if row is None or np.all(np.isnan(row)):
        return None
    return np.asarray(row, dtype=np.float32)


class TrajectoryPlayer:
    def __init__(self, csv_dir: Path, articulation, hz: float = 200.0,
                 seed_pose: "np.ndarray | list[float] | None" = None,
                 ramp_s: float = 1.5,
                 event_ramp_s: float = 1.0,
                 raw_mode: bool = False,
                 motor_mirror=None):
        self._init_state(articulation, hz, ramp_s, event_ramp_s,
                         csv_dir=Path(csv_dir), seed_pose=seed_pose,
                         raw_mode=raw_mode)
        self._motor_mirror = motor_mirror
        self._build()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _init_state(self, articulation, hz: float, ramp_s: float,
                    event_ramp_s: float = 1.0, *,
                    csv_dir: Path | None, seed_pose,
                    raw_mode: bool = False) -> None:
        """Initialize per-instance fields. Shared by __init__ and hold_pose."""
        self._csv_dir = csv_dir
        self._articulation = articulation
        self._hz = float(hz)
        self._seed_pose_override = seed_pose
        self._ramp_s = float(ramp_s)
        self.event_ramp_s = float(event_ramp_s)
        # raw_mode=True: CSV via point 을 그대로 (zero-order hold) 재생.
        # rosbag 에서 이미 spline 적용된 1kHz target 을 재샘플링 없이 재생할 때 사용.
        self._raw_mode = bool(raw_mode)
        # MotorStateMirror reference for set_target / reset_to_nominal events.
        # __init__ overrides via constructor arg; hold_pose / scenario inject later.
        self._motor_mirror = None

        self._active = False
        self._finished = False
        self._sample_idx = 0

        self._num_dof = int(getattr(articulation, "num_dof", 0) or 0)
        dof_names = list(getattr(articulation, "dof_names", []) or [])
        self._dof_names: list[str] = dof_names
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(dof_names)}

        # _dense_base = trajectory as built from CSV; _dense = ramp-in prepended.
        # _dense_vel_* are matching analytical velocity arrays [rad/s], same shape.
        self._dense_base: np.ndarray | None = None
        self._dense: np.ndarray | None = None
        self._dense_vel_base: np.ndarray | None = None
        self._dense_vel: np.ndarray | None = None
        self._duration_s: float = 0.0
        self._groups_used: list[str] = []
        self._missing_joints: list[str] = []

        # Cache of the velocity row that pairs with the most recent
        # ``get_current_target()``. Read by ``get_current_velocity_target``
        # without advancing the sample index.
        self._last_vel_target: np.ndarray | None = None

        # Sparse via events. Built in _build(); per-run sample_idx + tensor
        # baking happens in start(). _pending is sorted by sample_idx and
        # consumed forward; _active_ramps are events currently mid-ramp.
        self._events_spec: list[_ViaEvent] = []
        self._pending: list[_ViaEvent] = []
        self._pending_ptr: int = 0
        self._active_ramps: list[_ViaEvent] = []
        self._events_started: int = 0
        self._step_writes: int = 0

        # CPU-side timing of each per-step write batch. Logs only when over
        # threshold to avoid spam during normal operation.
        self.event_log_threshold_ms = 2.0
        self._slow_step_count = 0
        self._max_step_ms = 0.0

    @classmethod
    def hold_pose(cls, articulation, hz: float, target_pose,
                  ramp_s: float = 3.0) -> "TrajectoryPlayer":
        """Build a player that ramps to ``target_pose`` then holds, no CSV.

        Used by Reset to drive every joint to a fixed target (e.g. zeros) via
        the same smooth ramp-in pipeline as a normal trajectory.
        """
        inst = cls.__new__(cls)
        inst._init_state(articulation, hz, ramp_s,
                         csv_dir=None, seed_pose=None)

        target = np.asarray(target_pose, dtype=np.float32).reshape(-1)
        if target.size != inst._num_dof:
            raise ValueError(
                f"target_pose size {target.size} != num_dof {inst._num_dof}"
            )
        inst._dense_base = target[None, :].copy()
        inst._dense = inst._dense_base
        inst._dense_vel_base = np.zeros_like(inst._dense_base)
        inst._dense_vel = inst._dense_vel_base
        inst._duration_s = 0.0
        return inst

    # ------------------------------------------------------------------
    # Seed pose
    # ------------------------------------------------------------------
    def _resolve_seed_pose(self) -> np.ndarray:
        """Pick the best available neutral pose of length num_dof."""
        n = self._num_dof

        if self._seed_pose_override is not None:
            try:
                arr = self._coerce_to_numpy(self._seed_pose_override)
                if arr.size >= n:
                    return arr[:n].copy()
                if arr.size > 0:
                    pad = np.zeros(n, dtype=np.float32)
                    pad[: arr.size] = arr
                    return pad
            except Exception as exc:
                print(f"[ALLEX][Traj] seed_pose override invalid: {exc}")

        live = self._read_live_pose()
        if live is not None:
            return live.copy()

        return np.zeros(n, dtype=np.float32)

    @staticmethod
    def _coerce_to_numpy(value) -> np.ndarray:
        """Normalize torch tensors / list-of-tensors / arrays into 1D float32 numpy."""
        if hasattr(value, "detach"):                    # torch tensor (possibly cuda)
            value = value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)):
            value = [
                float(x.item()) if hasattr(x, "item") else float(x)
                for x in value
            ]
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _read_live_pose(self) -> "np.ndarray | None":
        """Best-effort read of current articulation joint positions."""
        try:
            pose = self._articulation.get_joint_positions()
        except Exception as exc:
            print(f"[ALLEX][Traj] live pose read failed: {exc}")
            return None
        if pose is None:
            return None
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        return arr if arr.size == self._num_dof else None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self) -> None:
        if self._num_dof <= 0 or not self._dof_names:
            print("[ALLEX][Traj] articulation has no dof_names yet; cannot build trajectory")
            return

        pose = self._resolve_seed_pose()

        group_data: dict[str, object] = {}
        max_dur = 0.0
        for csv_name in ALLEX_CSV_JOINT_NAMES.keys():
            csv_path = self._csv_dir / f"{csv_name}.csv"
            if not csv_path.exists():
                continue
            try:
                data = parse_via_csv(csv_path)
            except Exception as exc:
                print(f"[ALLEX][Traj] parse failed: {csv_path.name}: {exc}")
                continue
            if len(data.t_via) == 0:
                continue
            group_data[csv_name] = data
            max_dur = max(max_dur, float(data.t_via[-1]))
            self._groups_used.append(csv_name)

        if not group_data or max_dur <= 0.0:
            print(f"[ALLEX][Traj] no usable CSV groups found in {self._csv_dir}")
            return

        n_steps = int(max_dur * self._hz)
        # Dense target buffer, tiled with seed pose (hold for uncovered joints).
        dense = np.tile(pose.astype(np.float32)[None, :], (n_steps + 1, 1))
        # Velocity buffer: 0 by default (uncovered joints stay still).
        dense_vel = np.zeros_like(dense)

        events_spec: list[_ViaEvent] = []
        for csv_name, data in group_data.items():
            joint_names = ALLEX_CSV_JOINT_NAMES[csv_name]
            n_cols_csv = int(data.pos_via.shape[1])
            if n_cols_csv != len(joint_names):
                print(
                    f"[ALLEX][Traj] {csv_name}.csv has {n_cols_csv} joint cols but map "
                    f"expects {len(joint_names)}; skipping"
                )
                continue

            if self._raw_mode:
                # Zero-order hold: t_out 각 시점에 대해 t_via <= t_out 의 가장
                # 큰 idx 사용. rosbag 에서 이미 spline 적용된 1kHz target 을
                # 재샘플링 없이 재생할 때 사용.
                dt = 1.0 / self._hz
                t_out = np.arange(n_steps + 1, dtype=np.float64) * dt
                idx_arr = np.searchsorted(data.t_via, t_out, side="right") - 1
                idx_arr = np.clip(idx_arr, 0, len(data.t_via) - 1)
                pos_interp = data.pos_via[idx_arr]
                vel_interp = np.zeros_like(pos_interp)
            else:
                _, pos_interp, vel_interp = generate_trajectory(
                    data.t_via, data.pos_via, hz=self._hz, duration=max_dur
                )
            for j, jname in enumerate(joint_names):
                idx = self._name_to_idx.get(jname)
                if idx is None:
                    self._missing_joints.append(jname)
                    continue
                dense[:, idx] = pos_interp[:, j]
                dense_vel[:, idx] = vel_interp[:, j]

            events_spec.extend(self._extract_via_events(data, joint_names))

        self._events_spec = events_spec
        self._dense_base = dense
        self._dense = dense
        self._dense_vel_base = dense_vel
        self._dense_vel = dense_vel
        self._duration_s = float(dense.shape[0] - 1) / self._hz

        print(
            f"[ALLEX][Traj] loaded groups={self._groups_used} "
            f"duration={max_dur:.2f}s samples={n_steps + 1} hz={self._hz}"
        )
        if self._missing_joints:
            print(
                f"[ALLEX][Traj] unmatched joint names (skipped): "
                f"{sorted(set(self._missing_joints))}"
            )
        if events_spec:
            n_gain = sum(1 for e in events_spec if e.kps is not None or e.kds is not None)
            n_trq = sum(1 for e in events_spec if e.max_efforts is not None)
            print(
                f"[ALLEX][Traj] parsed {len(events_spec)} via events "
                f"(gain={n_gain}, torque_limit={n_trq}); "
                f"per-step additive clamp ramp (step sizes from "
                f"physics_config.json::newton.ramp_step_sizes)"
            )

    # ------------------------------------------------------------------
    # Via-event extraction (runtime PD-gain / torque-limit changes)
    # ------------------------------------------------------------------
    def _extract_via_events(self, data, joint_names: list[str]) -> list[_ViaEvent]:
        """Convert per-via sparse tuning columns into runtime events.

        Row 0 (t=0) is included — if the user authored explicit gain/effort
        values in the first CSV row they want them active from the start.
        ``start()`` schedules t=0 events at sample_idx=0 so the ramp begins
        immediately; the live value at that moment is the snapshot start.
        Rows where every gain/effort cell is NaN are filtered to None.
        """
        events: list[_ViaEvent] = []
        if not any(name in self._name_to_idx for name in joint_names):
            return events

        kps_via = getattr(data, "kps_via", None)
        kds_via = getattr(data, "kds_via", None)
        trq_via = getattr(data, "trq_via", None)
        for k in range(len(data.t_via)):
            kps = _row_to_array(kps_via[k]) if kps_via is not None else None
            kds = _row_to_array(kds_via[k]) if kds_via is not None else None
            trq = _row_to_array(trq_via[k]) if trq_via is not None else None
            if kps is None and kds is None and trq is None:
                continue
            events.append(_ViaEvent(
                t=float(data.t_via[k]),
                joint_names=list(joint_names),
                kps=kps, kds=kds, max_efforts=trq,
            ))
        return events

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------
    def set_motor_mirror(self, mirror) -> None:
        """Late-bind a MotorStateMirror so via events can fire set_target().

        scenario lazy-builds the mirror on first physics step, but the
        TrajectoryPlayer is usually constructed earlier via UI. This setter
        bridges the gap.
        """
        self._motor_mirror = mirror

    def is_ready(self) -> bool:
        return self._dense is not None and self._dense.shape[0] > 0

    def is_active(self) -> bool:
        return self._active and self.is_ready()

    def is_finished(self) -> bool:
        return self._finished

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def groups_used(self) -> list[str]:
        return list(self._groups_used)

    def start(self) -> bool:
        if self._dense_base is None or self._dense_base.shape[0] == 0:
            return False
        self._dense, self._dense_vel = self._build_with_ramp(
            self._dense_base, self._dense_vel_base
        )
        self._duration_s = float(self._dense.shape[0] - 1) / self._hz
        self._active = True
        self._finished = False
        self._sample_idx = 0
        self._pending_ptr = 0
        self._active_ramps = []
        self._events_started = 0
        self._step_writes = 0
        self._slow_step_count = 0
        self._max_step_ms = 0.0
        self._pending = self._build_events_for_run()
        # NOTE: MotorStateMirror owns joint_target_ke/kd/eff writes — events are
        # dispatched via motor_mirror.set_target in get_current_target().
        return True

    def stop(self) -> None:
        self._active = False
        if self._events_started or self._max_step_ms > 0.0:
            print(
                f"[ALLEX][Traj] stop: started {self._events_started}/{len(self._pending)} events; "
                f"step writes={self._step_writes}; "
                f"slow(>{self.event_log_threshold_ms:.1f}ms)={self._slow_step_count}; "
                f"max single step={self._max_step_ms:.2f} ms"
            )
        if self._motor_mirror is not None:
            self._motor_mirror.reset_to_nominal()
            print("[ALLEX][Traj] stop: ramping K_m back to nominal")

    def get_current_target(self) -> "np.ndarray | None":
        """Return the current target row and advance one sample.

        Hot-path responsibilities:
          1) Activate any pending events whose sample_idx has been reached
             (snapshot the live model values for their DOFs as ramp start).
          2) Advance every active ramp by one step — write the linearly
             interpolated value to the model views; drop completed ramps.
          3) If anything was written this step, run one solver DOF-property
             sync so the new gains take effect on the next physics step.
          4) Return the dense position target row for the controller.
        """
        if not self.is_active() or self._dense is None:
            return None
        last = self._dense.shape[0] - 1
        idx = min(self._sample_idx, last)

        # Drain any via events whose sample_idx has been reached. Each event
        # pushes new motor-domain ramp targets into MotorStateMirror; the
        # mirror's per-step host ramp absorbs the multi-step transition.
        if self._motor_mirror is not None:
            while (self._pending_ptr < len(self._pending)
                   and self._pending[self._pending_ptr].sample_idx <= self._sample_idx):
                ev = self._pending[self._pending_ptr]
                self._pending_ptr += 1
                self._motor_mirror.set_target(
                    ev.joint_names, ev.kps, ev.kds, ev.max_efforts,
                )
                self._events_started += 1

        target = self._dense[idx]
        if self._dense_vel is not None and idx < self._dense_vel.shape[0]:
            self._last_vel_target = self._dense_vel[idx]
        else:
            self._last_vel_target = None
        if self._sample_idx >= last:
            self._finished = True
        else:
            self._sample_idx += 1
        return target

    def get_current_velocity_target(self) -> "np.ndarray | None":
        """Return the velocity row paired with the most recent
        ``get_current_target()`` call (does NOT advance the sample index).

        Returns ``None`` if the player is not active or the dense velocity
        buffer hasn't been built (e.g. ``hold_pose`` factory before any
        ``get_current_target`` call). Units: rad/s.
        """
        if not self.is_active():
            return None
        return self._last_vel_target

    # ------------------------------------------------------------------
    # Event scheduling
    # ------------------------------------------------------------------
    def _build_events_for_run(self) -> list[_ViaEvent]:
        """Re-create _ViaEvent instances with sample_idx aligned to current ramp.

        ``t=0`` events fire at sample_idx=0 so a "from the start, use this gain"
        intent stays active during ramp-in too. ``t>0`` events shift by the
        ramp offset so via timestamps match the trajectory body.
        """
        ramp_offset = self._dense.shape[0] - self._dense_base.shape[0]
        out: list[_ViaEvent] = []
        for spec in self._events_spec:
            idx = 0 if spec.t == 0.0 else ramp_offset + int(round(spec.t * self._hz))
            out.append(_ViaEvent(
                t=spec.t, joint_names=spec.joint_names,
                kps=spec.kps, kds=spec.kds, max_efforts=spec.max_efforts,
                sample_idx=idx,
            ))
        out.sort(key=lambda e: e.sample_idx)
        return out

    # ------------------------------------------------------------------
    # Ramp-in (initial pose → first dense row)
    # ------------------------------------------------------------------
    def _build_with_ramp(
        self, dense_base: np.ndarray, dense_vel_base: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepend a smooth ramp from the live pose to dense_base[0].

        Quintic smoothstep ``s(u) = 6u^5 - 15u^4 + 10u^3`` (C2-continuous at
        both ends). 위치는 ``live + s(u)*delta``, 속도는 닫힌 형태 미분
        ``ds/dt = (30u^4 - 60u^3 + 30u^2) / ramp_s * delta``.
        u=0 / u=1 모두 속도 0 이므로 본 궤적 진입 시 jerk 없이 이어진다.

        Returns:
            (pos_out, vel_out) — same row count, same DOF axis.
        """
        if dense_vel_base is None:
            dense_vel_base = np.zeros_like(dense_base)
        if self._ramp_s <= 0.0:
            return dense_base, dense_vel_base
        live = self._read_live_pose()
        if live is None:
            return dense_base, dense_vel_base

        first = dense_base[0]
        delta = first - live
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_delta < 1e-4:
            return dense_base, dense_vel_base

        n_ramp = max(1, int(round(self._ramp_s * self._hz)))
        u = np.linspace(0.0, 1.0, n_ramp + 1, dtype=np.float32)[1:]
        s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)         # 6u^5 - 15u^4 + 10u^3
        sd = (30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2)        # ds/du
        ramp_pos = live[None, :] + s[:, None] * delta[None, :]
        ramp_vel = (sd / self._ramp_s)[:, None] * delta[None, :]

        # Live row prepended at u=0 (vel=0) so playback starts with the
        # current pose held; then the ramp segment carries velocity.
        live_row = live[None, :]
        zero_row = np.zeros_like(live_row)
        pos_out = np.concatenate([live_row, ramp_pos, dense_base[1:]], axis=0)
        vel_out = np.concatenate([zero_row, ramp_vel, dense_vel_base[1:]], axis=0)
        print(
            f"[ALLEX][Traj] ramp-in: {self._ramp_s:.2f}s ({n_ramp} samples), "
            f"max joint delta = {max_delta:.4f} rad"
        )
        return pos_out, vel_out
