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
from the CSV side columns) are *ramped* linearly from their pre-event live
value to the CSV target over a fixed duration (default 1.0s) — never
applied as an instant step change, which would risk PD discontinuities and
joint snap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hermite_spline import generate_trajectory, parse_via_csv
from .joint_name_map import ALLEX_CSV_JOINT_NAMES


# ---------------------------------------------------------------------------
# Per-DOF runtime parameter categories.
#
# Every via event can carry up to three independent payloads (PD position
# stiffness, PD velocity damping, actuator torque limit). The same triple
# appears in CSV parsing, event extraction, tensor baking, ramp activation,
# and per-step write — keeping the metadata in a single table avoids
# triple-rewriting the same names across the file.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Category:
    raw_attr: str       # _ViaEvent attribute holding the parsed numpy row
    idx_attr: str       # pre-baked torch index tensor (DOF indices)
    vals_attr: str      # pre-baked torch values tensor (CSV target)
    start_attr: str     # snapshot tensor captured at ramp activation
    view_attr: str      # zero-copy torch view over Newton Model array
    model_attr: str     # Newton Model attribute name to view
    toggle_attr: str    # per-instance bool toggle (True = apply category)


_CATEGORIES: tuple[_Category, ...] = (
    _Category("kps",         "kps_idx", "kps_vals", "kps_start",
              "_model_kp_view",  "joint_target_ke",    "apply_kps_events"),
    _Category("kds",         "kds_idx", "kds_vals", "kds_start",
              "_model_kd_view",  "joint_target_kd",    "apply_kds_events"),
    _Category("max_efforts", "eff_idx", "eff_vals", "eff_start",
              "_model_eff_view", "joint_effort_limit", "apply_eff_events"),
)


@dataclass
class _ViaEvent:
    """CSV via-point에서 파생된 sim 런타임 이벤트.

    값은 CSV 원본 단위 그대로 (스케일링 없음). Pre-baked torch tensors are
    materialized at start() so the hot path does no Python-side allocation.
    Ramp activation snapshots the current model value (start_*); per-step
    writes interpolate from start → vals over n_ramp steps.
    """
    t: float                      # via 절대 시간 [s] (dense_base 기준)
    joint_names: list[str]        # 대상 DOF 이름
    kps: np.ndarray | None = None
    kds: np.ndarray | None = None
    max_efforts: np.ndarray | None = None
    sample_idx: int = -1          # start() 시점에 ramp offset 반영하여 계산
    # Pre-baked torch tensors. NaN entries removed; idx aligned with vals.
    kps_idx: object = None
    kps_vals: object = None
    kds_idx: object = None
    kds_vals: object = None
    eff_idx: object = None
    eff_vals: object = None
    # Ramp state. ``ramp_start_step = -1`` means "not yet activated".
    n_ramp: int = 1
    ramp_start_step: int = -1
    # Snapshot of model view values at activation, per category.
    kps_start: object = None
    kds_start: object = None
    eff_start: object = None


def _row_to_array(row: np.ndarray | None) -> np.ndarray | None:
    """NaN-only row는 None, 부분 NaN이면 해당 원소만 NaN으로 유지한 array 반환."""
    if row is None or np.all(np.isnan(row)):
        return None
    return np.asarray(row, dtype=np.float32)


@dataclass(frozen=True)
class _NewtonHandles:
    """Pieces of the Newton stage we need for the fast-path."""
    model: object
    solver: object
    sample_view: object  # torch view of model.joint_target_ke
    device: object
    dtype: object


class TrajectoryPlayer:
    def __init__(self, csv_dir: Path, articulation, hz: float = 200.0,
                 seed_pose: "np.ndarray | list[float] | None" = None,
                 ramp_s: float = 1.5,
                 event_ramp_s: float = 1.0):
        self._init_state(articulation, hz, ramp_s, event_ramp_s,
                         csv_dir=Path(csv_dir), seed_pose=seed_pose)
        self._build()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _init_state(self, articulation, hz: float, ramp_s: float,
                    event_ramp_s: float, *,
                    csv_dir: Path | None, seed_pose) -> None:
        """Initialize per-instance fields. Shared by __init__ and hold_pose."""
        self._csv_dir = csv_dir
        self._articulation = articulation
        self._hz = float(hz)
        self._seed_pose_override = seed_pose
        self._ramp_s = float(ramp_s)
        self.event_ramp_s = float(event_ramp_s)

        self._active = False
        self._finished = False
        self._sample_idx = 0

        self._num_dof = int(getattr(articulation, "num_dof", 0) or 0)
        dof_names = list(getattr(articulation, "dof_names", []) or [])
        self._dof_names: list[str] = dof_names
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(dof_names)}

        # _dense_base = trajectory as built from CSV; _dense = ramp-in prepended.
        self._dense_base: np.ndarray | None = None
        self._dense: np.ndarray | None = None
        self._duration_s: float = 0.0
        self._groups_used: list[str] = []
        self._missing_joints: list[str] = []

        # Sparse via events. Built in _build(); per-run sample_idx + tensor
        # baking happens in start(). _pending is sorted by sample_idx and
        # consumed forward; _active_ramps are events currently mid-ramp.
        self._events_spec: list[_ViaEvent] = []
        self._pending: list[_ViaEvent] = []
        self._pending_ptr: int = 0
        self._active_ramps: list[_ViaEvent] = []
        self._events_started: int = 0
        self._step_writes: int = 0

        # Newton fast-path: zero-copy torch views over Model.joint_target_ke /
        # joint_target_kd / joint_effort_limit (device wp.array). Writes go
        # straight to GPU memory; one solver._update_joint_dof_properties call
        # per step re-syncs MuJoCo actuator gainprm/biasprm via warp kernels.
        # This bypasses ArticulationView.set_dof_* (~140ms/call) entirely.
        self._model_kp_view = None
        self._model_kd_view = None
        self._model_eff_view = None
        self._sync_dof_props = None     # solver._update_joint_dof_properties (preferred)
        self._notify_solver = None      # solver.notify_model_changed (fallback)
        self._notify_flag: int = 0      # SolverNotifyFlags.JOINT_DOF_PROPERTIES

        # Per-category event toggles. Defaults all True; flip for diagnosis.
        self.apply_kps_events = True
        self.apply_kds_events = True
        self.apply_eff_events = True

        # CPU-side timing of each per-step write batch. Logs only when over
        # threshold to avoid spam during normal operation.
        self.event_log_threshold_ms = 2.0
        self._slow_step_count = 0
        self._max_step_ms = 0.0

    @classmethod
    def hold_pose(cls, articulation, hz: float, target_pose,
                  ramp_s: float = 3.0,
                  event_ramp_s: float = 1.0) -> "TrajectoryPlayer":
        """Build a player that ramps to ``target_pose`` then holds, no CSV.

        Used by Reset to drive every joint to a fixed target (e.g. zeros) via
        the same smooth ramp-in pipeline as a normal trajectory.
        """
        inst = cls.__new__(cls)
        inst._init_state(articulation, hz, ramp_s, event_ramp_s,
                         csv_dir=None, seed_pose=None)

        target = np.asarray(target_pose, dtype=np.float32).reshape(-1)
        if target.size != inst._num_dof:
            raise ValueError(
                f"target_pose size {target.size} != num_dof {inst._num_dof}"
            )
        inst._dense_base = target[None, :].copy()
        inst._dense = inst._dense_base
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

            _, pos_interp = generate_trajectory(
                data.t_via, data.pos_via, hz=self._hz, duration=max_dur
            )
            for j, jname in enumerate(joint_names):
                idx = self._name_to_idx.get(jname)
                if idx is None:
                    self._missing_joints.append(jname)
                    continue
                dense[:, idx] = pos_interp[:, j]

            events_spec.extend(self._extract_via_events(data, joint_names))

        self._events_spec = events_spec
        self._dense_base = dense
        self._dense = dense
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
                f"will linear-ramp over {self.event_ramp_s:.2f}s on activation"
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
        self._dense = self._build_with_ramp(self._dense_base)
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
        if self._pending:
            self._bake_events(self._pending)
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

        # 1) Activate due events
        pending = self._pending
        ptr = self._pending_ptr
        n_pending = len(pending)
        while ptr < n_pending and pending[ptr].sample_idx <= idx:
            self._activate_event(pending[ptr], idx)
            self._active_ramps.append(pending[ptr])
            self._events_started += 1
            ptr += 1
        self._pending_ptr = ptr

        # 2) Advance ramps + 3) sync once if needed
        if self._active_ramps:
            self._step_active_ramps(idx)

        target = self._dense[idx]
        if self._sample_idx >= last:
            self._finished = True
        else:
            self._sample_idx += 1
        return target

    # ------------------------------------------------------------------
    # Event scheduling + baking
    # ------------------------------------------------------------------
    def _build_events_for_run(self) -> list[_ViaEvent]:
        """Re-create _ViaEvent instances with sample_idx aligned to current ramp.

        ``t=0`` events fire at sample_idx=0 so a "from the start, use this gain"
        intent stays active during ramp-in too. ``t>0`` events shift by the
        ramp offset so via timestamps match the trajectory body.
        """
        ramp_offset = self._dense.shape[0] - self._dense_base.shape[0]
        n_ramp = max(1, int(round(self.event_ramp_s * self._hz)))
        out: list[_ViaEvent] = []
        for spec in self._events_spec:
            idx = 0 if spec.t == 0.0 else ramp_offset + int(round(spec.t * self._hz))
            out.append(_ViaEvent(
                t=spec.t, joint_names=spec.joint_names,
                kps=spec.kps, kds=spec.kds, max_efforts=spec.max_efforts,
                sample_idx=idx, n_ramp=n_ramp,
            ))
        out.sort(key=lambda e: e.sample_idx)
        return out

    def _acquire_newton_fast_path(self) -> "_NewtonHandles | None":
        """Resolve Newton stage handles and the sample model array view."""
        try:
            from isaacsim.physics.newton import acquire_stage
            stage = acquire_stage()
        except Exception as exc:
            print(f"[ALLEX][Traj] Newton stage acquire failed: {exc}; events disabled")
            return None
        if stage is None:
            print("[ALLEX][Traj] Newton stage not ready; events disabled")
            return None

        model = getattr(stage, "model", None)
        solver = getattr(stage, "solver", None)
        if model is None or solver is None:
            print("[ALLEX][Traj] Newton stage missing model/solver; events disabled")
            return None

        try:
            import warp as wp
        except Exception as exc:
            print(f"[ALLEX][Traj] warp import failed: {exc}; events disabled")
            return None

        sample = getattr(model, "joint_target_ke", None)
        if sample is None:
            print("[ALLEX][Traj] model.joint_target_ke unavailable; events disabled")
            return None
        try:
            sample_view = wp.to_torch(sample)
        except Exception as exc:
            print(f"[ALLEX][Traj] wp.to_torch on joint_target_ke failed: {exc}; events disabled")
            return None

        return _NewtonHandles(
            model=model, solver=solver, sample_view=sample_view,
            device=sample_view.device, dtype=sample_view.dtype,
        )

    def _bake_events(self, events: list[_ViaEvent]) -> None:
        """Materialize per-event device tensors and wire the Newton fast-path."""
        try:
            import torch
        except ImportError:
            print("[ALLEX][Traj] torch unavailable; events disabled")
            return

        handles = self._acquire_newton_fast_path()
        if handles is None:
            return
        dev, dt = handles.device, handles.dtype

        # 1) Bake per-event tensors (idx + vals) for each category.
        category_used = {cat.raw_attr: False for cat in _CATEGORIES}
        n_baked_slots = 0
        for ev in events:
            pairs = [
                (self._name_to_idx[n], i)
                for i, n in enumerate(ev.joint_names)
                if n in self._name_to_idx
            ]
            if not pairs:
                continue
            dof_idxs = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
            csv_cols = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
            for cat in _CATEGORIES:
                src = getattr(ev, cat.raw_attr)
                if src is None:
                    continue
                v = np.asarray(src, dtype=np.float32)[csv_cols]
                mask = ~np.isnan(v)
                if not mask.any():
                    continue
                setattr(ev, cat.idx_attr,
                        torch.as_tensor(dof_idxs[mask], dtype=torch.long, device=dev))
                setattr(ev, cat.vals_attr,
                        torch.as_tensor(v[mask], dtype=dt, device=dev))
                category_used[cat.raw_attr] = True
                n_baked_slots += 1

        # 2) Wire fast-path views for any category that produced an event.
        try:
            import warp as wp
            from newton.solvers import SolverNotifyFlags
        except Exception as exc:
            print(f"[ALLEX][Traj] warp/newton import failed late: {exc}; events disabled")
            return

        for cat in _CATEGORIES:
            if not category_used[cat.raw_attr]:
                continue
            if cat.model_attr == "joint_target_ke":
                view = handles.sample_view
            else:
                arr = getattr(handles.model, cat.model_attr, None)
                view = wp.to_torch(arr) if arr is not None else None
            setattr(self, cat.view_attr, view)

        # 3) Cache solver sync methods. Prefer ``_update_joint_dof_properties``
        # (gain + DOF property warp kernels only). ``notify_model_changed``
        # additionally runs ``set_length_range`` + ``set_const_0`` for the full
        # MuJoCo model — unnecessary for pure gain / effort-limit changes and
        # the source of the ~140ms stall per call.
        self._sync_dof_props = getattr(handles.solver, "_update_joint_dof_properties", None)
        self._notify_solver = handles.solver.notify_model_changed
        self._notify_flag = int(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        sync_mode = ("_update_joint_dof_properties"
                     if self._sync_dof_props is not None
                     else f"notify_model_changed(flag={self._notify_flag})")
        n_ramp = events[0].n_ramp if events else 0
        print(
            f"[ALLEX][Traj] pre-baked {n_baked_slots} slot(s) across "
            f"{len(events)} event(s); ramp={self.event_ramp_s:.2f}s ({n_ramp} steps); "
            f"Newton views: kp={self._model_kp_view is not None} "
            f"kd={self._model_kd_view is not None} eff={self._model_eff_view is not None} "
            f"(device={dev} dtype={dt}, sync={sync_mode})"
        )

    # ------------------------------------------------------------------
    # Ramp-in (initial pose → first dense row)
    # ------------------------------------------------------------------
    def _build_with_ramp(self, dense_base: np.ndarray) -> np.ndarray:
        """Prepend a smooth ramp from the live pose to dense_base[0].

        Quintic smoothstep (C2-continuous start + end), so the articulation
        eases out of its current pose and joins the trajectory with zero
        velocity instead of being yanked at full controller gain.
        """
        if self._ramp_s <= 0.0:
            return dense_base
        live = self._read_live_pose()
        if live is None:
            return dense_base

        first = dense_base[0]
        delta = first - live
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_delta < 1e-4:
            return dense_base

        n_ramp = max(1, int(round(self._ramp_s * self._hz)))
        u = np.linspace(0.0, 1.0, n_ramp + 1, dtype=np.float32)[1:]
        s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)  # 6u^5 - 15u^4 + 10u^3
        ramp = live[None, :] + s[:, None] * delta[None, :]
        out = np.concatenate([live[None, :], ramp, dense_base[1:]], axis=0)
        print(
            f"[ALLEX][Traj] ramp-in: {self._ramp_s:.2f}s ({n_ramp} samples), "
            f"max joint delta = {max_delta:.4f} rad"
        )
        return out

    # ------------------------------------------------------------------
    # Event ramp dispatch (hot path)
    # ------------------------------------------------------------------
    def _activate_event(self, ev: _ViaEvent, cur_step: int) -> None:
        """Snapshot live model values into ev.*_start so the per-step linear
        interpolation has a well-defined origin. Categories with no baked
        index tensor (i.e. event didn't author that category) are skipped.
        """
        ev.ramp_start_step = cur_step
        for cat in _CATEGORIES:
            idx_t = getattr(ev, cat.idx_attr)
            if idx_t is None:
                continue
            view = getattr(self, cat.view_attr)
            if view is None:
                continue
            setattr(ev, cat.start_attr, view[idx_t].clone())

    def _step_active_ramps(self, cur_step: int) -> None:
        """Advance every active ramp by one step; write + sync once."""
        t0 = time.perf_counter()
        wrote = False
        still_active: list[_ViaEvent] = []
        for ev in self._active_ramps:
            elapsed = cur_step - ev.ramp_start_step
            if elapsed >= ev.n_ramp:
                progress = 1.0
                done = True
            else:
                progress = elapsed / ev.n_ramp
                done = False
            if self._write_ramp_step(ev, progress):
                wrote = True
            if not done:
                still_active.append(ev)
        self._active_ramps = still_active

        if wrote:
            sync = self._sync_dof_props
            if sync is not None:
                sync()
            elif self._notify_solver is not None:
                self._notify_solver(self._notify_flag)
            self._step_writes += 1

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms > self._max_step_ms:
                self._max_step_ms = dt_ms
            if dt_ms > self.event_log_threshold_ms:
                self._slow_step_count += 1
                print(
                    f"[ALLEX][Traj] slow step write {dt_ms:.1f} ms "
                    f"(t={cur_step / self._hz:.2f}s "
                    f"active_ramps={len(self._active_ramps) + (0 if not still_active else 0)})"
                )

    def _write_ramp_step(self, ev: _ViaEvent, progress: float) -> bool:
        """Write one interpolated step of ``ev`` into the model views.

        Returns True if any model array was actually written (i.e., the event
        had a baked tensor and the corresponding category toggle is on).
        """
        wrote = False
        for cat in _CATEGORIES:
            idx_t = getattr(ev, cat.idx_attr)
            if idx_t is None:
                continue
            view = getattr(self, cat.view_attr)
            if view is None or not getattr(self, cat.toggle_attr):
                continue
            target = getattr(ev, cat.vals_attr)
            if progress >= 1.0:
                view[idx_t] = target
            else:
                start = getattr(ev, cat.start_attr)
                view[idx_t] = start + progress * (target - start)
            wrote = True
        return wrote
