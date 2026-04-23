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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .hermite_spline import generate_trajectory, parse_via_csv
from .joint_name_map import ALLEX_CSV_JOINT_NAMES


class TrajectoryPlayer:
    def __init__(self, csv_dir: Path, articulation, hz: float = 200.0,
                 seed_pose: "np.ndarray | list[float] | None" = None,
                 raw_mode: bool = False):
        self._csv_dir = Path(csv_dir)
        self._articulation = articulation
        self._hz = float(hz)
        self._seed_pose_override = seed_pose
        # raw_mode=True: CSV via point 을 그대로 (zero-order hold) 재생.
        # rosbag 에서 이미 spline 적용된 1kHz target 을 재샘플링 없이 재생할 때 사용.
        self._raw_mode = bool(raw_mode)

        self._active: bool = False
        self._finished: bool = False
        self._sample_idx: int = 0

        self._num_dof: int = int(getattr(articulation, "num_dof", 0) or 0)
        dof_names = list(getattr(articulation, "dof_names", []) or [])
        self._dof_names: list[str] = dof_names
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(dof_names)}

        self._dense: np.ndarray | None = None  # shape (n_steps+1, num_dof)
        self._duration_s: float = 0.0
        self._groups_used: list[str] = []
        self._missing_joints: list[str] = []

        self._build()

    # ------------------------------------------------------------------
    # Seed pose
    # ------------------------------------------------------------------
    def _resolve_seed_pose(self) -> np.ndarray:
        """Pick the best available neutral pose of length num_dof."""
        n = self._num_dof

        # 1) explicit override wins
        if self._seed_pose_override is not None:
            try:
                src = self._seed_pose_override
                # torch tensor (GPU/CPU) → numpy
                if hasattr(src, "detach"):
                    src = src.detach().cpu().numpy()
                arr = np.asarray(src, dtype=np.float32).reshape(-1)
                if arr.size >= n:
                    return arr[:n].copy()
                if arr.size > 0:
                    pad = np.zeros(n, dtype=np.float32)
                    pad[: arr.size] = arr
                    return pad
            except Exception as exc:
                print(f"[ALLEX][Traj] seed_pose override invalid: {exc}")

        # 2) try live articulation positions (tensor-safe)
        try:
            pose = self._articulation.get_joint_positions()
            if pose is not None:
                if hasattr(pose, "detach"):  # torch tensor
                    pose = pose.detach().cpu().numpy()
                arr = np.asarray(pose, dtype=np.float32).reshape(-1)
                if arr.size == n:
                    return arr.copy()
        except Exception as exc:
            print(f"[ALLEX][Traj] get_joint_positions failed: {exc}; falling back to zeros")

        # 3) zeros
        return np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self) -> None:
        if self._num_dof <= 0 or not self._dof_names:
            print("[ALLEX][Traj] articulation has no dof_names yet; cannot build trajectory")
            return

        # Seed the dense buffer with a sensible pose so joints not covered by
        # any CSV hold their value instead of snapping to 0.
        pose = self._resolve_seed_pose()

        # Parse all group CSVs present in this folder, track max duration.
        group_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        max_dur = 0.0
        for csv_name in ALLEX_CSV_JOINT_NAMES.keys():
            csv_path = self._csv_dir / f"{csv_name}.csv"
            if not csv_path.exists():
                continue
            try:
                t_via, pos_via = parse_via_csv(csv_path)
            except Exception as exc:
                print(f"[ALLEX][Traj] parse failed: {csv_path.name}: {exc}")
                continue
            if len(t_via) == 0:
                continue
            group_data[csv_name] = (t_via, pos_via)
            max_dur = max(max_dur, float(t_via[-1]))
            self._groups_used.append(csv_name)

        if not group_data or max_dur <= 0.0:
            print(f"[ALLEX][Traj] no usable CSV groups found in {self._csv_dir}")
            return

        dt = 1.0 / self._hz
        n_steps = int(max_dur / dt)
        self._duration_s = max_dur

        # Dense target buffer, tiled with current pose (hold for uncovered joints).
        dense = np.tile(pose.astype(np.float32)[None, :], (n_steps + 1, 1))

        # Interpolate each group then write into the dof index slots.
        for csv_name, (t_via, pos_via) in group_data.items():
            joint_names = ALLEX_CSV_JOINT_NAMES[csv_name]
            n_cols_csv = int(pos_via.shape[1])
            if n_cols_csv != len(joint_names):
                print(
                    f"[ALLEX][Traj] {csv_name}.csv has {n_cols_csv} joint cols but map "
                    f"expects {len(joint_names)}; skipping"
                )
                continue

            if self._raw_mode:
                # Zero-order hold: t_out 각 시점에 대해 t_via <= t_out 의 가장 큰 idx 사용.
                t_out = np.arange(n_steps + 1, dtype=np.float64) * dt
                idx_arr = np.searchsorted(t_via, t_out, side="right") - 1
                idx_arr = np.clip(idx_arr, 0, len(t_via) - 1)
                pos_interp = pos_via[idx_arr]  # (n_steps+1, n_cols_csv)
            else:
                _, pos_interp = generate_trajectory(t_via, pos_via, hz=self._hz, duration=max_dur)
            # pos_interp shape: (n_steps+1, n_cols_csv)
            for j, jname in enumerate(joint_names):
                idx = self._name_to_idx.get(jname)
                if idx is None:
                    self._missing_joints.append(jname)
                    continue
                dense[:, idx] = pos_interp[:, j]

        self._dense = dense

        print(
            f"[ALLEX][Traj] loaded groups={self._groups_used} "
            f"duration={max_dur:.2f}s samples={n_steps + 1} hz={self._hz}"
        )
        if self._missing_joints:
            print(
                f"[ALLEX][Traj] unmatched joint names (skipped): "
                f"{sorted(set(self._missing_joints))}"
            )

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
        if not self.is_ready():
            return False
        self._active = True
        self._finished = False
        self._sample_idx = 0
        return True

    def stop(self) -> None:
        self._active = False

    def get_current_target(self) -> np.ndarray | None:
        """Return the current target row and advance one sample.

        When the last sample is reached the player stays there (hold) and
        `is_finished()` flips to True. Returns None if no trajectory is loaded
        or playback is not active.
        """
        if not self.is_active() or self._dense is None:
            return None
        last = self._dense.shape[0] - 1
        idx = min(self._sample_idx, last)
        target = self._dense[idx]
        if self._sample_idx >= last:
            self._finished = True
        else:
            self._sample_idx += 1
        return target
