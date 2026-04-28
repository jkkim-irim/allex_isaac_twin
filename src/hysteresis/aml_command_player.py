"""AML command trajectory 전용 dense replay player.

`command_joint_deg_0.csv` (columns: `t, abad_deg, mcp_deg, pip_deg`) 를
sim clock 기반 linear interpolation 으로 재생한다. 기존 Hermite via-point
방식의 `TrajectoryPlayer` 와 달리 원본 dense sample (~10 ms stride) 사이를
cubic spline 없이 선형만 채워 real command 를 충실히 복원한다.

Consumer API (`HysteresisScenario.set_trajectory_player` 호환):
    is_ready(), start(), stop(), is_active(), is_finished(),
    get_current_target() -> np.ndarray | None, duration_s, groups_used
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class AMLCommandPlayer:
    """절대시간 기반 linear-interp replay.

    3-단계 상태머신:
      1. ramp_in_s   동안 live seed pose → CSV 첫 row 로 선형 램프.
      2. post_ramp_hold_s 동안 CSV 첫 row (cmd[0]) 를 유지 — PD 가 정착하고
         측정/커맨드 타이밍이 깨끗이 리셋될 여유.
      3. 이후 매 step `sim_t - (ramp_in_s + post_ramp_hold_s)` 로 CSV 에
         searchsorted + lerp. CSV 끝에 도달하면 마지막 target 을 hold
         (finished=True).
    R_Index 3 DOF 외의 관절은 seed_pose 그대로 고정.
    """

    def __init__(
        self,
        csv_path: Path | str,
        articulation,
        joint_names: list[str],
        physics_hz: float,
        seed_pose: np.ndarray | None = None,
        ramp_in_s: float = 3.0,
        post_ramp_hold_s: float = 1.0,
    ):
        self._csv_path = Path(csv_path)
        self._articulation = articulation
        self._joint_names = list(joint_names)
        self._physics_hz = float(physics_hz)
        self._dt = 1.0 / self._physics_hz
        self._ramp_in_s = float(ramp_in_s)
        self._post_ramp_hold_s = float(post_ramp_hold_s)
        self._playback_start_s = self._ramp_in_s + self._post_ramp_hold_s
        self._seed_pose_override = seed_pose

        self._dof_names: list[str] = list(getattr(articulation, "dof_names", []) or [])
        self._num_dof = len(self._dof_names)
        name_to_idx = {n: i for i, n in enumerate(self._dof_names)}
        self._target_indices = np.asarray(
            [name_to_idx[n] for n in self._joint_names if n in name_to_idx],
            dtype=np.int64,
        )
        missing = [n for n in self._joint_names if n not in name_to_idx]
        if missing:
            print(f"[Hysteresis][AML] missing DOFs (skipped): {missing}")

        self._cmd_t: np.ndarray = np.empty(0, dtype=np.float64)
        self._cmd_pos_rad: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._seed_pose: np.ndarray = np.zeros(self._num_dof, dtype=np.float32)
        self._t0_abs: float = 0.0
        self._duration_s: float = 0.0

        self._step_count: int = 0
        self._active: bool = False
        self._finished: bool = False

        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self) -> None:
        if self._num_dof <= 0:
            print("[Hysteresis][AML] articulation has no dof_names yet")
            return
        if self._target_indices.size != len(self._joint_names):
            print(
                f"[Hysteresis][AML] target joint count mismatch: "
                f"{self._target_indices.size}/{len(self._joint_names)} — player disabled"
            )
            return

        t_list: list[float] = []
        pos_list: list[list[float]] = []
        with open(self._csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or len(header) < 1 + len(self._joint_names):
                print(f"[Hysteresis][AML] unexpected header: {header}")
                return
            n_joints = len(self._joint_names)
            for row in reader:
                if len(row) < 1 + n_joints:
                    continue
                t_list.append(float(row[0]))
                pos_list.append([float(row[1 + j]) for j in range(n_joints)])

        if not t_list:
            print(f"[Hysteresis][AML] empty CSV: {self._csv_path}")
            return

        t_arr = np.asarray(t_list, dtype=np.float64)
        pos_deg = np.asarray(pos_list, dtype=np.float32)

        self._t0_abs = float(t_arr[0])
        self._cmd_t = t_arr - self._t0_abs
        self._cmd_pos_rad = np.deg2rad(pos_deg).astype(np.float32)
        self._duration_s = self._playback_start_s + float(self._cmd_t[-1])

        live = self._read_live_pose()
        if self._seed_pose_override is not None:
            seed = self._coerce_to_numpy(self._seed_pose_override)
        elif live is not None:
            seed = live
        else:
            seed = np.zeros(self._num_dof, dtype=np.float32)
        self._seed_pose = seed.astype(np.float32, copy=True)

        print(
            f"[Hysteresis][AML] loaded {self._csv_path.name}: "
            f"{self._cmd_t.shape[0]} rows, span={self._cmd_t[-1]:.2f}s, "
            f"t0_abs={self._t0_abs:.3f}s, ramp_in={self._ramp_in_s:.1f}s, "
            f"post_ramp_hold={self._post_ramp_hold_s:.1f}s, "
            f"physics_hz={self._physics_hz:.1f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_live_pose(self) -> np.ndarray | None:
        pose = self._articulation.get_joint_positions()
        if pose is None:
            return None
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        return arr if arr.size == self._num_dof else None

    @staticmethod
    def _coerce_to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _interp_cmd(self, t_rel: float) -> np.ndarray:
        """CSV 에서 `t_rel` 시점의 R_Index 3-DOF 위치 (rad) 반환."""
        cmd_t = self._cmd_t
        if t_rel <= 0.0:
            return self._cmd_pos_rad[0]
        if t_rel >= cmd_t[-1]:
            return self._cmd_pos_rad[-1]
        i = int(np.searchsorted(cmd_t, t_rel, side="right")) - 1
        i = max(0, min(i, cmd_t.shape[0] - 2))
        t0, t1 = cmd_t[i], cmd_t[i + 1]
        span = t1 - t0
        alpha = 0.0 if span <= 0.0 else (t_rel - t0) / span
        return (1.0 - alpha) * self._cmd_pos_rad[i] + alpha * self._cmd_pos_rad[i + 1]

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return self._cmd_t.size > 0 and self._target_indices.size == len(self._joint_names)

    def is_active(self) -> bool:
        return self._active and self.is_ready()

    def is_finished(self) -> bool:
        return self._finished

    def is_ramp_done(self) -> bool:
        """True on the first tick *after* the seed→cmd[0] ramp phase.

        `_step_count` 은 매 get_current_target() 이후 증가하므로, 다음 틱의
        sim_t 는 `step_count*dt` 가 된다. 램프 마지막 틱(step=N, sim_t=N*dt
        가 ramp_in 과 같은 틱) 은 아직 램프 target 을 썼다고 봐야 하므로
        `>=` 대신 `>` 를 사용해 한 틱 뒤로 밀어 "첫 post-ramp 틱" 을 잡는다.
        """
        if not self._active:
            return False
        return self._step_count * self._dt > self._ramp_in_s

    def is_playback_started(self) -> bool:
        """True on the first tick *after* (ramp + post_ramp_hold) 구간.

        Hold 기간에는 cmd[0] 를 유지하며 PD 가 정착하도록 두고, 이 함수가
        True 로 바뀌는 순간이 실제 CSV 재생 시작 틱이자 측정 로깅 시작 틱.
        """
        if not self._active:
            return False
        return self._step_count * self._dt > self._playback_start_s

    @property
    def ramp_in_s(self) -> float:
        return self._ramp_in_s

    @property
    def post_ramp_hold_s(self) -> float:
        return self._post_ramp_hold_s

    @property
    def playback_start_s(self) -> float:
        """ramp + hold 합계. 이 시점부터 CSV 가 재생된다 (sim time)."""
        return self._playback_start_s

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def groups_used(self) -> list[str]:
        return [self._csv_path.stem]

    def start(self) -> bool:
        if not self.is_ready():
            return False
        self._step_count = 0
        self._active = True
        self._finished = False
        return True

    def stop(self) -> None:
        self._active = False

    def get_current_target(self) -> np.ndarray | None:
        if not self.is_active():
            return None

        sim_t = self._step_count * self._dt
        self._step_count += 1

        target = self._seed_pose.copy()

        # Phase 1: seed → cmd[0] 선형 램프
        if sim_t < self._ramp_in_s and self._ramp_in_s > 0.0:
            alpha = sim_t / self._ramp_in_s
            first_cmd = self._cmd_pos_rad[0]
            seed_at_idx = self._seed_pose[self._target_indices]
            target[self._target_indices] = (
                (1.0 - alpha) * seed_at_idx + alpha * first_cmd
            )
            return target

        # Phase 2: cmd[0] hold (PD settling window)
        if sim_t < self._playback_start_s:
            target[self._target_indices] = self._cmd_pos_rad[0]
            return target

        # Phase 3: CSV playback (t_rel 기준 선형보간)
        t_rel = sim_t - self._playback_start_s
        target[self._target_indices] = self._interp_cmd(t_rel)

        if t_rel >= self._cmd_t[-1]:
            self._finished = True

        return target
