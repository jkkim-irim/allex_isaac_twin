# SPDX-License-Identifier: Apache-2.0
"""Monotonic cubic Hermite spline trajectory generation.

Via point CSV (duration, joint_1, ..., joint_N) 에서
시간 균일 샘플링된 궤적을 생성.

C++ monotonic_cubic_spline / trajectory_manager와 동일한 로직.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_EPS = 1e-9
_HOLD_TOL_RAD = 1e-4  # P1≈P2이면 홀드(상수)


# ── Hermite basis functions ──
def _h00(u: np.ndarray) -> np.ndarray:
    return 2.0 * u**3 - 3.0 * u**2 + 1.0


def _h10(u: np.ndarray) -> np.ndarray:
    return u**3 - 2.0 * u**2 + u


def _h01(u: np.ndarray) -> np.ndarray:
    return -2.0 * u**3 + 3.0 * u**2


def _h11(u: np.ndarray) -> np.ndarray:
    return u**3 - u**2


def _hermite_1d(t_out: np.ndarray, t_via: np.ndarray, y_via: np.ndarray) -> np.ndarray:
    """1D monotonic cubic Hermite spline.

    Args:
        t_out: 출력 시간 배열 [s].
        t_via: via point 시간 배열 [s] (오름차순).
        y_via: via point 값 배열 [rad].

    Returns:
        t_out에 대응하는 보간된 값 배열.
    """
    n = len(t_via)
    out = np.empty_like(t_out)

    if n == 0:
        out[:] = 0.0
        return out
    if n == 1:
        out[:] = y_via[0]
        return out

    for i in range(n - 1):
        t0, t1 = t_via[i], t_via[i + 1]
        seg_dur = max(t1 - t0, _EPS)
        mask = (t_out >= t0) & (t_out <= t1)
        if not np.any(mask):
            continue

        P1 = float(y_via[i])
        P2 = float(y_via[i + 1])

        # P1 ≈ P2이면 홀드
        if abs(P2 - P1) <= _HOLD_TOL_RAD:
            out[mask] = P1
            continue

        u = (t_out[mask] - t0) / seg_dur

        # 이웃 포인트
        P0 = float(y_via[i - 1]) if i > 0 else P1
        P3 = float(y_via[i + 2]) if i + 2 < n else P2

        d01 = max(t_via[i] - t_via[i - 1], _EPS) if i > 0 else seg_dur
        d23 = max(t_via[i + 2] - t_via[i + 1], _EPS) if i + 2 < n else seg_dur

        s01 = (P1 - P0) / d01 if d01 > _EPS else 0.0
        s12 = (P2 - P1) / seg_dur
        s23 = (P3 - P2) / d23 if d23 > _EPS else 0.0

        # monotonic tangent
        m1 = 0.5 * (s01 + s12) if s01 * s12 > 0.0 else 0.0
        m2 = 0.5 * (s12 + s23) if s12 * s23 > 0.0 else 0.0

        T1, T2 = m1 * seg_dur, m2 * seg_dur
        pos = _h00(u) * P1 + _h10(u) * T1 + _h01(u) * P2 + _h11(u) * T2

        if not np.all(np.isfinite(pos)):
            pos = np.where(np.isfinite(pos), pos, P1 + (P2 - P1) * u)
        out[mask] = pos

    # 범위 밖
    out[t_out <= t_via[0]] = y_via[0]
    out[t_out >= t_via[-1]] = y_via[-1]
    return out


def parse_via_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Via point CSV 파싱.

    CSV format: duration, joint_1, joint_2, ..., joint_N (degree)
    duration <= 0 인 행은 주석으로 무시.
    첫 행의 duration은 초기 hold 시간.

    Returns:
        (time_arr, position_rad): via point 시간[s]과 위치[rad].
    """
    path = Path(path)
    rows = []
    n_cols = None

    with open(path, "r", encoding="utf-8") as f:
        header = next(f)  # skip header
        # joint 열만 카운트 (K_pos, K_vel 등 제외)
        header_parts = [p.strip() for p in header.split(",") if p.strip()]
        n_joint_cols = sum(1 for p in header_parts if p.startswith("joint_") or p == "duration")
        if n_joint_cols < 2:
            n_joint_cols = len(header_parts)
        n_cols = n_joint_cols

        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                d = float(parts[0])
            except ValueError:
                continue
            if d <= 0:
                continue
            try:
                row = [float(parts[i]) for i in range(min(len(parts), n_cols))]
            except (ValueError, IndexError):
                continue
            if len(row) == n_cols:
                rows.append(row)

    if not rows:
        return np.array([0.0]), np.zeros((1, 1))

    arr = np.array(rows)
    n_joints = arr.shape[1] - 1
    durations = arr[:, 0]

    # C++ 규약: row k의 duration = segment (k-1)→(k) 소요 시간
    # time[0] = 0, time[i] = sum(duration[1:i+1])
    time_arr = np.concatenate([[0.0], np.cumsum(durations[1:])])
    pos_deg = arr[:, 1: 1 + n_joints]
    pos_rad = np.deg2rad(pos_deg)

    return time_arr, pos_rad


def generate_trajectory(
    time_via: np.ndarray,
    pos_via_rad: np.ndarray,
    hz: float = 200.0,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Via points에서 시간 균일 궤적 생성.

    Args:
        time_via: via point 시간 [s], shape (N,).
        pos_via_rad: via point 위치 [rad], shape (N, n_joints).
        hz: 출력 샘플링 주파수 [Hz].
        duration: 출력 궤적 길이 [s]. None이면 via point 끝까지.

    Returns:
        (time_out, pos_out_rad): 균일 샘플링된 시간과 위치.
    """
    if duration is None:
        duration = float(time_via[-1])

    dt = 1.0 / hz
    n_steps = int(duration / dt)
    t_out = np.arange(n_steps + 1) * dt
    n_joints = pos_via_rad.shape[1]

    pos_out = np.zeros((n_steps + 1, n_joints), dtype=np.float32)
    for j in range(n_joints):
        pos_out[:, j] = _hermite_1d(t_out, time_via, pos_via_rad[:, j])

    return t_out.astype(np.float32), pos_out


def generate_trajectory_from_csv(
    csv_path: str | Path,
    hz: float = 200.0,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CSV 파일에서 직접 궤적 생성.

    Returns:
        (time_out, pos_out_rad): 균일 샘플링된 시간[s]과 위치[rad].
    """
    time_via, pos_via = parse_via_csv(csv_path)
    return generate_trajectory(time_via, pos_via, hz=hz, duration=duration)
