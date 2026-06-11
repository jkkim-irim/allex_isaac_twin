# SPDX-License-Identifier: Apache-2.0
"""Monotonic cubic Hermite spline trajectory generation.

Via point CSV (duration, joint_1, ..., joint_N) 에서
시간 균일 샘플링된 궤적을 생성.

C++ monotonic_cubic_spline / trajectory_manager와 동일한 로직.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPS = 1e-9
_HOLD_TOL_RAD = 1e-4  # P1≈P2이면 홀드(상수)


@dataclass
class ViaCSVData:
    """Via point CSV 파싱 결과.

    position 외에 via별 torque limit / PD gain 컬럼(있을 때만)을 같은 레이아웃
    (N, n_joints) 배열로 노출합니다. 해당 via 행에 값이 생략된 경우 NaN.

    단위는 CSV 원본을 그대로 유지 (deg → rad 변환은 position에만 적용).
    """

    t_via: np.ndarray                    # (N,) seconds
    pos_via: np.ndarray                  # (N, n_joints) rad
    kps_via: np.ndarray | None = None    # (N, n_joints) CSV units
    kds_via: np.ndarray | None = None    # (N, n_joints) CSV units
    trq_via: np.ndarray | None = None    # (N, n_joints) CSV units
    sections: list[str] | None = None    # (N,) section label per via, from "-1, <name>" markers


# ── Hermite basis functions (and their derivatives wrt u) ──
def _h00(u: np.ndarray) -> np.ndarray:
    return 2.0 * u**3 - 3.0 * u**2 + 1.0


def _h10(u: np.ndarray) -> np.ndarray:
    return u**3 - 2.0 * u**2 + u


def _h01(u: np.ndarray) -> np.ndarray:
    return -2.0 * u**3 + 3.0 * u**2


def _h11(u: np.ndarray) -> np.ndarray:
    return u**3 - u**2


def _h00p(u: np.ndarray) -> np.ndarray:
    return 6.0 * u**2 - 6.0 * u


def _h10p(u: np.ndarray) -> np.ndarray:
    return 3.0 * u**2 - 4.0 * u + 1.0


def _h01p(u: np.ndarray) -> np.ndarray:
    return -6.0 * u**2 + 6.0 * u


def _h11p(u: np.ndarray) -> np.ndarray:
    return 3.0 * u**2 - 2.0 * u


def _hermite_1d(
    t_out: np.ndarray, t_via: np.ndarray, y_via: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """1D monotonic cubic Hermite spline (position + analytic velocity).

        pos = h00*P1 + h10*T1 + h01*P2 + h11*T2
        vel = (h00'*P1 + h10'*T1 + h01'*P2 + h11'*T2) / seg_dur
    where ``T1 = m1*seg_dur``, ``T2 = m2*seg_dur`` and ``m1, m2`` are the
    monotonic Fritsch–Carlson tangents (in pos/time).

    Args:
        t_out: 출력 시간 배열 [s].
        t_via: via point 시간 배열 [s] (오름차순).
        y_via: via point 값 배열 [rad].

    Returns:
        (pos, vel) — 위치 [rad] 와 속도 [rad/s]. Hold 구간 / 범위 밖에서는
        속도 0.
    """
    n = len(t_via)
    pos = np.empty_like(t_out)
    vel = np.zeros_like(t_out)

    if n == 0:
        pos[:] = 0.0
        return pos, vel
    if n == 1:
        pos[:] = y_via[0]
        return pos, vel

    for i in range(n - 1):
        t0, t1 = t_via[i], t_via[i + 1]
        seg_dur = max(t1 - t0, _EPS)
        mask = (t_out >= t0) & (t_out <= t1)
        if not np.any(mask):
            continue

        P1 = float(y_via[i])
        P2 = float(y_via[i + 1])

        # P1 ≈ P2이면 홀드 (vel = 0; 이미 초기화돼 있음)
        if abs(P2 - P1) <= _HOLD_TOL_RAD:
            pos[mask] = P1
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

        # monotonic tangent (pos/time)
        m1 = 0.5 * (s01 + s12) if s01 * s12 > 0.0 else 0.0
        m2 = 0.5 * (s12 + s23) if s12 * s23 > 0.0 else 0.0

        T1, T2 = m1 * seg_dur, m2 * seg_dur
        seg_pos = _h00(u) * P1 + _h10(u) * T1 + _h01(u) * P2 + _h11(u) * T2
        seg_vel = (
            _h00p(u) * P1 + _h10p(u) * T1 + _h01p(u) * P2 + _h11p(u) * T2
        ) / seg_dur

        if not np.all(np.isfinite(seg_pos)):
            seg_pos = np.where(np.isfinite(seg_pos), seg_pos, P1 + (P2 - P1) * u)
            seg_vel = np.where(np.isfinite(seg_vel), seg_vel, (P2 - P1) / seg_dur)
        pos[mask] = seg_pos
        vel[mask] = seg_vel

    # 범위 밖: 위치 hold, 속도 0 (이미 0으로 초기화)
    pos[t_out <= t_via[0]] = y_via[0]
    pos[t_out >= t_via[-1]] = y_via[-1]
    return pos, vel


def _parse_float(s: str) -> float:
    """Parse one CSV cell. Empty/missing → NaN. Non-numeric → raises."""
    s = s.strip()
    if not s:
        return float("nan")
    return float(s)


def parse_via_csv(path: str | Path) -> ViaCSVData:
    """Via point CSV 파싱.

    지원 헤더 컬럼 (대소문자 구분, 순서 자유):
        duration, joint_1..N, trq_lim_1..N, K_pos_1..N, K_vel_1..N

    - **Multi-section 헤더 지원**: ``duration``으로 시작하는 라인을 만나면
      그 시점부터 새 섹션 schema 로 전환. 즉 한 파일 안에서 trq_lim 만 있는
      섹션과 K_pos/K_vel 만 있는 섹션을 섞어 쓸 수 있음. 이전에는 첫 헤더만
      읽혀 후속 섹션의 추가 컬럼이 모두 버려졌음.
    - 행이 현재 섹션에서 정의되지 않은 카테고리(예: K_pos 컬럼이 없는
      섹션의 PD gain)는 NaN 으로 기록되어, 후속 이벤트 추출 시 ``None`` 으로
      정규화되거나 NaN 마스크로 적용에서 제외됨.
    - duration <= 0 인 행은 주석/섹션 마커로 무시 (예: ``-1,팔짱 [9.0 sec]``).
    - 첫 데이터 행의 duration 은 초기 hold 시간 (time[0] = 0 고정).
    """
    path = Path(path)

    rows_dur: list[float] = []
    rows_pos: list[list[float]] = []
    rows_trq: list[list[float]] = []
    rows_kp: list[list[float]] = []
    rows_kd: list[list[float]] = []
    # Section labels for each via row. `-1, <label>` lines mark END of the
    # preceding section. We buffer the unlabeled count and back-fill on hit.
    rows_section: list[str | None] = []
    unlabeled_since: int = 0  # index of first via in current (unlabeled) section

    # Section-local schema. Reset on every "duration,..." header line.
    joint_cols: list[int] = []
    trq_cols: list[int] = []
    kp_cols: list[int] = []
    kd_cols: list[int] = []
    n_joints: int | None = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue

            # Header (or re-header for new section)
            if parts[0] == "duration":
                joint_cols = []
                trq_cols = []
                kp_cols = []
                kd_cols = []
                for i, name in enumerate(parts):
                    if name.startswith("joint_"):
                        joint_cols.append(i)
                    elif name.startswith("trq_lim_"):
                        trq_cols.append(i)
                    elif name.startswith("K_pos_"):
                        kp_cols.append(i)
                    elif name.startswith("K_vel_"):
                        kd_cols.append(i)
                if n_joints is None:
                    n_joints = len(joint_cols)
                continue

            if not joint_cols:
                # No header seen yet; cannot interpret data.
                continue

            try:
                d = float(parts[0])
            except ValueError:
                continue
            # Section marker: "-1, <label>". Back-fill section label for all
            # rows since the last marker.
            if d < 0:
                if len(parts) > 1:
                    label = parts[1].strip()
                    # Strip trailing duration annotation like "[9.0 sec]"
                    label = label.split("[")[0].strip()
                    for k in range(unlabeled_since, len(rows_section)):
                        rows_section[k] = label
                unlabeled_since = len(rows_section)
                continue
            if d == 0:
                continue

            def cell(idx: int) -> float:
                return _parse_float(parts[idx]) if idx < len(parts) else float("nan")

            pos_row = [cell(c) for c in joint_cols]
            if any(np.isnan(v) for v in pos_row):
                # position은 필수 — 결측 행은 스킵.
                continue

            n = n_joints or len(joint_cols)

            def cells(cols: list[int]) -> list[float]:
                if not cols:
                    return [float("nan")] * n
                return [cell(c) for c in cols]

            rows_dur.append(d)
            rows_pos.append(pos_row)
            rows_trq.append(cells(trq_cols))
            rows_kp.append(cells(kp_cols))
            rows_kd.append(cells(kd_cols))
            rows_section.append(None)  # filled in on next -1 marker

    if not rows_pos:
        return ViaCSVData(t_via=np.array([0.0]), pos_via=np.zeros((1, 1)))

    durations = np.asarray(rows_dur, dtype=np.float64)
    # C++ 규약: row k의 duration = segment (k-1)→(k) 소요 시간
    # time[0] = 0, time[i] = sum(duration[1:i+1])
    t_via = np.concatenate([[0.0], np.cumsum(durations[1:])])

    pos_via = np.deg2rad(np.asarray(rows_pos, dtype=np.float32))

    def _arr_or_none(rows: list[list[float]]) -> "np.ndarray | None":
        arr = np.asarray(rows, dtype=np.float32)
        return None if arr.size == 0 or np.all(np.isnan(arr)) else arr

    return ViaCSVData(
        t_via=t_via,
        pos_via=pos_via,
        trq_via=_arr_or_none(rows_trq),
        kps_via=_arr_or_none(rows_kp),
        kds_via=_arr_or_none(rows_kd),
        sections=rows_section,
    )


def _resolve_hz(hz: float | None) -> float:
    """Return ``hz`` if given, otherwise fall back to ``physics_config.json``.

    Single source of truth: ``world.physics_hz`` in JSON. Pass an explicit
    ``hz`` only when you intentionally want a non-physics rate (e.g. coarse
    preview).
    """
    if hz is not None and hz > 0:
        return float(hz)
    from ..utils.sim_settings_utils import get_physics_hz
    return get_physics_hz()


def generate_trajectory(
    time_via: np.ndarray,
    pos_via_rad: np.ndarray,
    hz: float | None = None,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Via points에서 시간 균일 궤적 생성.

    Args:
        time_via: via point 시간 [s], shape (N,).
        pos_via_rad: via point 위치 [rad], shape (N, n_joints).
        hz: 출력 샘플링 주파수 [Hz]. ``None`` 이면
            ``physics_config.json::world.physics_hz`` 값 사용 (단일 소스).
        duration: 출력 궤적 길이 [s]. None이면 via point 끝까지.

    Returns:
        (time_out, pos_out_rad, vel_out_rad_s): 균일 샘플링된 시간, 위치, 속도.
        속도는 Hermite 기저의 해석적 미분으로 계산되어 수치 차분 노이즈 없음.
    """
    hz = _resolve_hz(hz)

    if duration is None:
        duration = float(time_via[-1])

    dt = 1.0 / hz
    n_steps = int(duration / dt)
    t_out = np.arange(n_steps + 1) * dt
    n_joints = pos_via_rad.shape[1]

    pos_out = np.zeros((n_steps + 1, n_joints), dtype=np.float32)
    vel_out = np.zeros((n_steps + 1, n_joints), dtype=np.float32)
    for j in range(n_joints):
        p, v = _hermite_1d(t_out, time_via, pos_via_rad[:, j])
        pos_out[:, j] = p
        vel_out[:, j] = v

    return t_out.astype(np.float32), pos_out, vel_out


def generate_trajectory_from_csv(
    csv_path: str | Path,
    hz: float | None = None,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSV 파일에서 직접 궤적 생성.

    Args:
        csv_path: via point CSV 경로.
        hz: 출력 샘플링 주파수 [Hz]. ``None`` 이면
            ``physics_config.json::world.physics_hz`` 값 사용.
        duration: 출력 궤적 길이 [s]. None이면 via point 끝까지.

    Returns:
        (time_out, pos_out_rad, vel_out_rad_s): 시간 [s], 위치 [rad], 속도 [rad/s].
    """
    data = parse_via_csv(csv_path)
    return generate_trajectory(data.t_via, data.pos_via, hz=hz, duration=duration)
