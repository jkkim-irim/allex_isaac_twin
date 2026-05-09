"""Trajectory 재생 + 모터-도메인 미러 + sim 상태 로거 패키지.

모듈별 역할:
    hermite_spline.py        — CSV via points → 1 kHz 보간 dense trajectory
    joint_name_map.py        — CSV 파일명 → joint 이름 리스트 매핑
    trajectory_player.py     — TrajectoryPlayer (재생 + 게인 이벤트 → mirror)
    motor_state_mirror.py    — MotorStateMirror (motor↔joint warp kernel host)
    motor_joint_transform.py — ground-truth 다항식 (numpy reference)
    jk_kernel.py             — warp kernel 5종 (scalar/elbow/wrist/finger/thumb)
    sim_state_logger.py      — SimStateLogger (q/τ/K/limit CSV 기록)

데이터 흐름:
    CSV (via points + K_pos/K_vel/τ_lim)
        → hermite_spline.parse_via_csv
        → TrajectoryPlayer  (dense + events)
        → MotorStateMirror.set_target  (모터-도메인 ramp target)
        → jk_kernel.*  (매 step 실시간 motor→joint 변환)
        → Newton joint_target_ke/kd/effort_limit  (zero-copy view write)

Public API:
    TrajectoryPlayer, MotorStateMirror, SimStateLogger,
    parse_via_csv, generate_trajectory, ALLEX_CSV_JOINT_NAMES.
"""
from .hermite_spline import generate_trajectory, generate_trajectory_from_csv, parse_via_csv
from .joint_name_map import ALLEX_CSV_JOINT_NAMES
from .motor_state_mirror import MotorStateMirror
from .sim_state_logger import SimStateLogger
from .trajectory_player import TrajectoryPlayer

__all__ = [
    "ALLEX_CSV_JOINT_NAMES",
    "MotorStateMirror",
    "SimStateLogger",
    "TrajectoryPlayer",
    "generate_trajectory",
    "generate_trajectory_from_csv",
    "parse_via_csv",
]
