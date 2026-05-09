"""ALLEX Digital Twin 유틸리티 모듈 — config 로더 + 도메인 격리 모듈.

JSON 설정 + 적용 함수 패턴 (dvcc/10_arch_decisions §1):
    sim_settings_utils.py   — physics_config.json 적용 (Newton, MuJoCo solver,
                              physicsScene, gravcomp body, motor_step_per_ms,
                              nominal_motor_gains 로더)
    ros2_settings_utils.py  — ros2_config.json 적용 (topic 매핑, joint groups)
    ui_settings_utils.py    — ui_config.json 적용 + UI 헬퍼 클래스
                              (UIComponentFactory, ButtonStyleManager 등;
                              순환 import 방지 차원의 중립 위치)

도메인 격리 모듈 (UI 패널 아닌 도메인 서브시스템):
    contact_force_viz.py    — MuJoCo contact 데이터 → debug_draw 시각화

상수:
    constants.py            — TOTAL_JOINTS / EFFECTIVE_JOINTS 등 invariant.
"""

from .constants import *
from . import sim_settings_utils
from . import ros2_settings_utils
from . import ui_settings_utils
from . import contact_force_viz

__all__ = [
    'TOTAL_JOINTS',
    'EFFECTIVE_JOINTS',
    'DEFAULT_JOINT_POSITIONS',
    'sim_settings_utils',
    'ros2_settings_utils',
    'ui_settings_utils',
    'contact_force_viz',
]