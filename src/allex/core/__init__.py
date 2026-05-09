"""ALLEX Digital Twin core 서브시스템 — scenario.py 가 호출하는 도메인 layer.

모듈별 역할:
    initialization.py   — ALLEXInitializer (카메라뷰, 초기 자세, reset)
    asset_manager.py    — ALLEXAssetManager (USD 로딩, articulation 핸들 관리)
    joint_controller.py — ALLEXJointController (관절 제어 generator,
                          coupled joint 처리, pre_step_fn 훅 — motor mirror 호출지점)
    simulation_loop.py  — ALLEXSimulationLoop (Isaac Sim physics step 콜백 등록)
    newton_bridge.py    — Newton ModelBuilder finalize 후크
                          (MJCF equality 주입, mjc:gravcomp 인증, actuator routing)
    gravcomp_debug.py   — GravcompTorqueProbe (PD/Grav/Sum 분리 출력 진단)

호출 방향:
    scenario.py → core/* (단방향). UI 는 core 직접 import 금지.
"""

from .initialization import ALLEXInitializer
from .asset_manager import ALLEXAssetManager
from .joint_controller import ALLEXJointController
from .simulation_loop import ALLEXSimulationLoop

__all__ = [
    'ALLEXInitializer',
    'ALLEXAssetManager',
    'ALLEXJointController',
    'ALLEXSimulationLoop',
]
