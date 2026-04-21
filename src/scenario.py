"""
ALLEX Digital Twin 메인 시나리오 — Real2Sim 관절 위치 매핑
"""

import logging
import os
from .core import (
    ALLEXInitializer,
    ALLEXAssetManager,
    ALLEXJointController,
    ALLEXSimulationLoop,
    ForceTorqueVisualizer,
)


logger = logging.getLogger("allex.scenario")


class ALLEXDigitalTwin:
    """ALLEX Real2Sim 디지털 트윈 — ROS2 관절 데이터로 시뮬레이션 로봇 구동"""

    def __init__(self):
        self._initializer = ALLEXInitializer()
        self._asset_manager = ALLEXAssetManager()
        self._joint_controller = ALLEXJointController()
        self._simulation_loop = ALLEXSimulationLoop()
        self._articulation = None
        self._ros2_manager = None
        self._visualizer = None

    def setup(self):
        """시나리오 초기 설정 — 카메라 뷰 + coupled joint config 로드"""
        self._initializer.setup_camera_view()

        extension_root = os.path.dirname(os.path.abspath(__file__))
        joint_config_path = os.path.join(extension_root, "joint_config.json")
        self._joint_controller.load_coupled_joint_config(joint_config_path)

        print("ALLEX Digital Twin setup complete")

    def load_example_assets(self):
        """로봇 USD 에셋 로딩 및 제너레이터 설정.

        visualizer prim 은 asset load 직후(= articulation 초기화 전) 바로 생성한다.
        Play 전에도 torque ring / force arrow prim 이 stage 에 존재해야 visibility 토글이
        의미가 있다.
        """
        self._articulation = self._asset_manager.load_robot_asset()

        if self._articulation is None:
            return None

        # 1단계: asset load 직후 — stage 기반 prim 즉시 생성 (articulation 불필요)
        self._create_visualizer_prims()

        init_success = self._asset_manager.initialize_articulation()
        if init_success:
            self._setup_joint_control_generator()
            # 2단계: articulation 초기화 완료 → LUT 만 attach
            self._attach_visualizer_articulation()
        else:
            print("RUN 버튼을 누르면 Articulation이 초기화됩니다")

        return self._articulation

    def delayed_initialization(self):
        """지연 초기화 — 시뮬레이션 시작 후 Articulation 초기화"""
        if self._articulation is None:
            return False

        init_success = self._asset_manager.initialize_articulation()
        if not init_success:
            return False

        self._initializer.initialize_joint_positions(self._articulation)
        self._setup_joint_control_generator()
        # 혹시 prim 이 아직 없으면 생성, 그리고 articulation attach
        if self._visualizer is None:
            self._create_visualizer_prims()
        self._attach_visualizer_articulation()
        print("Articulation 초기화 완료")
        return True

    def reset(self):
        """시스템 리셋 — prim 은 유지, articulation 재초기화 시점에 재-attach 만 수행."""
        self._initializer.reset(self._articulation)
        self._simulation_loop.reset()

        # visualizer prim 은 cleanup 하지 않는다. articulation 만 detach 후 재 attach.
        if self._visualizer is not None:
            try:
                self._visualizer.detach_articulation()
            except Exception as e:
                logger.warning(f"visualizer detach warn: {e}")

        if self._articulation is not None:
            self._setup_joint_control_generator()
            # prim 이 없으면 안전하게 재생성
            if self._visualizer is None:
                self._create_visualizer_prims()
            self._attach_visualizer_articulation()

        print("시스템 리셋 완료")

    def update(self, step: float):
        """매 physics step마다 호출"""
        done = self._simulation_loop.update(step)
        if self._visualizer is not None:
            try:
                self._visualizer.update(step)
            except Exception as e:
                logger.warning(f"visualizer update warn: {e}")
        return done

    # ========================================
    # Visualizer 접근자 (UI 에서 사용)
    # ========================================
    def get_visualizer(self):
        return self._visualizer

    def _create_visualizer_prims(self):
        """1단계: articulation 없이 stage 기반 prim 만 생성."""
        if self._visualizer is not None:
            return
        try:
            self._visualizer = ForceTorqueVisualizer(stage=None)
            # 생성자 안에서 self._ensure_prims 가 호출됨 — 여기서는 확인만.
            self._visualizer.ensure_initialized()
            print("ForceTorqueVisualizer prims created (pre-articulation)")
        except Exception as e:
            logger.warning(f"ForceTorqueVisualizer prim create failed: {e}")
            self._visualizer = None

    def _attach_visualizer_articulation(self):
        """2단계: articulation 준비 완료 후 LUT 만 구축."""
        if self._visualizer is None:
            # 1단계를 놓친 경우 방어적으로 생성
            self._create_visualizer_prims()
            if self._visualizer is None:
                return
        if self._articulation is None:
            return
        try:
            self._visualizer.attach_articulation(
                self._articulation, self._joint_controller,
            )
            print("ForceTorqueVisualizer articulation attached")
        except Exception as e:
            logger.warning(f"ForceTorqueVisualizer attach failed: {e}")

    # 호환용 shim — 예전 코드 경로에서 호출되는 경우 대비
    def _setup_visualizer(self):
        if self._visualizer is None:
            self._create_visualizer_prims()
        if self._articulation is not None:
            self._attach_visualizer_articulation()

    # ========================================
    # 내부 헬퍼
    # ========================================
    def _setup_joint_control_generator(self):
        """관절 제어 제너레이터 생성 및 시뮬레이션 루프에 설정"""
        self._initializer.initialize_joint_positions(self._articulation)

        def get_target_positions():
            ros2_positions = self._joint_controller.get_unified_target_positions()
            if ros2_positions and any(pos != 0.0 for pos in ros2_positions):
                return ros2_positions
            return self._initializer.target_joint_positions

        generator = self._joint_controller.create_joint_control_generator(
            articulation=self._articulation,
            get_target_positions_func=get_target_positions,
        )
        self._simulation_loop.set_script_generator(generator)

    # ========================================
    # Public API
    # ========================================
    def set_ros2_manager(self, ros2_manager):
        self._ros2_manager = ros2_manager

    def get_robot_info(self):
        return self._asset_manager.get_joint_info()

    def is_simulation_running(self):
        return self._simulation_loop.is_running()

    def stop_simulation(self):
        self._simulation_loop.stop()
