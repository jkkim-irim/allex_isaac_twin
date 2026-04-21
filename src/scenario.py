"""
ALLEX Digital Twin 메인 시나리오 — Real2Sim 관절 위치 매핑
"""

import os
from .core import ALLEXInitializer, ALLEXAssetManager, ALLEXJointController, ALLEXSimulationLoop


class ALLEXDigitalTwin:
    """ALLEX Real2Sim 디지털 트윈 — ROS2 관절 데이터로 시뮬레이션 로봇 구동"""

    def __init__(self):
        self._initializer = ALLEXInitializer()
        self._asset_manager = ALLEXAssetManager()
        self._joint_controller = ALLEXJointController()
        self._simulation_loop = ALLEXSimulationLoop()
        self._articulation = None
        self._ros2_manager = None
        self._trajectory_player = None

    def setup(self): 
        """시나리오 초기 설정 — 카메라 뷰 + coupled joint config 로드"""
        self._initializer.setup_camera_view()

        extension_root = os.path.dirname(os.path.abspath(__file__))
        joint_config_path = os.path.join(extension_root, "joint_config.json")
        self._joint_controller.load_coupled_joint_config(joint_config_path)

        print("ALLEX Digital Twin setup complete")

    def load_example_assets(self):
        """로봇 USD 에셋 로딩 및 제너레이터 설정"""
        self._articulation = self._asset_manager.load_robot_asset()

        if self._articulation is None:
            return None

        init_success = self._asset_manager.initialize_articulation()
        if init_success:
            self._setup_joint_control_generator()
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
        print("Articulation 초기화 완료")
        return True

    def reset(self):
        """시스템 리셋"""
        self._initializer.reset(self._articulation)
        self._simulation_loop.reset()

        if self._articulation is not None:
            self._setup_joint_control_generator()

        print("시스템 리셋 완료")

    def update(self, step: float):
        """매 physics step마다 호출"""
        return self._simulation_loop.update(step)

    # ========================================
    # 내부 헬퍼
    # ========================================
    def _setup_joint_control_generator(self):
        """관절 제어 제너레이터 생성 및 시뮬레이션 루프에 설정"""
        self._initializer.initialize_joint_positions(self._articulation)

        def get_target_positions():
            # Trajectory playback takes priority when active.
            if self._trajectory_player is not None and self._trajectory_player.is_active():
                traj_target = self._trajectory_player.get_current_target()
                if traj_target is not None:
                    return traj_target.tolist()

            ros2_positions = self._joint_controller.get_unified_target_positions()
            if ros2_positions and any(pos != 0.0 for pos in ros2_positions):
                return ros2_positions
            return self._initializer.target_joint_positions

        def _traj_active():
            return self._trajectory_player is not None and self._trajectory_player.is_active()

        generator = self._joint_controller.create_joint_control_generator(
            articulation=self._articulation,
            get_target_positions_func=get_target_positions,
            is_external_active_fn=_traj_active,
        )
        self._simulation_loop.set_script_generator(generator)

    # ========================================
    # Public API
    # ========================================
    def set_ros2_manager(self, ros2_manager):
        self._ros2_manager = ros2_manager

    def set_trajectory_player(self, player):
        """Install a TrajectoryPlayer; replaces any existing one."""
        if self._trajectory_player is not None:
            try:
                self._trajectory_player.stop()
            except Exception:
                pass
        self._trajectory_player = player

    def get_trajectory_player(self):
        return self._trajectory_player

    def get_robot_info(self):
        return self._asset_manager.get_joint_info()

    def is_simulation_running(self):
        return self._simulation_loop.is_running()

    def stop_simulation(self):
        self._simulation_loop.stop()
