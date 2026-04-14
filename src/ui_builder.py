"""
ALLEX Digital Twin UI Builder — Isaac Sim 확장 UI 및 이벤트 라우팅
"""

import omni.timeline
from omni.usd import StageEventType

from .config import ROS2Config, JointConfig, UIConfig
from .ros2 import ROS2IntegratedManager
from .ui import WorldControlsBuilder, ROS2ControlsBuilder
from .scenario import ALLEXDigitalTwin


class UIBuilder:
    """Isaac Sim 확장 UI를 구성하고 시나리오/ROS2 이벤트를 처리"""

    def __init__(self):
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()

        # ROS2 상태 라벨 (UI에서 설정됨)
        self._sub_status_label = None
        self._topic_mode_status_label = None

        # Articulation 지연 초기화 추적
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0
        self._max_initialization_attempts = JointConfig.MAX_INITIALIZATION_ATTEMPTS

        self._on_init()

    def _on_init(self):
        self._scenario = ALLEXDigitalTwin()
        self._ros2_manager = ROS2IntegratedManager(scenario_ref=self._scenario)
        self._scenario.set_ros2_manager(self._ros2_manager)

    # ========================================
    # ROS2 관리
    # ========================================
    def _setup_ros2_manager(self):
        """ROS2 Manager 초기화"""
        if self._ros2_manager is None:
            self._ros2_manager = ROS2IntegratedManager(scenario_ref=self._scenario)
            self._scenario.set_ros2_manager(self._ros2_manager)

        success = self._ros2_manager.initialize()
        if success:
            if self._scenario and self._scenario._joint_controller:
                self._scenario._joint_controller.set_ros2_subscriber_status(False)
            self._update_topic_mode_label()
        return success

    def _toggle_subscriber(self):
        """Subscriber 토글"""
        if not self._ros2_manager or not self._ros2_manager.is_initialized():
            return
        success, status = self._ros2_manager.toggle_subscriber()
        if self._sub_status_label:
            self._sub_status_label.text = status
        self._update_topic_mode_label()

    def _toggle_topic_mode(self):
        """Current ↔ Desired 토픽 모드 전환"""
        if not self._ros2_manager or not self._ros2_manager.is_initialized():
            return
        success, status_message = self._ros2_manager.toggle_topic_mode()
        if success:
            self._update_topic_mode_label()

    def _update_topic_mode_label(self):
        """토픽 모드 상태 라벨 업데이트"""
        if not self._topic_mode_status_label:
            return
        if self._ros2_manager and self._ros2_manager.is_initialized():
            current_mode = self._ros2_manager.get_current_topic_mode()
            mode_display = ROS2Config.get_topic_mode_display_name(current_mode)
            self._topic_mode_status_label.text = f"Control Mode: {mode_display}"

    def _cleanup_ros2(self):
        """ROS2 정리"""
        if self._ros2_manager:
            self._ros2_manager.cleanup()
            self._ros2_manager = None

        if self._sub_status_label:
            self._sub_status_label.text = "Subscriber: OFF"
        if self._topic_mode_status_label:
            self._topic_mode_status_label.text = "Control Mode: Current"

    # ========================================
    # 시나리오 관리 (ScenarioEventHandler 인라인)
    # ========================================
    def _setup_scene(self):
        """LOAD 버튼 콜백 — 새 스테이지 생성 + 환경 + 로봇 에셋 로딩"""
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.core.api.world import World

        create_new_stage()

        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode(lighting_mode="Stage Lights")

        # FlatGrid 바닥
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.nucleus import get_assets_root_path
        assets_root = get_assets_root_path()
        add_reference_to_stage(assets_root + "/Isaac/Environments/Grid/default_environment.usd", "/FlatGrid")

        # 조명
        import omni.usd
        from pxr import Sdf, UsdLux
        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.CreateIntensityAttr(1000)

        loaded_objects = self._scenario.load_example_assets()

        world = World.instance()
        if isinstance(loaded_objects, (list, tuple)):
            for obj in loaded_objects:
                world.scene.add(obj)
        else:
            world.scene.add(loaded_objects)

    def _setup_scenario(self):
        """Scenario 설정 — LOAD 완료 후 호출"""
        self._scenario.setup()

        if hasattr(self, "_scenario_state_btn") and self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = True
        if hasattr(self, "_reset_btn") and self._reset_btn:
            self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        """RESET 버튼 콜백"""
        self._cleanup_ros2()
        self._scenario.reset()
        self._setup_ros2_manager()

        if hasattr(self, "_scenario_state_btn") and self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """Physics step 콜백 — 지연 초기화 + 시뮬레이션 업데이트"""
        self._physics_step_count += 1

        if (not self._articulation_initialized
                and self._physics_step_count >= 3
                and self._initialization_attempts < self._max_initialization_attempts):
            self._initialization_attempts += 1
            success = self._scenario.delayed_initialization()
            if success:
                self._articulation_initialized = True
            elif self._initialization_attempts >= self._max_initialization_attempts:
                print("Articulation 초기화 최대 시도 횟수 초과")

        done = self._scenario.update(step)
        if done and hasattr(self, "_scenario_state_btn") and self._scenario_state_btn:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        """RUN 버튼 클릭"""
        self._timeline.play()
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0

    def _on_run_scenario_b_text(self):
        """STOP 버튼 클릭"""
        self._timeline.pause()

    # ========================================
    # Isaac Sim 이벤트 콜백 (extension.py에서 호출)
    # ========================================
    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            if hasattr(self, "_scenario_state_btn") and self._scenario_state_btn:
                self._scenario_state_btn.reset()
                self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            try:
                ui_elem.cleanup()
            except Exception:
                pass
        self.wrapped_ui_elements.clear()

        for attr in ('_load_btn', '_reset_btn', '_scenario_state_btn',
                     '_sub_status_label', '_topic_mode_status_label'):
            if hasattr(self, attr):
                setattr(self, attr, None)

        self._cleanup_ros2()

    def build_ui(self):
        """확장 UI 구성"""
        WorldControlsBuilder(self).build()
        ROS2ControlsBuilder(self).build()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        if hasattr(self, "_scenario_state_btn") and self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        if hasattr(self, "_reset_btn") and self._reset_btn:
            self._reset_btn.enabled = False
