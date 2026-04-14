"""
UI 섹션별 빌더
"""

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import StateButton

from .ui_components import UIComponentFactory
from .ui_styles import ButtonStyleManager
from ..config import UIConfig
from ..config.ui_config import UIColors, UILayout


class WorldControlsBuilder:
    """World Controls 섹션 (LOAD / RESET / RUN)"""

    def __init__(self, ui_builder_ref):
        self.ui_builder = ui_builder_ref

    def build(self):
        world_controls_frame = CollapsableFrame("World Controls", collapsed=UIConfig.WORLD_CONTROLS_COLLAPSED)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL):
                self._build_load_button()
                self._build_reset_button()
                self._build_scenario_button()
                self._apply_button_styles()

    def _build_load_button(self):
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_SUCCESS)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)

            self.ui_builder._load_btn = LoadButton(
                "Scene Loader", "LOAD",
                setup_scene_fn=self.ui_builder._setup_scene,
                setup_post_load_fn=self.ui_builder._setup_scenario,
            )
            self.ui_builder._load_btn.set_world_settings(
                physics_dt=UIConfig.PHYSICS_DT,
                rendering_dt=UIConfig.RENDERING_DT,
                sim_params={"gravity": UIConfig.GRAVITY},
            )
            self.ui_builder.wrapped_ui_elements.append(self.ui_builder._load_btn)
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)

    def _build_reset_button(self):
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_RESET)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)

            self.ui_builder._reset_btn = ResetButton(
                "System Reset", "RESET",
                pre_reset_fn=None,
                post_reset_fn=self.ui_builder._on_post_reset_btn,
            )
            self.ui_builder._reset_btn.enabled = False
            self.ui_builder.wrapped_ui_elements.append(self.ui_builder._reset_btn)
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)

    def _build_scenario_button(self):
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_STATE)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)

            self.ui_builder._scenario_state_btn = StateButton(
                "Run Scenario", "RUN", "STOP",
                on_a_click_fn=self.ui_builder._on_run_scenario_a_text,
                on_b_click_fn=self.ui_builder._on_run_scenario_b_text,
                physics_callback_fn=self.ui_builder._update_scenario,
            )
            self.ui_builder._scenario_state_btn.enabled = False
            self.ui_builder.wrapped_ui_elements.append(self.ui_builder._scenario_state_btn)

    def _apply_button_styles(self):
        ButtonStyleManager.apply_button_styles(
            self.ui_builder._load_btn,
            self.ui_builder._reset_btn,
            self.ui_builder._scenario_state_btn,
        )


class ROS2ControlsBuilder:
    """ROS2 Controls 섹션 (Initialize / Subscriber / Current|Desired)"""

    def __init__(self, ui_builder_ref):
        self.ui_builder = ui_builder_ref

    def build(self):
        ros2_frame = CollapsableFrame("ROS2 Studio", collapsed=UIConfig.JOINT_CONTROL_COLLAPSED)

        with ros2_frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                # Initialize ROS2
                UIComponentFactory.create_styled_button(
                    "Initialize ROS2",
                    callback=self.ui_builder._setup_ros2_manager,
                    color_scheme='green',
                    height=UILayout.BUTTON_HEIGHT,
                )

                UIComponentFactory.create_separator(UILayout.SEPARATOR_HEIGHT)

                # Subscriber + Current|Desired 버튼
                with ui.HStack(height=UILayout.BUTTON_HEIGHT_LARGE):
                    UIComponentFactory.create_styled_button(
                        "Subscriber",
                        callback=self.ui_builder._toggle_subscriber,
                        color_scheme='yellow',
                        height=UILayout.BUTTON_HEIGHT,
                    )
                    UIComponentFactory.create_styled_button(
                        "Current | Desired",
                        callback=self.ui_builder._toggle_topic_mode,
                        color_scheme='blue',
                        height=UILayout.BUTTON_HEIGHT,
                    )

                # 상태 라벨
                with ui.HStack(height=UILayout.LABEL_HEIGHT):
                    self.ui_builder._sub_status_label = UIComponentFactory.create_status_label(
                        "Subscriber: OFF", UILayout.LABEL_WIDTH_LARGE
                    )
                    self.ui_builder._topic_mode_status_label = UIComponentFactory.create_status_label(
                        "Control Mode: Current", UILayout.LABEL_WIDTH_LARGE
                    )
