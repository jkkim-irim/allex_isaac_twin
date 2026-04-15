"""
ROS2 Controls — ROS2 bridge initialization, subscriber, topic mode
"""

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style

from .ui_components import UIComponentFactory
from ..config import UIConfig, ROS2Config
from ..config.ui_config import UILayout
from ..ros2 import ROS2IntegratedManager


class ROS2Controls:
    """ROS2 Controls section — Initialize / Subscriber / Topic Mode"""

    def __init__(self, ui_ref):
        self._ui = ui_ref
        self._ros2_manager = None
        self._domain_id_field = None
        self._sub_status_label = None
        self._topic_mode_status_label = None
        self._init_manager()

    def _init_manager(self):
        self._ros2_manager = ROS2IntegratedManager(scenario_ref=self._ui._scenario)
        self._ui._scenario.set_ros2_manager(self._ros2_manager)

    # ========================================
    # UI Build
    # ========================================
    def build(self):
        frame = CollapsableFrame("ROS2 Studio", collapsed=UIConfig.JOINT_CONTROL_COLLAPSED)

        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Domain ID:", width=70)
                    self._domain_id_field = ui.IntField(height=UILayout.BUTTON_HEIGHT)
                    self._domain_id_field.model.set_value(ROS2Config.DOMAIN_ID)

                UIComponentFactory.create_styled_button(
                    "Initialize ROS2",
                    callback=self.initialize,
                    color_scheme='green',
                    height=UILayout.BUTTON_HEIGHT,
                )

                UIComponentFactory.create_separator(UILayout.SEPARATOR_HEIGHT)

                with ui.HStack(height=UILayout.BUTTON_HEIGHT_LARGE):
                    UIComponentFactory.create_styled_button(
                        "Subscriber",
                        callback=self._toggle_subscriber,
                        color_scheme='yellow',
                        height=UILayout.BUTTON_HEIGHT,
                    )
                    UIComponentFactory.create_styled_button(
                        "Current | Desired",
                        callback=self._toggle_topic_mode,
                        color_scheme='blue',
                        height=UILayout.BUTTON_HEIGHT,
                    )

                with ui.HStack(height=UILayout.LABEL_HEIGHT):
                    self._sub_status_label = UIComponentFactory.create_status_label(
                        "Subscriber: OFF", UILayout.LABEL_WIDTH_LARGE,
                    )
                    self._topic_mode_status_label = UIComponentFactory.create_status_label(
                        "Control Mode: Current", UILayout.LABEL_WIDTH_LARGE,
                    )

    # ========================================
    # ROS2 management
    # ========================================
    def initialize(self):
        if self._ros2_manager is None:
            self._init_manager()

        # Apply Domain ID from input field — reinitialize if changed
        if self._domain_id_field:
            new_domain_id = self._domain_id_field.model.get_value_as_int()
            if new_domain_id != ROS2Config.DOMAIN_ID or self._ros2_manager.is_initialized():
                ROS2Config.DOMAIN_ID = new_domain_id
                self._ros2_manager.cleanup()

        success = self._ros2_manager.initialize()
        if success:
            jc = self._ui._scenario._joint_controller
            if jc:
                jc.set_ros2_subscriber_status(False)
            self._update_topic_mode_label()
        return success

    def _toggle_subscriber(self):
        if not self._ros2_manager or not self._ros2_manager.is_initialized():
            return
        success, status = self._ros2_manager.toggle_subscriber()
        if self._sub_status_label:
            self._sub_status_label.text = status
        self._update_topic_mode_label()

    def _toggle_topic_mode(self):
        if not self._ros2_manager or not self._ros2_manager.is_initialized():
            return
        success, _ = self._ros2_manager.toggle_topic_mode()
        if success:
            self._update_topic_mode_label()

    def _update_topic_mode_label(self):
        if not self._topic_mode_status_label:
            return
        if self._ros2_manager and self._ros2_manager.is_initialized():
            current_mode = self._ros2_manager.get_current_topic_mode()
            mode_display = ROS2Config.get_topic_mode_display_name(current_mode)
            self._topic_mode_status_label.text = f"Control Mode: {mode_display}"

    # ========================================
    # Lifecycle
    # ========================================
    def on_scenario_changed(self):
        """Called when scenario is recreated (stage opened)"""
        self.cleanup()
        self._init_manager()

    def cleanup(self):
        if self._ros2_manager:
            self._ros2_manager.cleanup()
            self._ros2_manager = None
        if self._sub_status_label:
            self._sub_status_label.text = "Subscriber: OFF"
        if self._topic_mode_status_label:
            self._topic_mode_status_label.text = "Control Mode: Current"
