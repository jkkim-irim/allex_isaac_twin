"""
ALLEX Digital Twin UI Builder — Isaac Sim extension UI entry point
"""

import omni.timeline
from omni.usd import StageEventType

from .scenario import ALLEXDigitalTwin
from .ui_utils import WorldControls, ROS2Controls, TrajStudioControls, VisualizerControls


class UIBuilder:
    """Isaac Sim extension UI — delegates to feature controllers"""

    def __init__(self):
        self._timeline = omni.timeline.get_timeline_interface()
        self._scenario = None
        self._world_controls = None
        self._ros2_controls = None
        self._traj_studio = None
        self._visualizer = None
        self._on_init()

    def _on_init(self):
        self._scenario = ALLEXDigitalTwin()

        if self._world_controls is None:
            self._world_controls = WorldControls(self)
        if self._visualizer is None:
            self._visualizer = VisualizerControls(scenario=self._scenario)
        else:
            # stage reload 시 scenario 재주입
            self._visualizer.attach_scenario(self._scenario)

        if self._ros2_controls is None:
            self._ros2_controls = ROS2Controls(self)
        else:
            self._ros2_controls.on_scenario_changed()

        if self._traj_studio is None:
            self._traj_studio = TrajStudioControls(self)

    # ========================================
    # Isaac Sim event callbacks
    # ========================================
    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._world_controls.on_timeline_stop()

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._on_init()
            self._world_controls.reset_ui()

    def build_ui(self):
        self._world_controls.build()
        self._ros2_controls.build()
        self._traj_studio.build()
        self._visualizer.build()

    def cleanup(self):
        # Tear down plotter subprocesses before other controllers to avoid
        # leaking Tk windows across extension reload / stage reopen.
        if self._scenario is not None:
            try:
                self._scenario.shutdown()
            except Exception as exc:
                print(f"[ALLEX] scenario shutdown warn: {exc}")
        self._world_controls.cleanup()
        self._ros2_controls.cleanup()
        self._traj_studio.cleanup()
