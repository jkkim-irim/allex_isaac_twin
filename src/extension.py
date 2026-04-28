# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import gc
from pathlib import Path

import omni
import omni.kit.commands
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from omni.usd import StageEventType

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, HYSTERESIS_TITLE
from .allex.ui import AllExUI

"""
This file serves as a basic template for the standard boilerplate operations
that make a UI-based extension appear on the toolbar.

This implementation is meant to cover most use-cases without modification.
Various callbacks are hooked up to the AllExUI class in .allex.ui (4 UI
panels + orchestrator + Hysteresis window). Most users will be able to make
their desired UI extension by interacting solely with AllExUI.

This class sets up standard useful callback functions in UIBuilder:
    on_menu_callback: Called when extension is opened
    on_timeline_event: Called when timeline is stopped, paused, or played
    on_physics_step: Called on every physics step
    on_stage_event: Called when stage is opened or closed
    cleanup: Called when resources such as physics subscriptions should be cleaned up
    build_ui: User function that creates the UI they want.
"""


def _reload_submodules():
    """핫 리로드 시 캐싱된 서브모듈 강제 제거"""
    import sys
    to_remove = [key for key in sys.modules if key.startswith("src.") or key.startswith("src")]
    for key in to_remove:
        del sys.modules[key]


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        _reload_submodules()

        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()

        # Build main ALLEX window
        self._window = ui.Window(
            title=EXTENSION_TITLE, 
            width=400, 
            height=500, 
            visible=False, 
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE | ui.WINDOW_FLAGS_NO_RESIZE,
            padding_x=10,
            padding_y=10,
            #dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        self._window.set_visibility_changed_fn(self._on_window)

        # Build Hysteresis standalone window (Robotics_Study_python pattern)
        self._hysteresis_window = ScrollingWindow(
            title=HYSTERESIS_TITLE,
            width=420,
            height=520,
            visible=False,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._hysteresis_ui_built = False

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            ext_id,
            f"CreateUIExtension:{EXTENSION_TITLE}",
            self._menu_callback,
            description=f"Add {EXTENSION_TITLE} Extension to UI toolbar",
        )
        action_registry.register_action(
            ext_id,
            f"CreateUIExtension:{HYSTERESIS_TITLE}",
            self._hysteresis_callback,
            description=f"Add {HYSTERESIS_TITLE} Extension to UI toolbar",
        )
        self._menu_items = [
            MenuItemDescription(name=EXTENSION_TITLE, onclick_action=(ext_id, f"CreateUIExtension:{EXTENSION_TITLE}")),
            MenuItemDescription(name=HYSTERESIS_TITLE, onclick_action=(ext_id, f"CreateUIExtension:{HYSTERESIS_TITLE}")),
        ]

        add_menu_items(self._menu_items, EXTENSION_TITLE)

        # Filled in with User Functions
        self.ui_builder = AllExUI()

        # Events
        self._usd_context = omni.usd.get_context()
        self._physics_step_sub = None
        self._stage_event_sub = None
        self._timeline = omni.timeline.get_timeline_interface()

        # Newton: inject MJCF equality constraints before finalize.
        from .allex.core import newton_bridge
        joint_config = Path(__file__).parent / "allex" / "config" / "joint_config.json"
        newton_bridge.install(joint_config)

    def on_shutdown(self):
        self._models = {}
        self._deregister_physics_step()
        from .allex.core import newton_bridge
        newton_bridge.uninstall()
        remove_menu_items(self._menu_items, EXTENSION_TITLE)

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_action(self.ext_id, f"CreateUIExtension:{EXTENSION_TITLE}")
        action_registry.deregister_action(self.ext_id, f"CreateUIExtension:{HYSTERESIS_TITLE}")

        if self._window:
            self._window = None
        if self._hysteresis_window:
            self._hysteresis_window.visible = False
            self._hysteresis_window = None
        self._hysteresis_ui_built = False
        self.ui_builder.cleanup()
        gc.collect()

    def _on_window(self, visible):
        if self._window.visible:
            self._setup_event_subscriptions()
            self._build_ui()
        else:
            # Keep subscriptions alive if the Hysteresis window is still open.
            if not (self._hysteresis_window and self._hysteresis_window.visible):
                self._usd_context = None
                self._stage_event_sub = None
                self._timeline_event_sub = None
                self.ui_builder.cleanup()

    def _setup_event_subscriptions(self):
        """Timeline / stage 이벤트 구독을 한 번만 설치."""
        self._usd_context = omni.usd.get_context()
        if self._stage_event_sub is None:
            events = self._usd_context.get_stage_event_stream()
            self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)
        if getattr(self, "_timeline_event_sub", None) is None:
            stream = self._timeline.get_timeline_event_stream()
            self._timeline_event_sub = stream.create_subscription_to_pop(self._on_timeline_event)

    def _build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                self._build_extension_ui()

        # async def dock_window():
        #     await omni.kit.app.get_app().next_update_async()

        #     def dock(space, name, location, pos=0.5):
        #         window = omni.ui.Workspace.get_window(name)
        #         if window and space:
        #             window.dock_in(space, location, pos)
        #         return window

        #     tgt = ui.Workspace.get_window("Viewport")
        #     dock(tgt, EXTENSION_TITLE, omni.ui.DockPosition.LEFT, 0.33)
        #     await omni.kit.app.get_app().next_update_async()

        # self._task = asyncio.ensure_future(dock_window())

    #################################################################
    # Functions below this point call user functions
    #################################################################

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        self.ui_builder.on_menu_callback()

    def _hysteresis_callback(self):
        """Hysteresis 메뉴 토글 — 최초 클릭 시에만 UI 빌드, 이후엔 visible 토글."""
        if not self._hysteresis_ui_built:
            self.ui_builder.build_hysteresis_ui(self._hysteresis_window)
            self._hysteresis_ui_built = True
        self._hysteresis_window.visible = not self._hysteresis_window.visible
        if self._hysteresis_window.visible:
            self._setup_event_subscriptions()

    def _on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if self._physics_step_sub is None:
                self._physics_step_sub = SimulationManager.register_callback(
                    self._on_physics_step,
                    event=SimulationEvent.PHYSICS_POST_STEP,
                )
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._deregister_physics_step()

        self.ui_builder.on_timeline_event(event)

    def _deregister_physics_step(self):
        if self._physics_step_sub is not None:
            try:
                SimulationManager.deregister_callback(self._physics_step_sub)
            except Exception:
                pass
            self._physics_step_sub = None

    def _on_physics_step(self, step_dt, context):
        self.ui_builder.on_physics_step(step_dt)

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED) or event.type == int(StageEventType.CLOSED):
            # stage was opened or closed, cleanup
            self._deregister_physics_step()
            self.ui_builder.cleanup()

            if self._window and self._window.visible:
                self._build_ui()
            if (self._hysteresis_window and self._hysteresis_window.visible
                    and self._hysteresis_ui_built):
                self.ui_builder.build_hysteresis_ui(self._hysteresis_window)

        self.ui_builder.on_stage_event(event)

    def _build_extension_ui(self):
        # Call user function for building UI
        self.ui_builder.build_ui()
