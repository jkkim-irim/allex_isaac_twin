"""
World Controls — scene loading, scenario management, simulation control
"""

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton

from .ui_components import UIComponentFactory
from .ui_styles import ButtonStyleManager
from ..config import UIConfig, JointConfig
from ..config.ui_config import UIColors, UILayout


class WorldControls:
    """World Controls section — LOAD / RESET / RUN + scenario lifecycle"""

    def __init__(self, ui_ref):
        self._ui = ui_ref
        self._wrapped_elements = []
        self._load_btn = None
        self._reset_btn = None
        self._scenario_state_btn = None

        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0
        self._max_init_attempts = JointConfig.MAX_INITIALIZATION_ATTEMPTS

    # ========================================
    # UI Build
    # ========================================
    def build(self):
        frame = CollapsableFrame("World Controls", collapsed=UIConfig.WORLD_CONTROLS_COLLAPSED)
        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL):
                self._build_load_button()
                self._build_reset_button()
                self._build_scenario_button()
                ButtonStyleManager.apply_button_styles(
                    self._load_btn, self._reset_btn, self._scenario_state_btn,
                )

    def _build_load_button(self):
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_SUCCESS)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)
            self._load_btn = LoadButton(
                "Scene Loader", "LOAD",
                setup_scene_fn=self._setup_scene,
                setup_post_load_fn=self._setup_scenario,
            )
            self._load_btn.set_world_settings(
                physics_dt=UIConfig.PHYSICS_DT,
                rendering_dt=UIConfig.RENDERING_DT,
                sim_params={"gravity": UIConfig.GRAVITY},
            )
            self._wrapped_elements.append(self._load_btn)
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)

    def _build_reset_button(self):
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_RESET)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)
            self._reset_btn = ResetButton(
                "System Reset", "RESET",
                pre_reset_fn=None,
                post_reset_fn=self._on_reset,
            )
            self._reset_btn.enabled = False
            self._wrapped_elements.append(self._reset_btn)
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)

    def _build_scenario_button(self):
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_STATE)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)
            self._scenario_state_btn = StateButton(
                "Run Scenario", "RUN", "STOP",
                on_a_click_fn=self._on_run,
                on_b_click_fn=self._on_stop,
                physics_callback_fn=self._update_scenario,
            )
            self._scenario_state_btn.enabled = False
            self._wrapped_elements.append(self._scenario_state_btn)

    # ========================================
    # Scene / Scenario callbacks
    # ========================================
    def _setup_scene(self):
        from isaacsim.core.utils.stage import create_new_stage, add_reference_to_stage
        from isaacsim.storage.native.nucleus import get_assets_root_path
        from isaacsim.core.api.world import World
        import omni.usd
        from pxr import Sdf, UsdLux

        create_new_stage()

        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode(lighting_mode="Stage Lights")

        assets_root = get_assets_root_path()
        add_reference_to_stage(
            assets_root + "/Isaac/Environments/Grid/default_environment.usd", "/FlatGrid",
        )

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.CreateIntensityAttr(1000)

        loaded_objects = self._ui._scenario.load_example_assets()

        world = World.instance()
        if isinstance(loaded_objects, (list, tuple)):
            for obj in loaded_objects:
                world.scene.add(obj)
        else:
            world.scene.add(loaded_objects)

    def _setup_scenario(self):
        self._ui._scenario.setup()
        if self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = True
        if self._reset_btn:
            self._reset_btn.enabled = True

    def _on_reset(self):
        self._ui._ros2_controls.cleanup()
        self._ui._scenario.reset()
        self._ui._ros2_controls.initialize()
        if self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float, context=None):
        self._physics_step_count += 1

        if (not self._articulation_initialized
                and self._physics_step_count >= 3
                and self._initialization_attempts < self._max_init_attempts):
            self._initialization_attempts += 1
            success = self._ui._scenario.delayed_initialization()
            if success:
                self._articulation_initialized = True
            elif self._initialization_attempts >= self._max_init_attempts:
                print("Articulation initialization max attempts exceeded")

        done = self._ui._scenario.update(step)
        if done and self._scenario_state_btn:
            self._scenario_state_btn.enabled = False

    def _on_run(self):
        self._ui._timeline.play()
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0

    def _on_stop(self):
        self._ui._timeline.pause()

    # ========================================
    # Event handling / Lifecycle
    # ========================================
    def on_timeline_stop(self):
        if self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def reset_ui(self):
        if self._scenario_state_btn:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        if self._reset_btn:
            self._reset_btn.enabled = False

    def cleanup(self):
        for elem in self._wrapped_elements:
            try:
                elem.cleanup()
            except Exception:
                pass
        self._wrapped_elements.clear()
        self._load_btn = None
        self._reset_btn = None
        self._scenario_state_btn = None
