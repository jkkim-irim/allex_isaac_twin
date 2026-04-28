"""ALLEX 통합 UI — 4개 패널 + 오케스트레이터 단일 진입점.

이 파일은 기존 `src/allex/ui_utils/` 의 4개 패널 클래스
(`WorldControls`, `ROS2Controls`, `TrajStudioControls`, `VisualizerControls`)
와 `src/ui_builder.py` 의 `UIBuilder` 오케스트레이터를 합친 결과입니다.

UI 헬퍼 (`UIComponentFactory`, `ButtonStyleManager`) 와 데이터 클래스
(`UIColors`, `UILayout`, `UIConfig`) 는 `utils/ui_settings_utils.py` 에 있어서
도메인 모듈 (`utils/contact_force_viz.py`) 이 순환 의존 없이 import 가능합니다.

UI 가 아닌 도메인 서브시스템:
- `utils/showcase_logger.py` (CSV 로깅, 623 LOC)
- `utils/contact_force_viz.py` (충돌력 시각화, 547 LOC)
는 유지보수 격리 차원에서 utils/ 에 분리되어 있습니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType

from .config import JointConfig, ROS2Config, UIConfig
from .ros2 import ROS2IntegratedManager
from .scenario import ALLEXDigitalTwin
from .trajectory_generate import TrajectoryPlayer
from .utils.contact_force_viz import ContactForceVisualizer
from .utils.showcase_logger import ShowcaseDataLogger
from .utils.sim_settings_utils import get_world_settings
from .utils.ui_settings_utils import (
    ButtonStyleManager,
    UIColors,
    UIComponentFactory,
    UILayout,
)
from ..hysteresis import HysteresisUI


# `__file__` 가 src/allex/ui.py 이므로 3 레벨 = ext_root.
_EXT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAJECTORY_DIR = _EXT_ROOT / "trajectory"
_RESET_RAMP_S = 3.0


# ============================================================================
# Panel 1 — World Controls (LOAD / RESET / RUN + scenario lifecycle)
# ============================================================================
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

    # ------------------------------------------------------------------
    # UI Build
    # ------------------------------------------------------------------
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
            ws = get_world_settings()
            self._load_btn.set_world_settings(
                physics_dt=ws["physics_dt"],
                rendering_dt=ws["rendering_dt"],
                sim_params={"gravity": ws["gravity"]},
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

    # ------------------------------------------------------------------
    # Scene / Scenario callbacks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Event handling / Lifecycle
    # ------------------------------------------------------------------
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


# ============================================================================
# Panel 2 — ROS2 Bridge (Initialize / Subscriber / Topic Mode)
# ============================================================================
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

    # ------------------------------------------------------------------
    # UI Build
    # ------------------------------------------------------------------
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
                    color_scheme="green",
                    height=UILayout.BUTTON_HEIGHT,
                )

                UIComponentFactory.create_separator(UILayout.SEPARATOR_HEIGHT)

                with ui.HStack(height=UILayout.BUTTON_HEIGHT_LARGE):
                    UIComponentFactory.create_styled_button(
                        "Subscriber",
                        callback=self._toggle_subscriber,
                        color_scheme="yellow",
                        height=UILayout.BUTTON_HEIGHT,
                    )
                    UIComponentFactory.create_styled_button(
                        "Current | Desired",
                        callback=self._toggle_topic_mode,
                        color_scheme="blue",
                        height=UILayout.BUTTON_HEIGHT,
                    )

                with ui.HStack(height=UILayout.LABEL_HEIGHT):
                    self._sub_status_label = UIComponentFactory.create_status_label(
                        "Subscriber: OFF", UILayout.LABEL_WIDTH_LARGE,
                    )
                    self._topic_mode_status_label = UIComponentFactory.create_status_label(
                        "Control Mode: Current", UILayout.LABEL_WIDTH_LARGE,
                    )

    # ------------------------------------------------------------------
    # ROS2 management
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
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


# ============================================================================
# Panel 3 — Trajectory Studio (CSV group dropdown + Run/Stop/Reset)
# ============================================================================
class TrajStudioControls:
    """Traj Studio collapsable section — Dropdown / Run / Stop / Reset."""

    def __init__(self, ui_ref):
        self._ui = ui_ref
        self._combo_frame: ui.Frame | None = None
        self._combo: ui.ComboBox | None = None
        self._items: list[str] = []
        self._status_label: ui.Label | None = None

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------
    def build(self) -> None:
        frame = CollapsableFrame("Traj Studio", collapsed=True)
        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Group:", width=60)
                    self._combo_frame = ui.Frame()
                    self._rebuild_combo()

                UIComponentFactory.create_separator(UILayout.SEPARATOR_HEIGHT)

                with ui.HStack(height=UILayout.BUTTON_HEIGHT_LARGE):
                    UIComponentFactory.create_styled_button(
                        "Run",
                        callback=self._on_run,
                        color_scheme="green",
                        height=UILayout.BUTTON_HEIGHT,
                    )
                    UIComponentFactory.create_styled_button(
                        "Stop",
                        callback=self._on_stop,
                        color_scheme="yellow",
                        height=UILayout.BUTTON_HEIGHT,
                    )
                    UIComponentFactory.create_styled_button(
                        "Reset",
                        callback=self._on_reset,
                        color_scheme="blue",
                        height=UILayout.BUTTON_HEIGHT,
                    )

                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    self._status_label = UIComponentFactory.create_status_label(
                        "Status: idle", UILayout.LABEL_WIDTH_XLARGE,
                    )

    def cleanup(self) -> None:
        scenario = getattr(self._ui, "_scenario", None)
        if scenario is not None:
            player = scenario.get_trajectory_player() if hasattr(scenario, "get_trajectory_player") else None
            if player is not None:
                try:
                    player.stop()
                except Exception:
                    pass
        self._combo_frame = None
        self._combo = None
        self._status_label = None

    # ------------------------------------------------------------------
    # Dropdown
    # ------------------------------------------------------------------
    def _scan_groups(self) -> list[str]:
        if not _TRAJECTORY_DIR.is_dir():
            return []
        return sorted(
            p.name for p in _TRAJECTORY_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def _rebuild_combo(self) -> None:
        if self._combo_frame is None:
            return
        self._items = self._scan_groups()
        self._combo_frame.clear()
        with self._combo_frame:
            if self._items:
                self._combo = ui.ComboBox(0, *self._items, height=UILayout.BUTTON_HEIGHT)
            else:
                self._combo = None
                ui.Label("(no groups in trajectory/)", height=UILayout.BUTTON_HEIGHT)

    def _get_selected_group(self) -> str | None:
        if not self._items or self._combo is None:
            return None
        try:
            idx = self._combo.model.get_item_value_model().get_value_as_int()
        except Exception:
            return None
        if 0 <= idx < len(self._items):
            return self._items[idx]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.text = text
        print(f"[ALLEX][Traj] {text}")

    def _resolve_articulation(self) -> "tuple[object, object] | tuple[None, None]":
        """Return (scenario, articulation) ready for playback, or (None, None) on failure."""
        scenario = getattr(self._ui, "_scenario", None)
        if scenario is None:
            self._set_status("Status: scenario not ready")
            return None, None
        articulation = scenario._asset_manager.get_articulation()
        if articulation is None:
            self._set_status("Status: load robot (LOAD) + start sim (RUN) first")
            return None, None
        if not getattr(articulation, "dof_names", None):
            self._set_status("Status: articulation not initialized — press RUN first")
            return None, None
        return scenario, articulation

    def _query_physics_hz(self, default: float = 200.0) -> float:
        try:
            from isaacsim.physics.newton import acquire_stage
            st = acquire_stage()
            if st is not None:
                f = getattr(st, "physics_frequency", None)
                if f and f > 0:
                    return float(f)
        except Exception as exc:
            print(f"[ALLEX][Traj] physics_frequency query failed: {exc}; using hz={default}")
        return default

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        group = self._get_selected_group()
        if group is None:
            self._set_status("Status: no group selected")
            return

        scenario, articulation = self._resolve_articulation()
        if scenario is None:
            return

        # Prefer the initializer's neutral pose as the seed; it avoids
        # snapping non-covered joints to 0 rad when live pose can't be read.
        seed_pose = None
        initializer = getattr(scenario, "_initializer", None)
        if initializer is not None:
            seed_pose = getattr(initializer, "target_joint_positions", None)

        # Match dense-trajectory sample rate to physics step rate, otherwise
        # playback runs slower/faster than real-time (one sample consumed per
        # physics step).
        hz = self._query_physics_hz()

        csv_dir = _TRAJECTORY_DIR / group
        player = TrajectoryPlayer(csv_dir, articulation, hz=hz, seed_pose=seed_pose)
        self._set_status(f"Status: loading '{group}' @ {hz:.1f} Hz ...")
        if not player.is_ready():
            self._set_status(f"Status: no usable CSVs in {group}")
            return

        scenario.set_trajectory_player(player)
        player.start()
        self._set_status(
            f"Status: Playing '{group}' ({player.duration_s:.2f}s, "
            f"{len(player.groups_used)} group(s))"
        )

        # Showcase data logger — record per physics step only for groups
        # listed in `config/showcase_logger_config.json::logged_groups`.
        logger = getattr(self._ui, "_showcase_logger", None)
        if logger is not None:
            from .utils.showcase_logger import is_group_logged
            if is_group_logged(group):
                try:
                    logger.start_recording(articulation, player, physics_dt=1.0 / hz)
                except Exception as exc:
                    print(f"[ALLEX][Showcase] start_recording failed: {exc}")
            else:
                print(f"[ALLEX][Showcase] group '{group}' not in logged_groups → CSV skipped")

    def _on_stop(self) -> None:
        scenario = getattr(self._ui, "_scenario", None)
        player = scenario.get_trajectory_player() if scenario is not None else None
        if player is None:
            self._set_status("Status: nothing to stop")
            return
        player.stop()
        logger = getattr(self._ui, "_showcase_logger", None)
        if logger is not None:
            logger.stop_recording()
        self._set_status("Status: Stopped (holding current target)")

    def _on_reset(self) -> None:
        scenario, articulation = self._resolve_articulation()
        if scenario is None:
            return
        # Reset belongs to a different lifecycle than Run — finalize any active
        # showcase recording before swapping the player.
        logger = getattr(self._ui, "_showcase_logger", None)
        if logger is not None:
            logger.stop_recording()
        hz = self._query_physics_hz()
        num_dof = int(getattr(articulation, "num_dof", 0) or 0)
        if num_dof <= 0:
            self._set_status("Status: articulation has no dof")
            return
        zero_pose = np.zeros(num_dof, dtype=np.float32)
        player = TrajectoryPlayer.hold_pose(
            articulation, hz=hz, target_pose=zero_pose, ramp_s=_RESET_RAMP_S,
        )
        scenario.set_trajectory_player(player)
        player.start()
        self._set_status(f"Status: Reset (ramp to zero pose, {_RESET_RAMP_S:.1f}s)")


# ============================================================================
# Panel 4 — Visualizer (Force viz toggle + Contact force overlay)
# ============================================================================
class VisualizerControls:
    """Visualizer section — model visualization options"""

    _FORCE_VIZ_PRIM_PATHS = [
        "/ALLEX/L_Palm_Link/visuals/force_viz",
        "/ALLEX/L_Thumb_Distal_Link/visuals/force_viz",
        "/ALLEX/L_Index_Distal_Link/visuals/force_viz",
        "/ALLEX/L_Middle_Distal_Link/visuals/force_viz",
        "/ALLEX/L_Ring_Distal_Link/visuals/force_viz",
        "/ALLEX/L_Little_Distal_Link/visuals/force_viz",
        "/ALLEX/R_Palm_Link/visuals/force_viz",
        "/ALLEX/R_Thumb_Distal_Link/visuals/force_viz",
        "/ALLEX/R_Index_Distal_Link/visuals/force_viz",
        "/ALLEX/R_Middle_Distal_Link/visuals/force_viz",
        "/ALLEX/R_Ring_Distal_Link/visuals/force_viz",
        "/ALLEX/R_Little_Distal_Link/visuals/force_viz",
    ]

    def __init__(self):
        self._force_viz_enabled = False
        self._force_viz_status_label = None
        self._contact_force_viz = ContactForceVisualizer()

    # ------------------------------------------------------------------
    # UI Build
    # ------------------------------------------------------------------
    def build(self):
        frame = CollapsableFrame("Visualizer", collapsed=UIConfig.VISUALIZER_COLLAPSED)

        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                ui.Label(
                    "Control visualization options for ALLEX model.",
                    height=UILayout.LABEL_HEIGHT,
                )

                UIComponentFactory.create_styled_button(
                    "Force Visualizer",
                    callback=self._toggle_force_viz,
                    color_scheme="yellow",
                    height=UILayout.BUTTON_HEIGHT,
                )

                self._force_viz_status_label = UIComponentFactory.create_status_label(
                    "Force Viz: OFF", UILayout.LABEL_WIDTH_LARGE,
                )

                UIComponentFactory.create_separator(UILayout.SEPARATOR_HEIGHT)
                self._contact_force_viz.build()

    # ------------------------------------------------------------------
    # Force Visualizer
    # ------------------------------------------------------------------
    def _toggle_force_viz(self):
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        self._force_viz_enabled = not self._force_viz_enabled
        imageable_token = UsdGeom.Tokens.inherited if self._force_viz_enabled else UsdGeom.Tokens.invisible

        for prim_path in self._FORCE_VIZ_PRIM_PATHS:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"[Visualizer] Prim not found: {prim_path}")
                continue
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(imageable_token)

        status = "ON" if self._force_viz_enabled else "OFF"
        if self._force_viz_status_label:
            self._force_viz_status_label.text = f"Force Viz: {status}"
        print(f"[Visualizer] Force Viz: {status}")

    # ------------------------------------------------------------------
    # Per-step + lifecycle delegation
    # ------------------------------------------------------------------
    def on_physics_step(self, step_dt: float) -> None:
        self._contact_force_viz.on_physics_step(step_dt)

    def cleanup(self) -> None:
        self._contact_force_viz.cleanup()


# ============================================================================
# Orchestrator — was `src/ui_builder.py::UIBuilder`
# ============================================================================
class AllExUI:
    """Isaac Sim extension UI — 4 패널 wire up + lifecycle 디스패치.

    ALLEX 메인 창의 진입점. 각 패널 인스턴스를 보유하고, Isaac Sim 의 timeline /
    stage / physics-step 이벤트를 패널들로 디스패치한다. 추가로 Hysteresis 별도
    창의 lifecycle 도 함께 관리한다.

    필드명 `_world_controls`, `_ros2_controls`, `_traj_studio`, `_visualizer`,
    `_showcase_logger`, `_scenario`, `_timeline` 은 패널들이 `self._ui._scenario`
    같이 cross-reference 하므로 변경 시 주의.
    """

    def __init__(self):
        self._timeline = omni.timeline.get_timeline_interface()
        self._scenario = None
        self._world_controls = None
        self._ros2_controls = None
        self._traj_studio = None
        self._visualizer = None
        self._showcase_logger = None
        self._hysteresis = None
        self._on_init()

    def _on_init(self):
        self._scenario = ALLEXDigitalTwin()

        if self._world_controls is None:
            self._world_controls = WorldControls(self)
        if self._visualizer is None:
            self._visualizer = VisualizerControls()
        if self._showcase_logger is None:
            self._showcase_logger = ShowcaseDataLogger()

        if self._ros2_controls is None:
            self._ros2_controls = ROS2Controls(self)
        else:
            self._ros2_controls.on_scenario_changed()

        if self._traj_studio is None:
            self._traj_studio = TrajStudioControls(self)

    # ------------------------------------------------------------------
    # Isaac Sim event callbacks
    # ------------------------------------------------------------------
    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            if self._world_controls is not None:
                self._world_controls.on_timeline_stop()
            if self._hysteresis is not None:
                self._hysteresis.on_timeline_stop()

    def on_physics_step(self, step: float):
        if self._visualizer is not None:
            self._visualizer.on_physics_step(step)
        if self._showcase_logger is not None:
            self._showcase_logger.on_physics_step(step)
        if self._hysteresis is not None:
            self._hysteresis.on_physics_step(step)

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._on_init()
            self._world_controls.reset_ui()
        if self._hysteresis is not None:
            self._hysteresis.on_stage_event(event)

    # ------------------------------------------------------------------
    # Window builders
    # ------------------------------------------------------------------
    def build_ui(self):
        """메인 ALLEX 창용 UI 빌드."""
        self._world_controls.build()
        self._ros2_controls.build()
        self._traj_studio.build()
        self._visualizer.build()

    def build_hysteresis_ui(self, window):
        """Hysteresis 전용 ScrollingWindow 에 UI 빌드. 최초 1회만 호출."""
        if self._hysteresis is None:
            self._hysteresis = HysteresisUI()
        self._hysteresis.build_ui(window)

    def cleanup(self):
        self._world_controls.cleanup()
        self._ros2_controls.cleanup()
        self._traj_studio.cleanup()
        if self._visualizer is not None:
            self._visualizer.cleanup()
        if self._showcase_logger is not None:
            self._showcase_logger.cleanup()
        if self._hysteresis is not None:
            self._hysteresis.cleanup()
