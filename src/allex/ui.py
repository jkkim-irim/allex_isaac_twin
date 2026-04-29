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
from .utils.showcase_replay import ShowcaseReplayControls
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

        try:
            assets_root = get_assets_root_path()
            if assets_root:
                add_reference_to_stage(
                    assets_root + "/Isaac/Environments/Grid/default_environment.usd", "/FlatGrid",
                )
            else:
                print("[ALLEX] assets_root empty — skipping FlatGrid")
        except Exception as _e:
            print(f"[ALLEX] FlatGrid load skipped (Nucleus unavailable): {_e}")

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.CreateIntensityAttr(1000)

        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode(lighting_mode="Stage Lights")

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
        self._raw_checkbox: ui.CheckBox | None = None

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

                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    UIComponentFactory.create_styled_button(
                        "Refresh",
                        callback=self._on_refresh,
                        color_scheme="blue",
                        height=UILayout.BUTTON_HEIGHT,
                    )

                # Raw mode: CSV via point 을 Hermite spline 없이 zero-order hold 로 재생.
                # rosbag 에서 뽑은 CSV (이미 1kHz spline 적용된 target) 재생 시 유용.
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Raw mode (no spline):", width=UILayout.LABEL_WIDTH_LARGE)
                    self._raw_checkbox = ui.CheckBox()

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

        raw_mode = False
        if self._raw_checkbox is not None:
            try:
                raw_mode = bool(self._raw_checkbox.model.get_value_as_bool())
            except Exception:
                raw_mode = False

        csv_dir = _TRAJECTORY_DIR / group
        player = TrajectoryPlayer(csv_dir, articulation, hz=hz, seed_pose=seed_pose,
                                  raw_mode=raw_mode)
        mode_tag = "raw" if raw_mode else "spline"
        self._set_status(f"Status: loading '{group}' @ {hz:.1f} Hz [{mode_tag}] ...")
        if not player.is_ready():
            self._set_status(f"Status: no usable CSVs in {group}")
            return

        scenario.set_trajectory_player(player)
        player.start()
        self._set_status(
            f"Status: Playing '{group}' ({player.duration_s:.2f}s, "
            f"{len(player.groups_used)} group(s), {mode_tag})"
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

    def _on_refresh(self) -> None:
        """Refresh trajectory group dropdown — re-scans `_TRAJECTORY_DIR`.
        새 trajectory 폴더 추가 후 extension reload 없이 드롭다운 갱신."""
        self._rebuild_combo()
        self._set_status(f"Status: refreshed ({len(self._items)} group(s))")

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
_ROBOT_ROOT = "/ALLEX"
_VIZ_SKIP = ("torque_ring", "force_viz", "force_vec")
_OPACITY_VALUE = 0.3

_MODE_OFF = "off"
_MODE_REAL = "real"
_MODE_SIM = "sim"
_MODE_BOTH = "both"
_MODE_CHOICES = [_MODE_REAL, _MODE_SIM, _MODE_BOTH]


class VisualizerControls:
    """Visualizer section — ALLEX viz toggles.

    Force vector 시각화는 두 가지 백엔드를 가집니다 (mutually exclusive):
      - debug draw OFF (기본): ForceTorqueVisualizer (USD prim) — viz mode real/sim/both
      - debug draw ON       : ContactForceVisualizer (debug_draw immediate-mode)
    """

    def __init__(self, scenario=None):
        self._scenario = scenario

        # 상태 — visualizer 와 기본값 동기화 (기본 OFF, 사용자 토글 시 표시)
        self._force_viz_enabled: bool = False
        self._viz_mode: str = _MODE_BOTH
        self._torque_viz_enabled: bool = False
        self._torque_viz_mode: str = _MODE_BOTH

        # opacity 토글
        self._opacity_enabled: bool = False
        self._opacity_backup: dict = {}  # path -> {attr_name -> original_value}

        # Torque plot 상태
        self._torque_plot_body_running: bool = False
        self._torque_plot_hand_running: bool = False

        # Contact force debug-draw backend (mutually exclusive with prim-based force viz)
        self._contact_force_viz = ContactForceVisualizer()
        self._debug_draw_model = None  # ui.SimpleBoolModel — debug-draw checkbox

        # UI refs
        self._force_viz_status_label = None
        self._mode_combo = None
        self._torque_mode_combo = None
        self._torque_viz_status_label = None
        self._opacity_status_label = None
        self._torque_plot_body_label = None
        self._torque_plot_hand_label = None
        self._plot_hz_model = None
        # Plot mode is read at Start time. ["rolling", "cumulative"]
        self._plot_mode_combo = None
        self._plot_mode: str = "rolling"
        # Save-on-stop is read at Start time.
        self._save_on_exit_model = None

    # ========================================
    # scenario 주입 지연 허용
    # ========================================
    def attach_scenario(self, scenario):
        self._scenario = scenario

    # ========================================
    # UI Build
    # ========================================
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

                # Mode ComboBox
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Viz Mode:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._mode_combo = ui.ComboBox(
                        _MODE_CHOICES.index(self._viz_mode),
                        *[m.capitalize() for m in _MODE_CHOICES],
                    )
                    try:
                        model = self._mode_combo.model.get_item_value_model()
                        model.add_value_changed_fn(self._on_mode_change)
                    except Exception:
                        pass

                self._force_viz_status_label = UIComponentFactory.create_status_label(
                    "Force Viz: OFF", UILayout.LABEL_WIDTH_LARGE,
                )

                # Debug-draw mode toggle. Switches force-vector backend between
                # ForceTorqueVisualizer (prim, default) and ContactForceVisualizer
                # (debug_draw immediate-mode, drawn from contact_config.json pairs).
                # Switching disables whichever backend was active.
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Debug draw:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._debug_draw_model = ui.SimpleBoolModel(False)
                    ui.CheckBox(self._debug_draw_model)
                    try:
                        self._debug_draw_model.add_value_changed_fn(
                            self._on_debug_draw_change
                        )
                    except Exception:
                        pass

                ui.Separator(height=4)

                UIComponentFactory.create_styled_button(
                    "Torque Rings",
                    callback=self._toggle_torque_rings,
                    color_scheme='blue',
                    height=UILayout.BUTTON_HEIGHT,
                )

                # Torque Mode ComboBox
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Torque Mode:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._torque_mode_combo = ui.ComboBox(
                        _MODE_CHOICES.index(self._torque_viz_mode),
                        *[m.capitalize() for m in _MODE_CHOICES],
                    )
                    try:
                        model = self._torque_mode_combo.model.get_item_value_model()
                        model.add_value_changed_fn(self._on_torque_mode_change)
                    except Exception:
                        pass

                self._torque_viz_status_label = UIComponentFactory.create_status_label(
                    "Torque Viz: OFF", UILayout.LABEL_WIDTH_LARGE,
                )

                ui.Separator(height=4)

                UIComponentFactory.create_styled_button(
                    "Robot Opacity",
                    callback=self._toggle_opacity,
                    color_scheme='green',
                    height=UILayout.BUTTON_HEIGHT,
                )

                self._opacity_status_label = UIComponentFactory.create_status_label(
                    "Opacity: OFF", UILayout.LABEL_WIDTH_LARGE,
                )

                ui.Separator(height=4)

                # ContactForceVisualizer (debug_draw) — its own button + status
                # label. Activated only when the "Debug draw" checkbox is on.
                self._contact_force_viz.build()

                ui.Separator(height=4)

                # ---- Torque plotter (matplotlib Tk 별도 window) ----
                ui.Label("Realtime joint torque plot (separate Tk window).",
                         height=UILayout.LABEL_HEIGHT)

                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Plot Hz:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._plot_hz_model = ui.SimpleIntModel(50)
                    ui.IntField(self._plot_hz_model)
                    self._plot_hz_model.add_end_edit_fn(
                        lambda _m: self._apply_plot_hz()
                    )

                # Plot mode selector — applied at next Start (subprocess respawn).
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Plot Mode:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._plot_mode_combo = ui.ComboBox(0, "Rolling 5s", "Cumulative")
                    try:
                        m = self._plot_mode_combo.model.get_item_value_model()
                        m.add_value_changed_fn(self._on_plot_mode_change)
                    except Exception:
                        pass

                # Save-on-stop checkbox — saves CSV+PNG when subprocess exits.
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Save on Stop:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._save_on_exit_model = ui.SimpleBoolModel(False)
                    ui.CheckBox(self._save_on_exit_model)

                UIComponentFactory.create_styled_button(
                    "Torque Plot (Body)",
                    callback=lambda: self._toggle_torque_plot("body"),
                    color_scheme='yellow',
                    height=UILayout.BUTTON_HEIGHT,
                )
                self._torque_plot_body_label = UIComponentFactory.create_status_label(
                    "Torque Plot (Body): OFF", UILayout.LABEL_WIDTH_LARGE,
                )

                UIComponentFactory.create_styled_button(
                    "Torque Plot (Hand)",
                    callback=lambda: self._toggle_torque_plot("hand"),
                    color_scheme='blue',
                    height=UILayout.BUTTON_HEIGHT,
                )
                self._torque_plot_hand_label = UIComponentFactory.create_status_label(
                    "Torque Plot (Hand): OFF", UILayout.LABEL_WIDTH_LARGE,
                )

    # ========================================
    # Toggle — Force Visualizer (prim-based ForceTorqueVisualizer)
    # ========================================
    def _toggle_force_viz(self):
        if self._is_debug_draw_mode():
            # Debug draw 모드일 때는 ContactForceVisualizer 버튼이 진짜 토글.
            # Force Visualizer 버튼은 비활성으로 안내.
            if self._force_viz_status_label:
                self._force_viz_status_label.text = (
                    "Force Viz: (debug-draw mode — use Contact Forces button)"
                )
            return

        visualizer = self._get_visualizer()
        prev_enabled = self._force_viz_enabled
        self._force_viz_enabled = not self._force_viz_enabled

        if visualizer is None:
            self._force_viz_enabled = prev_enabled
            if self._force_viz_status_label:
                self._force_viz_status_label.text = "Force Viz: NOT READY"
            return

        mode = self._viz_mode if self._force_viz_enabled else _MODE_OFF
        try:
            visualizer.set_mode(mode)
        except Exception as e:
            self._force_viz_enabled = prev_enabled
            print(f"[Visualizer] set_mode failed: {e}")
            if self._force_viz_status_label:
                self._force_viz_status_label.text = "Force Viz: ERROR"
            return

        if self._force_viz_enabled:
            status_text = f"Force Viz: ON ({self._viz_mode})"
        else:
            status_text = "Force Viz: OFF"
        if self._force_viz_status_label:
            self._force_viz_status_label.text = status_text
        print(f"[Visualizer] {status_text}")

    # ========================================
    # Mode change
    # ========================================
    def _on_mode_change(self, model):
        try:
            idx = model.as_int
        except Exception:
            return
        if idx < 0 or idx >= len(_MODE_CHOICES):
            return
        self._viz_mode = _MODE_CHOICES[idx]
        visualizer = self._get_visualizer()
        if visualizer is not None and self._force_viz_enabled and not self._is_debug_draw_mode():
            try:
                visualizer.set_mode(self._viz_mode)
            except Exception as e:
                print(f"[Visualizer] set_mode failed: {e}")
        if self._force_viz_status_label and self._force_viz_enabled:
            self._force_viz_status_label.text = f"Force Viz: ON ({self._viz_mode})"
        print(f"[Visualizer] Mode -> {self._viz_mode}")

    # ========================================
    # Debug-draw backend toggle (mutually exclusive force-vector backend)
    # ========================================
    def _is_debug_draw_mode(self) -> bool:
        if self._debug_draw_model is None:
            return False
        try:
            return bool(self._debug_draw_model.as_bool)
        except Exception:
            return False

    def _on_debug_draw_change(self, model):
        try:
            checked = bool(model.as_bool)
        except Exception:
            return
        if checked:
            # Tear down ForceTorqueVisualizer prim mode if active.
            if self._force_viz_enabled:
                visualizer = self._get_visualizer()
                if visualizer is not None:
                    try:
                        visualizer.set_mode(_MODE_OFF)
                    except Exception as exc:
                        print(f"[Visualizer] set_mode(off) on debug-draw enable failed: {exc}")
                self._force_viz_enabled = False
                if self._force_viz_status_label:
                    self._force_viz_status_label.text = (
                        "Force Viz: (debug-draw mode — use Contact Forces button)"
                    )
            else:
                if self._force_viz_status_label:
                    self._force_viz_status_label.text = (
                        "Force Viz: (debug-draw mode — use Contact Forces button)"
                    )
            print("[Visualizer] backend -> debug-draw (ContactForceVisualizer)")
        else:
            # Tear down ContactForceVisualizer if active.
            if self._contact_force_viz.is_enabled:
                try:
                    self._contact_force_viz.set_enabled(False)
                except Exception as exc:
                    print(f"[Visualizer] contact_force_viz set_enabled(False) failed: {exc}")
            if self._force_viz_status_label:
                self._force_viz_status_label.text = "Force Viz: OFF"
            print("[Visualizer] backend -> prim (ForceTorqueVisualizer)")

    # ========================================
    # Torque rings
    # ========================================
    def _toggle_torque_rings(self):
        visualizer = self._get_visualizer()
        prev_enabled = self._torque_viz_enabled
        self._torque_viz_enabled = not self._torque_viz_enabled

        if visualizer is None:
            self._torque_viz_enabled = prev_enabled
            if self._torque_viz_status_label:
                self._torque_viz_status_label.text = "Torque Viz: NOT READY"
            return

        mode = self._torque_viz_mode if self._torque_viz_enabled else _MODE_OFF
        try:
            visualizer.set_torque_mode(mode)
        except Exception as e:
            self._torque_viz_enabled = prev_enabled
            print(f"[Visualizer] set_torque_mode failed: {e}")
            if self._torque_viz_status_label:
                self._torque_viz_status_label.text = "Torque Viz: ERROR"
            return

        if self._torque_viz_enabled:
            status_text = f"Torque Viz: ON ({self._torque_viz_mode})"
        else:
            status_text = "Torque Viz: OFF"
        if self._torque_viz_status_label:
            self._torque_viz_status_label.text = status_text
        print(f"[Visualizer] {status_text}")

    def _on_torque_mode_change(self, model):
        try:
            idx = model.as_int
        except Exception:
            return
        if idx < 0 or idx >= len(_MODE_CHOICES):
            return
        self._torque_viz_mode = _MODE_CHOICES[idx]
        visualizer = self._get_visualizer()
        if visualizer is not None and self._torque_viz_enabled:
            try:
                visualizer.set_torque_mode(self._torque_viz_mode)
            except Exception as e:
                print(f"[Visualizer] set_torque_mode failed: {e}")
        if self._torque_viz_status_label and self._torque_viz_enabled:
            self._torque_viz_status_label.text = f"Torque Viz: ON ({self._torque_viz_mode})"
        print(f"[Visualizer] Torque Mode -> {self._torque_viz_mode}")

    # ========================================
    # Robot Opacity
    # ========================================
    def _toggle_opacity(self):
        import omni.usd
        from pxr import UsdShade, Sdf

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            if self._opacity_status_label:
                self._opacity_status_label.text = "Opacity: NO STAGE"
            return

        self._opacity_enabled = not self._opacity_enabled

        if self._opacity_enabled:
            self._opacity_backup.clear()
            self._apply_opacity(stage, _OPACITY_VALUE)
            text = f"Opacity: ON ({int(_OPACITY_VALUE * 100)}%)"
        else:
            self._restore_opacity(stage)
            text = "Opacity: OFF"

        if self._opacity_status_label:
            self._opacity_status_label.text = text
        print(f"[Visualizer] {text}")

    def _apply_opacity(self, stage, opacity: float):
        from pxr import UsdShade, Sdf

        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(_ROBOT_ROOT):
                continue
            if "Looks" not in path:
                continue
            if any(s in path for s in _VIZ_SKIP):
                continue
            if prim.GetTypeName() != "Shader":
                continue

            # OmniPBR MDL 파라미터: enable_opacity(bool) + opacity_constant(float)
            self._set_attr(prim, path, "inputs:enable_opacity", Sdf.ValueTypeNames.Bool, True)
            self._set_attr(prim, path, "inputs:opacity_constant", Sdf.ValueTypeNames.Float, opacity)

    def _restore_opacity(self, stage):
        # OmniPBR MDL default: enable_opacity=False, opacity_constant=1.0
        _MDL_DEFAULTS = {
            "enable_opacity": False,
            "opacity_constant": 1.0,
        }
        from pxr import UsdShade
        for path_str, attrs in self._opacity_backup.items():
            prim = stage.GetPrimAtPath(path_str)
            if not prim.IsValid():
                continue
            shader = UsdShade.Shader(prim)
            for attr_name, original in attrs.items():
                input_name = attr_name.removeprefix("inputs:")
                try:
                    value = original if original is not None else _MDL_DEFAULTS.get(input_name)
                    if value is None:
                        continue
                    inp = shader.GetInput(input_name)
                    if inp:
                        inp.Set(value)
                except Exception as exc:
                    print(f"[Opacity restore] {path_str}/{attr_name}: {exc}")
        self._opacity_backup.clear()

    def _set_attr(self, prim, path_str, attr_name, vtype, value):
        try:
            from pxr import UsdShade
            input_name = attr_name.removeprefix("inputs:")
            shader = UsdShade.Shader(prim)
            backup = self._opacity_backup.setdefault(path_str, {})
            if attr_name not in backup:
                try:
                    backup[attr_name] = shader.GetInput(input_name).Get()
                except Exception:
                    backup[attr_name] = None
            inp = shader.CreateInput(input_name, vtype)
            inp.Set(value)
        except Exception as e:
            print(f"[Opacity] {path_str} / {attr_name}: {e}")

    # ========================================
    # Torque Plotter (matplotlib Tk)
    # ========================================
    def _get_torque_plotter(self, subset: str):
        if self._scenario is None:
            return None
        getter = getattr(self._scenario, "get_torque_plotter", None)
        if getter is None:
            return None
        try:
            return getter(subset)
        except Exception as exc:
            print(f"[TorquePlot] get_torque_plotter({subset}) failed: {exc}")
            return None

    def _torque_plot_label(self, subset: str):
        return (self._torque_plot_body_label if subset == "body"
                else self._torque_plot_hand_label)

    def _apply_plot_hz(self):
        if self._plot_hz_model is None:
            return
        try:
            hz = int(self._plot_hz_model.as_int)
        except Exception:
            return
        hz = max(1, min(hz, 500))
        for subset in ("body", "hand"):
            plotter = self._get_torque_plotter(subset)
            if plotter is None:
                continue
            try:
                plotter.set_plot_hz(float(hz))
            except Exception as exc:
                print(f"[TorquePlot] set_plot_hz({subset}) failed: {exc}")
        print(f"[TorquePlot] plot_hz -> {hz}")

    def _on_plot_mode_change(self, model):
        try:
            idx = int(model.as_int)
        except Exception:
            return
        self._plot_mode = "cumulative" if idx == 1 else "rolling"
        print(f"[TorquePlot] plot_mode -> {self._plot_mode} "
              "(applies on next Start)")

    def _toggle_torque_plot(self, subset: str):
        plotter = self._get_torque_plotter(subset)
        label = self._torque_plot_label(subset)
        if plotter is None:
            if label:
                label.text = f"Torque Plot ({subset.capitalize()}): NOT READY (press RUN first)"
            return

        try:
            running = plotter.is_running()
        except Exception:
            running = bool(getattr(self, f"_torque_plot_{subset}_running", False))

        if running:
            try:
                plotter.stop()
            except Exception as exc:
                print(f"[TorquePlot] stop({subset}) failed: {exc}")
            setattr(self, f"_torque_plot_{subset}_running", False)
            if label:
                label.text = f"Torque Plot ({subset.capitalize()}): OFF"
            return

        try:
            plotter.set_plot_mode(self._plot_mode)
        except Exception as exc:
            print(f"[TorquePlot] set_plot_mode({subset}) failed: {exc}")

        try:
            save_on_exit = bool(self._save_on_exit_model.as_bool) \
                if self._save_on_exit_model is not None else False
            plotter.set_save_on_exit(save_on_exit)
            if save_on_exit:
                print(f"[TorquePlot] save_on_exit=ON for {subset} — CSV+PNG will be written on Stop")
        except Exception as exc:
            print(f"[TorquePlot] set_save_on_exit({subset}) failed: {exc}")

        try:
            ok = plotter.start()
        except Exception as exc:
            print(f"[TorquePlot] start({subset}) failed: {exc}")
            if label:
                label.text = f"Torque Plot ({subset.capitalize()}): ERROR"
            return
        if ok:
            setattr(self, f"_torque_plot_{subset}_running", True)
            if label:
                label.text = f"Torque Plot ({subset.capitalize()}): ON"
        else:
            has_joints = bool(getattr(plotter, "_plot_indices", None))
            if label:
                if has_joints:
                    label.text = f"Torque Plot ({subset.capitalize()}): Install python3-tk"
                else:
                    label.text = f"Torque Plot ({subset.capitalize()}): NO JOINTS"

    # ========================================
    # Helpers
    # ========================================
    def _get_visualizer(self):
        if self._scenario is None:
            return None
        getter = getattr(self._scenario, "get_visualizer", None)
        if getter is not None:
            return getter()
        return getattr(self._scenario, "_visualizer", None)

    # ------------------------------------------------------------------
    # Per-step + lifecycle delegation
    # ------------------------------------------------------------------
    def on_physics_step(self, step_dt: float) -> None:
        # ContactForceVisualizer 는 자체적으로 _enabled 가드를 가짐 — 안전.
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
        self._showcase_replay = None
        self._visualizer = None
        self._showcase_logger = None
        self._hysteresis = None
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
        if self._showcase_logger is None:
            self._showcase_logger = ShowcaseDataLogger()

        if self._ros2_controls is None:
            self._ros2_controls = ROS2Controls(self)
        else:
            self._ros2_controls.on_scenario_changed()

        if self._traj_studio is None:
            self._traj_studio = TrajStudioControls(self)

        if self._showcase_replay is None:
            self._showcase_replay = ShowcaseReplayControls(self)

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
        self._showcase_replay.build()
        self._visualizer.build()

    def build_hysteresis_ui(self, window):
        """Hysteresis 전용 ScrollingWindow 에 UI 빌드. 최초 1회만 호출."""
        if self._hysteresis is None:
            self._hysteresis = HysteresisUI()
        self._hysteresis.build_ui(window)

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
        if self._showcase_replay is not None:
            self._showcase_replay.cleanup()
        if self._visualizer is not None:
            self._visualizer.cleanup()
        if self._showcase_logger is not None:
            self._showcase_logger.cleanup()
        if self._hysteresis is not None:
            self._hysteresis.cleanup()
