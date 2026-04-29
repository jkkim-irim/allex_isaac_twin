"""Hysteresis UI — 핸드 전용 독립 창 UI 빌더.

주어진 `ScrollingWindow` 에 LOAD/RESET/RUN + AML Replay + (Corrector
placeholder) 섹션을 구축한다. Measured Logger 는
    Play → ramp → post-ramp hold → playback_started → record →
    player finished → auto-flush
순서로 완전 자동 동작하므로 UI 섹션 없이 console print 만 노출한다.
ALLEXDigitalTwin 과는 독립적인 `HysteresisScenario` 인스턴스 하나를 소유해
stage 교체에 영향받지 않는 독립된 lifecycle을 가진다.
"""

from __future__ import annotations

from pathlib import Path

import omni.ui as ui
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.storage.native.nucleus import get_assets_root_path
from pxr import Sdf, UsdLux

from ..allex.utils.ui_settings_utils import (
    ButtonStyleManager,
    UIColors,
    UIComponentFactory,
    UILayout,
)
from ..allex.trajectory_generate.joint_name_map import ALLEX_CSV_JOINT_NAMES
from .aml_command_player import AMLCommandPlayer
from .config import PhysicsConfig, UIConfig
from .measured_logger import MeasuredLogger
from .scenario import HysteresisScenario


_EXT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAJECTORY_DIR = _EXT_ROOT / "trajectory"
_DEFAULT_GROUP = "aml_hysteresis_group"
_DEFAULT_CSV_STEM = "Hand_R_index_wir"
_DEFAULT_COMMAND_CSV = _TRAJECTORY_DIR / _DEFAULT_GROUP / "real_data" / "command_joint_deg_0.csv"
_DEFAULT_LOG_PATH = _TRAJECTORY_DIR / _DEFAULT_GROUP / "sim_measured.csv"

_R_INDEX_JOINTS = ALLEX_CSV_JOINT_NAMES[_DEFAULT_CSV_STEM]


class HysteresisUI:
    """Hysteresis 전용 창 UI 컨트롤러."""

    def __init__(self):
        self._scenario = HysteresisScenario()
        self._wrapped_elements: list = []
        self._timeline = None

        # World
        self._load_btn: LoadButton | None = None
        self._reset_btn: ResetButton | None = None
        self._state_btn: StateButton | None = None
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0

        # Replay
        self._replay_status_label: ui.Label | None = None

        # Logger — armed/recording is driven by the replay player:
        #   Play → arm (waiting for ramp-in to finish) → recording → flush.
        # No UI section; status is only surfaced via console prints.
        self._logger: MeasuredLogger | None = None
        self._logger_active = False        # True while on_step() should append
        self._logger_armed = False         # True between Play and ramp-done

        import omni.timeline
        self._timeline = omni.timeline.get_timeline_interface()

    # ==================================================================
    # Build entry — Robotics_Study_python Assignment1UI.build_ui(window) 패턴
    # ==================================================================
    def build_ui(self, window) -> None:
        with window.frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                self._build_world_section()
                self._build_replay_section()
                self._build_corrector_section()

    # ==================================================================
    # World section (LOAD / RESET / RUN)
    # ==================================================================
    def _build_world_section(self) -> None:
        frame = CollapsableFrame("World", collapsed=False)
        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL):
                self._build_load_button()
                self._build_reset_button()
                self._build_state_button()
                ButtonStyleManager.apply_button_styles(
                    self._load_btn, self._reset_btn, self._state_btn,
                )

    def _build_load_button(self) -> None:
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_SUCCESS)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)
            self._load_btn = LoadButton(
                "Hand Scene Loader", "LOAD",
                setup_scene_fn=self._setup_scene,
                setup_post_load_fn=self._setup_scenario,
            )
            self._load_btn.set_world_settings(
                physics_dt=PhysicsConfig.PHYSICS_DT,
                rendering_dt=PhysicsConfig.RENDERING_DT,
                sim_params={"gravity": PhysicsConfig.GRAVITY},
            )
            self._wrapped_elements.append(self._load_btn)
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)

    def _build_reset_button(self) -> None:
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_RESET)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)
            self._reset_btn = ResetButton(
                "System Reset", "RESET",
                pre_reset_fn=self._pre_reset,
                post_reset_fn=self._on_reset,
            )
            self._reset_btn.enabled = False
            self._wrapped_elements.append(self._reset_btn)
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)

    def _build_state_button(self) -> None:
        with ui.HStack():
            UIComponentFactory.create_spacer(UILayout.SPACING_SMALL)
            UIComponentFactory.create_colored_sidebar(UIColors.SIDEBAR_STATE)
            UIComponentFactory.create_spacer(UILayout.SPACING_MEDIUM)
            self._state_btn = StateButton(
                "Run Scenario", "RUN", "STOP",
                on_a_click_fn=self._on_run,
                on_b_click_fn=self._on_stop,
                physics_callback_fn=self._update_scenario,
            )
            self._state_btn.enabled = False
            self._wrapped_elements.append(self._state_btn)

    # ------------------------------------------------------------------
    # World callbacks
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        create_new_stage()

        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode(lighting_mode="Stage Lights")

        assets_root = get_assets_root_path()
        from isaacsim.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(
            assets_root + "/Isaac/Environments/Grid/default_environment.usd", "/FlatGrid",
        )

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.CreateIntensityAttr(1000)

        articulation = self._scenario.load_example_assets()
        if articulation is not None:
            World.instance().scene.add(articulation)

    def _setup_scenario(self) -> None:
        self._scenario.setup()
        if self._state_btn:
            self._state_btn.reset()
            self._state_btn.enabled = True
        if self._reset_btn:
            self._reset_btn.enabled = True
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0

    def _pre_reset(self) -> None:
        """world.reset_async() 이전 정리 — physics callback/active player 제거.

        ResetButton 의 async 플로우는:
            pre_reset_fn() → world.reset_async() → post_reset_fn()
        순서인데, reset_async 실행 중에도 StateButton 의 physics callback 은
        계속 살아 있어 `apply_action` 을 articulation 이 재초기화되는 도중에
        호출한다. 이 경로에서 segfault 가 재현되므로, 여기서 콜백과
        트라젝토리 플레이어를 먼저 끊는다.
        """
        self._stop_logger(flush=False)
        player = self._scenario.get_trajectory_player()
        if player is not None:
            player.stop()
        if self._state_btn:
            self._state_btn.reset()
            self._state_btn.enabled = False

    def _on_reset(self) -> None:
        self._scenario.set_trajectory_player(None)
        self._scenario.reset()
        if self._state_btn:
            self._state_btn.reset()
            self._state_btn.enabled = True
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0
        self._set_replay_status("Status: idle")

    def _on_run(self) -> None:
        self._timeline.play()
        self._articulation_initialized = False
        self._physics_step_count = 0
        self._initialization_attempts = 0

    def _on_stop(self) -> None:
        self._timeline.pause()

    def _update_scenario(self, step: float, context=None) -> None:
        self._physics_step_count += 1

        if (not self._articulation_initialized
                and self._physics_step_count >= UIConfig.INITIALIZATION_DELAY_STEPS
                and self._initialization_attempts < UIConfig.MAX_INITIALIZATION_ATTEMPTS):
            self._initialization_attempts += 1
            success = self._scenario.delayed_initialization()
            if success:
                self._articulation_initialized = True

        done = self._scenario.update(step)
        if done and self._state_btn:
            self._state_btn.enabled = False

    # ==================================================================
    # AML Replay section
    # ==================================================================
    def _build_replay_section(self) -> None:
        frame = CollapsableFrame("AML Command Replay", collapsed=False)
        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                ui.Label(
                    f"CSV: {_DEFAULT_COMMAND_CSV.name} (dense, linear interp)",
                    height=UILayout.LABEL_HEIGHT,
                )
                with ui.HStack(height=UILayout.BUTTON_HEIGHT_LARGE):
                    UIComponentFactory.create_styled_button(
                        "Play", callback=self._on_replay_play,
                        color_scheme="green", height=UILayout.BUTTON_HEIGHT,
                    )
                self._replay_status_label = UIComponentFactory.create_status_label(
                    "Status: idle", UILayout.LABEL_WIDTH_XLARGE,
                )

    def _set_replay_status(self, text: str) -> None:
        if self._replay_status_label is not None:
            self._replay_status_label.text = text
        print(f"[Hysteresis][Replay] {text}")

    def _on_replay_play(self) -> None:
        articulation = self._scenario.get_articulation()
        if articulation is None or not getattr(articulation, "dof_names", None):
            self._set_replay_status("Status: LOAD + RUN first")
            return

        if not _DEFAULT_COMMAND_CSV.is_file():
            self._set_replay_status(f"Status: CSV missing {_DEFAULT_COMMAND_CSV.name}")
            return

        hz = self._query_physics_hz()
        seed_pose = self._scenario.get_seed_pose()
        player = AMLCommandPlayer(
            csv_path=_DEFAULT_COMMAND_CSV,
            articulation=articulation,
            joint_names=_R_INDEX_JOINTS,
            physics_hz=hz,
            seed_pose=seed_pose,
        )
        if not player.is_ready():
            self._set_replay_status(f"Status: CSV parse failed {_DEFAULT_COMMAND_CSV.name}")
            return

        # Replace any previous armed/active logger session before starting a
        # new replay — avoids mixing two runs into the same CSV.
        self._stop_logger(flush=False)

        self._scenario.set_trajectory_player(player)
        player.start()
        self._arm_logger(articulation, hz)
        self._set_replay_status(
            f"Status: Playing ({player.duration_s:.2f}s @ {hz:.0f}Hz, "
            f"ramp={player.ramp_in_s:.1f}s, hold={player.post_ramp_hold_s:.1f}s, "
            f"linear interp)"
        )

    def _query_physics_hz(self, default: float = 200.0) -> float:
        try:
            from isaacsim.physics.newton import acquire_stage
            st = acquire_stage()
            if st is not None:
                f = getattr(st, "physics_frequency", None)
                if f and f > 0:
                    return float(f)
        except Exception as exc:
            print(f"[Hysteresis][Replay] physics_frequency query failed: {exc}; using hz={default}")
        return default

    # ==================================================================
    # Measured Logger — fully automatic, no UI surface.
    # Play → arm → (ramp done) → record → player finished → auto-flush.
    # Status is reported via console prints only.
    # ==================================================================
    def _arm_logger(self, articulation, hz: float) -> None:
        """Create a fresh logger (timestamped path) and wait for playback start.

        Playback start = ramp 종료 + post-ramp hold 종료 순간. 이 시점이
        실제 CSV 커맨드가 들어가기 시작하는 틱이자 로깅 t=0 기준이 된다.
        """
        self._logger = MeasuredLogger(
            articulation=articulation,
            joint_names=_R_INDEX_JOINTS,
            out_path=_DEFAULT_LOG_PATH,
            hz=hz,
            timestamp=True,
        )
        self._logger_armed = True
        self._logger_active = False
        print(
            f"[Hysteresis][Logger] armed (waiting for ramp+hold end) → "
            f"{self._logger.out_path.name}"
        )

    def _begin_recording(self, hz: float) -> None:
        self._logger_armed = False
        self._logger_active = True
        print(f"[Hysteresis][Logger] recording @ {hz:.1f} Hz")

    def _stop_logger(self, flush: bool) -> None:
        if self._logger is None:
            return
        was_active = self._logger_active
        self._logger_armed = False
        self._logger_active = False
        if flush and was_active:
            samples = self._logger.sample_count
            path = self._logger.flush()
            print(f"[Hysteresis][Logger] flushed {samples} samples → {path.name}")
        elif not flush:
            print("[Hysteresis][Logger] cancelled (no flush)")
        self._logger = None

    # ==================================================================
    # Corrector placeholder (2차에서 활성화)
    # ==================================================================
    def _build_corrector_section(self) -> None:
        frame = CollapsableFrame("Hysteresis Corrector", collapsed=True)
        with frame:
            with ui.VStack(style=get_style(), spacing=UILayout.SPACING_SMALL, height=0):
                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    cb = ui.CheckBox(width=20)
                    cb.enabled = False
                    ui.Label("Apply PI Corrector (coming soon)")

    # ==================================================================
    # Extension event hooks — called by UIBuilder
    # ==================================================================
    def on_physics_step(self, step: float) -> None:
        player = self._scenario.get_trajectory_player()

        # Arm → record transition: wait until ramp + post-ramp hold are both
        # done so the first logged sample aligns with the tick that starts
        # issuing actual CSV commands. If the player was stopped before
        # reaching playback, drop the armed session.
        if self._logger_armed and self._logger is not None:
            if player is None or not player.is_active():
                self._stop_logger(flush=False)
            elif player.is_playback_started():
                self._begin_recording(self._query_physics_hz())

        if self._logger_active and self._logger is not None:
            self._logger.on_step()
            # Auto-flush when the CSV command stream ran out. `is_active()`
            # can also drop if a scenario/stage teardown path calls player.stop(),
            # so keep both conditions — they converge on the same cleanup.
            if player is not None and (
                not player.is_active() or player.is_finished()
            ):
                self._stop_logger(flush=True)

    def on_stage_event(self, event) -> None:
        from omni.usd import StageEventType
        if event.type == int(StageEventType.OPENED):
            self._stop_logger(flush=False)
            player = self._scenario.get_trajectory_player()
            if player is not None:
                player.stop()
            self._scenario.set_trajectory_player(None)
            if self._state_btn:
                self._state_btn.reset()
                self._state_btn.enabled = False
            if self._reset_btn:
                self._reset_btn.enabled = False

    def on_timeline_stop(self) -> None:
        if self._state_btn:
            self._state_btn.reset()
            self._state_btn.enabled = False

    def cleanup(self) -> None:
        self._stop_logger(flush=False)
        for elem in self._wrapped_elements:
            try:
                elem.cleanup()
            except Exception:
                pass
        self._wrapped_elements.clear()
        self._load_btn = None
        self._reset_btn = None
        self._state_btn = None
        self._replay_status_label = None
