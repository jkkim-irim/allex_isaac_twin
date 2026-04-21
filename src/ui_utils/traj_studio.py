"""Traj Studio — Cubic Hermite spline trajectory playback for ALLEX.

Scans `extension_root/trajectory/` for group folders (each containing
the 14 per-region CSVs), lets the user pick one via dropdown, and plays
it back via `TrajectoryPlayer` installed on the scenario.
"""
from __future__ import annotations

from pathlib import Path

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style

from .ui_components import UIComponentFactory
from ..config.ui_config import UILayout
from ..trajectory import TrajectoryPlayer


_EXT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAJECTORY_DIR = _EXT_ROOT / "trajectory"


class TrajStudioControls:
    """Traj Studio collapsable section — Dropdown / Refresh / Run / Stop."""

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

                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    UIComponentFactory.create_styled_button(
                        "Refresh",
                        callback=self._on_refresh,
                        color_scheme="blue",
                        height=UILayout.BUTTON_HEIGHT,
                    )

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
    # Button callbacks
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.text = text
        print(f"[ALLEX][Traj] {text}")

    def _on_refresh(self) -> None:
        self._rebuild_combo()
        self._set_status(f"Status: scanned {len(self._items)} group(s)")

    def _on_run(self) -> None:
        group = self._get_selected_group()
        if group is None:
            self._set_status("Status: no group selected")
            return

        scenario = getattr(self._ui, "_scenario", None)
        if scenario is None:
            self._set_status("Status: scenario not ready")
            return
        articulation = scenario._asset_manager.get_articulation()
        if articulation is None:
            self._set_status("Status: load robot (LOAD) + start sim (RUN) first")
            return
        if not getattr(articulation, "dof_names", None):
            self._set_status("Status: articulation not initialized — press RUN first")
            return

        # Prefer the initializer's neutral pose as the seed; it avoids
        # snapping non-covered joints to 0 rad when live pose can't be read.
        seed_pose = None
        initializer = getattr(scenario, "_initializer", None)
        if initializer is not None:
            seed_pose = getattr(initializer, "target_joint_positions", None)

        # Match dense-trajectory sample rate to physics step rate, otherwise
        # playback runs slower/faster than real-time (one sample consumed per
        # physics step). Query Newton stage for actual physics_frequency.
        hz = 200.0
        try:
            from isaacsim.physics.newton import acquire_stage
            st = acquire_stage()
            if st is not None:
                f = getattr(st, "physics_frequency", None)
                if f and f > 0:
                    hz = float(f)
        except Exception as exc:
            print(f"[ALLEX][Traj] physics_frequency query failed: {exc}; using hz={hz}")

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

    def _on_stop(self) -> None:
        scenario = getattr(self._ui, "_scenario", None)
        player = scenario.get_trajectory_player() if scenario is not None else None
        if player is None:
            self._set_status("Status: nothing to stop")
            return
        player.stop()
        self._set_status("Status: Stopped (holding current target)")
