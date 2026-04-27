"""Traj Studio — Cubic Hermite spline trajectory playback for ALLEX.

Scans `extension_root/trajectory/` for group folders (each containing
the 14 per-region CSVs), lets the user pick one via dropdown, and plays
it back via `TrajectoryPlayer` installed on the scenario.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style

from .ui_components import UIComponentFactory
from ..config.ui_config import UILayout
from ..trajectory import TrajectoryPlayer


_EXT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAJECTORY_DIR = _EXT_ROOT / "trajectory"

_RESET_RAMP_S = 3.0


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

    def _on_stop(self) -> None:
        scenario = getattr(self._ui, "_scenario", None)
        player = scenario.get_trajectory_player() if scenario is not None else None
        if player is None:
            self._set_status("Status: nothing to stop")
            return
        player.stop()
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
