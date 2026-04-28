"""Showcase Replay UI — kinematic CSV replay with sim/real overlay.

Folder layout (under ``extension_root/trajectory/``):

    trajectory/
      <group>/
          sim_*.csv     # exactly one sim_-prefixed csv (optional)
          real_*.csv    # exactly one real_-prefixed csv (optional)

A group needs at least one of (sim, real) to appear in the dropdown.
The user picks which channel is "main" (drives kinematic playback);
both channels' torque/force are always pushed to the visualizer, and
sim/real visibility is controlled by the existing Visualizer Controls
mode (off/real/sim/both).
"""
from __future__ import annotations

from pathlib import Path

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style

from .ui_components import UIComponentFactory
from ..config.ui_config import UILayout
from ..replay import CsvReplayer, ShowcaseReader

_EXT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAJECTORY_DIR = _EXT_ROOT / "trajectory"


def _scan_group_csvs(group_dir: Path) -> tuple[Path | None, Path | None]:
    """``(sim_csv, real_csv)`` strict-prefix match. None 이면 그쪽 채널 없음."""
    sim = None
    real = None
    if not group_dir.is_dir():
        return None, None
    for p in sorted(group_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if p.name.startswith("sim_") and sim is None:
            sim = p
        elif p.name.startswith("real_") and real is None:
            real = p
    return sim, real


class ShowcaseReplayControls:
    """Showcase Replay collapsable section."""

    def __init__(self, ui_ref):
        self._ui = ui_ref
        self._combo_frame: ui.Frame | None = None
        self._combo: ui.ComboBox | None = None
        self._items: list[str] = []
        self._availability_label: ui.Label | None = None
        self._main_combo: ui.ComboBox | None = None
        self._status_label: ui.Label | None = None

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------
    def build(self) -> None:
        frame = CollapsableFrame("Showcase Replay", collapsed=True)
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

                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Available:", width=80)
                    self._availability_label = ui.Label("(select a group)")

                with ui.HStack(height=UILayout.BUTTON_HEIGHT):
                    ui.Label("Main source:", width=UILayout.LABEL_WIDTH_LARGE)
                    self._main_combo = ui.ComboBox(0, "sim", "real",
                                                   height=UILayout.BUTTON_HEIGHT)

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

        # group 선택 변경 시 availability 라벨 갱신
        self._update_availability_label()

    def cleanup(self) -> None:
        scenario = getattr(self._ui, "_scenario", None)
        if scenario is not None and hasattr(scenario, "get_csv_replayer"):
            replayer = scenario.get_csv_replayer()
            if replayer is not None:
                try:
                    replayer.stop()
                except Exception:
                    pass
                try:
                    scenario.set_csv_replayer(None)
                except Exception:
                    pass
        self._combo_frame = None
        self._combo = None
        self._main_combo = None
        self._status_label = None
        self._availability_label = None

    # ------------------------------------------------------------------
    # Dropdown
    # ------------------------------------------------------------------
    def _scan_groups(self) -> list[str]:
        if not _TRAJECTORY_DIR.is_dir():
            return []
        names: list[str] = []
        for p in _TRAJECTORY_DIR.iterdir():
            if not p.is_dir() or p.name.startswith("."):
                continue
            sim, real = _scan_group_csvs(p)
            if sim is None and real is None:
                continue
            names.append(p.name)
        return sorted(names)

    def _rebuild_combo(self) -> None:
        if self._combo_frame is None:
            return
        self._items = self._scan_groups()
        self._combo_frame.clear()
        with self._combo_frame:
            if self._items:
                self._combo = ui.ComboBox(0, *self._items, height=UILayout.BUTTON_HEIGHT)
                # 콤보 변경 시 availability label 갱신.
                try:
                    self._combo.model.add_item_changed_fn(
                        lambda *_: self._update_availability_label()
                    )
                except Exception:
                    pass
            else:
                self._combo = None
                ui.Label("(no showcase groups)", height=UILayout.BUTTON_HEIGHT)

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

    def _get_main_source(self) -> str:
        if self._main_combo is None:
            return "sim"
        try:
            idx = self._main_combo.model.get_item_value_model().get_value_as_int()
        except Exception:
            return "sim"
        return "real" if idx == 1 else "sim"

    def _update_availability_label(self) -> None:
        if self._availability_label is None:
            return
        group = self._get_selected_group()
        if group is None:
            self._availability_label.text = "(select a group)"
            return
        sim, real = _scan_group_csvs(_TRAJECTORY_DIR / group)
        sim_tag = "sim [yes]" if sim else "sim [missing]"
        real_tag = "real [yes]" if real else "real [missing]"
        self._availability_label.text = f"{sim_tag}  |  {real_tag}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.text = text
        print(f"[ALLEX][Showcase] {text}")

    def _resolve_articulation(self):
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

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _on_refresh(self) -> None:
        self._rebuild_combo()
        self._update_availability_label()
        self._set_status(f"Status: refreshed ({len(self._items)} group(s))")

    def _on_run(self) -> None:
        group = self._get_selected_group()
        if group is None:
            self._set_status("Status: no group selected")
            return

        sim_csv, real_csv = _scan_group_csvs(_TRAJECTORY_DIR / group)
        if sim_csv is None and real_csv is None:
            self._set_status(f"Status: '{group}' has no sim_*.csv or real_*.csv")
            return

        scenario, articulation = self._resolve_articulation()
        if scenario is None:
            return

        main_src = self._get_main_source()
        main_csv = sim_csv if main_src == "sim" else real_csv
        sec_csv = real_csv if main_src == "sim" else sim_csv
        if main_csv is None:
            self._set_status(
                f"Status: main source CSV missing — '{main_src}_*.csv' not in '{group}'"
            )
            return

        try:
            reader_main = ShowcaseReader(main_csv)
        except Exception as exc:
            self._set_status(f"Status: failed to load main CSV: {exc}")
            return

        reader_sec = None
        if sec_csv is not None:
            try:
                reader_sec = ShowcaseReader(sec_csv)
            except Exception as exc:
                self._set_status(f"Status: failed to load secondary CSV: {exc}")
                return

        viz = scenario.get_visualizer() if hasattr(scenario, "get_visualizer") else None
        if viz is None:
            self._set_status("Status: visualizer not ready")
            return

        # plot 도 CSV 값으로 그리도록 plotter 들 수집 (있으면).
        plotters = []
        for attr in ("_torque_plotter_body", "_torque_plotter_hand"):
            p = getattr(scenario, attr, None)
            if p is not None:
                plotters.append(p)

        try:
            replayer = CsvReplayer(
                reader_main=reader_main,
                reader_secondary=reader_sec,
                articulation=articulation,
                visualizer=viz,
                main_source=main_src,
                plotters=plotters,
            )
        except Exception as exc:
            self._set_status(f"Status: replayer init failed: {exc}")
            return

        try:
            scenario.set_csv_replayer(replayer)
        except Exception as exc:
            self._set_status(f"Status: scenario.set_csv_replayer failed: {exc}")
            return
        replayer.start()

        sec_tag = "none" if reader_sec is None else f"{'real' if main_src == 'sim' else 'sim'}"
        self._set_status(
            f"Status: Playing main={main_src} ({reader_main.duration_s:.2f}s, "
            f"sec={sec_tag})"
        )

    def _on_stop(self) -> None:
        scenario = getattr(self._ui, "_scenario", None)
        if scenario is None:
            self._set_status("Status: scenario not ready")
            return
        replayer = scenario.get_csv_replayer() if hasattr(scenario, "get_csv_replayer") else None
        if replayer is None:
            self._set_status("Status: nothing to stop")
            return
        try:
            replayer.stop()
        except Exception as exc:
            self._set_status(f"Status: stop warn: {exc}")
            return
        try:
            scenario.set_csv_replayer(None)
        except Exception:
            pass
        self._set_status("Status: Stopped")
