"""
Visualizer Controls — ALLEX model visualization options
"""

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style

from .ui_components import UIComponentFactory
from ..config import UIConfig
from ..config.ui_config import UILayout


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
                    color_scheme='yellow',
                    height=UILayout.BUTTON_HEIGHT,
                )

                self._force_viz_status_label = UIComponentFactory.create_status_label(
                    "Force Viz: OFF", UILayout.LABEL_WIDTH_LARGE,
                )

    # ========================================
    # Force Visualizer
    # ========================================
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
