"""
Visualizer Controls — ALLEX model visualization options

- Force Visualizer 토글: off ↔ 직전 모드 (real/sim/both)
- Viz Mode ComboBox: real / sim / both 전환
- Torque Rings 토글: 링 가시성
"""

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style

from .ui_components import UIComponentFactory
from ..config import UIConfig
from ..config.ui_config import UILayout

_ROBOT_ROOT = "/ALLEX"
_VIZ_SKIP = ("torque_ring", "force_viz", "force_vec")
_OPACITY_VALUE = 0.3

_MODE_OFF = "off"
_MODE_REAL = "real"
_MODE_SIM = "sim"
_MODE_BOTH = "both"
_MODE_CHOICES = [_MODE_REAL, _MODE_SIM, _MODE_BOTH]


class VisualizerControls:
    """Visualizer section — model visualization options"""

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

        # UI refs
        self._force_viz_status_label = None
        self._mode_combo = None
        self._torque_mode_combo = None
        self._torque_viz_status_label = None
        self._opacity_status_label = None

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
                    color_scheme='yellow',
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

    # ========================================
    # Toggle — Force Visualizer
    # ========================================
    def _toggle_force_viz(self):
        # scenario 가 없거나 visualizer 가 없으면 legacy fallback (real 화살표 직접 토글)
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

        # S5: mode 표시 포함
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
        if visualizer is not None and self._force_viz_enabled:
            try:
                visualizer.set_mode(self._viz_mode)
            except Exception as e:
                print(f"[Visualizer] set_mode failed: {e}")
        # S5: status label 에도 현재 mode 반영
        if self._force_viz_status_label and self._force_viz_enabled:
            self._force_viz_status_label.text = f"Force Viz: ON ({self._viz_mode})"
        print(f"[Visualizer] Mode -> {self._viz_mode}")

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
            # info:id=UsdPreviewSurface 여도 implementationSource=sourceAsset 이면
            # 실제 렌더는 OmniPBR.mdl 이 담당 → MDL 파라미터로 써야 반영됨
            self._set_attr(prim, path, "inputs:enable_opacity", Sdf.ValueTypeNames.Bool, True)
            self._set_attr(prim, path, "inputs:opacity_constant", Sdf.ValueTypeNames.Float, opacity)

    def _restore_opacity(self, stage):
        from pxr import UsdShade
        for path_str, attrs in self._opacity_backup.items():
            prim = stage.GetPrimAtPath(path_str)
            if not prim.IsValid():
                continue
            shader = UsdShade.Shader(prim)
            for attr_name, original in attrs.items():
                input_name = attr_name.removeprefix("inputs:")
                try:
                    inp = shader.GetInput(input_name)
                    if original is None:
                        inp.GetAttr().Clear()
                    else:
                        inp.Set(original)
                except Exception:
                    pass
        self._opacity_backup.clear()

    def _set_attr(self, prim, path_str, attr_name, vtype, value):
        try:
            from pxr import UsdShade
            input_name = attr_name.removeprefix("inputs:")
            shader = UsdShade.Shader(prim)
            # 기존 값 백업 (처음 한 번만)
            backup = self._opacity_backup.setdefault(path_str, {})
            if attr_name not in backup:
                try:
                    backup[attr_name] = shader.GetInput(input_name).Get()
                except Exception:
                    backup[attr_name] = None
            # CreateInput: 없으면 타입과 함께 생성, 있으면 기존 반환 (idempotent)
            inp = shader.CreateInput(input_name, vtype)
            inp.Set(value)
        except Exception as e:
            print(f"[Opacity] {path_str} / {attr_name}: {e}")

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
