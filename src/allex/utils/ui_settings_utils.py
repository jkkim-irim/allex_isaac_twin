"""UI data + helpers backed by `src/allex/config/ui_config.json`.

Three data classes (``UIColors`` / ``UILayout`` / ``UIConfig``) loaded from
JSON, plus two stateless helper classes (``UIComponentFactory`` /
``ButtonStyleManager``) used across panels.

Color values are authored as ``"0xAARRGGBB"`` strings in JSON for readability
and parsed to ``int`` here. Numeric/boolean fields pass through unchanged.

The helpers live here (not in `ui.py`) so domain modules like
`contact_force_viz.py` can import `UIComponentFactory` without creating a
circular dependency on the panel orchestrator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "ui_config.json"
)


def _load() -> dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_color(v: Any) -> int:
    """Accept ``"0xAARRGGBB"`` / ``"#AARRGGBB"`` / int — return int."""
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s.startswith("#"):
        s = "0x" + s[1:]
    return int(s, 16)


_cfg = _load()


class UIColors:
    """UI 색상 상수들 (AARRGGBB int)."""
    pass


for _k, _v in _cfg.get("colors", {}).items():
    setattr(UIColors, _k, _parse_color(_v))


class UILayout:
    """UI 레이아웃 상수들 (px / cell)."""
    pass


for _k, _v in _cfg.get("layout", {}).items():
    setattr(UILayout, _k, _v)


class UIConfig:
    """UI 전반적인 설정."""
    pass


for _k, _v in _cfg.get("config", {}).items():
    setattr(UIConfig, _k, _v)


del _k, _v


# ---------------------------------------------------------------------------
# UI Component Factory
# ---------------------------------------------------------------------------

import omni.ui as ui


class UIComponentFactory:
    """UI 컴포넌트 생성 팩토리 — stateless, omni.ui wrapper."""

    @staticmethod
    def create_separator(height=UILayout.SEPARATOR_HEIGHT):
        return ui.Separator(height=height)

    @staticmethod
    def create_spacer(width=UILayout.SPACING_SMALL):
        return ui.Spacer(width=width)

    @staticmethod
    def create_status_label(text, width=UILayout.LABEL_WIDTH_LARGE):
        return ui.Label(text, width=width)

    @staticmethod
    def create_colored_sidebar(color, width=UILayout.SIDEBAR_WIDTH, height=UILayout.BUTTON_HEIGHT):
        return ui.Rectangle(
            width=width,
            height=height,
            style={
                "background_color": color,
                "border_radius": UILayout.BUTTON_BORDER_RADIUS,
            },
        )

    @staticmethod
    def create_styled_button(text, callback=None, color_scheme="default", height=UILayout.BUTTON_HEIGHT):
        """스타일이 적용된 버튼 생성"""
        style_map = {
            "green": {
                "Button": {
                    "background_color": UIColors.STATE_BUTTON_BG,
                    "border_width": UILayout.BUTTON_BORDER_WIDTH,
                    "border_color": UIColors.STATE_BUTTON_BORDER,
                    "border_radius": UILayout.BUTTON_BORDER_RADIUS_LARGE,
                },
                "Button:hovered": {"background_color": UIColors.STATE_BUTTON_HOVER},
            },
            "yellow": {
                "Button": {
                    "background_color": UIColors.YELLOW_BUTTON_BG,
                    "border_width": UILayout.BUTTON_BORDER_WIDTH,
                    "border_color": UIColors.YELLOW_BUTTON_BORDER,
                    "border_radius": UILayout.BUTTON_BORDER_RADIUS_LARGE,
                },
                "Button:hovered": {"background_color": UIColors.YELLOW_BUTTON_HOVER},
            },
            "blue": {
                "Button": {
                    "background_color": UIColors.BLUE_BUTTON_BG,
                    "border_width": UILayout.BUTTON_BORDER_WIDTH,
                    "border_color": UIColors.BLUE_BUTTON_BORDER,
                    "border_radius": UILayout.BUTTON_BORDER_RADIUS_LARGE,
                },
                "Button:hovered": {"background_color": UIColors.BLUE_BUTTON_HOVER},
            },
        }
        style = style_map.get(color_scheme)
        if style is not None:
            return ui.Button(text, clicked_fn=callback, height=height, style=style)
        return ui.Button(text, clicked_fn=callback, height=height)


# ---------------------------------------------------------------------------
# Button Style Manager
# ---------------------------------------------------------------------------

class ButtonStyleManager:
    """LOAD / RESET / STATE 버튼 스타일 — stateless, isaacsim.gui get_style() 위에 덮어씀."""

    @staticmethod
    def _load_get_style():
        from isaacsim.gui.components.ui_utils import get_style
        return get_style()

    @staticmethod
    def get_load_button_style():
        return {
            **ButtonStyleManager._load_get_style(),
            "Button": {
                "background_color": UIColors.LOAD_BUTTON_BG,
                "border_width": 2,
                "border_color": UIColors.LOAD_BUTTON_BORDER,
                "border_radius": 4,
                "margin": UILayout.BUTTON_MARGIN,
                "padding": UILayout.BUTTON_PADDING,
            },
            "Button:hovered": {"background_color": UIColors.LOAD_BUTTON_HOVER},
            "Button.Label": {"color": UIColors.TEXT_PRIMARY},
        }

    @staticmethod
    def get_reset_button_style():
        return {
            **ButtonStyleManager._load_get_style(),
            "Button": {
                "background_color": UIColors.RED_BUTTON_BG,
                "border_width": 2,
                "border_color": UIColors.RED_BUTTON_BORDER,
                "border_radius": 4,
                "margin": UILayout.BUTTON_MARGIN,
                "padding": UILayout.BUTTON_PADDING,
            },
            "Button:hovered": {"background_color": UIColors.RED_BUTTON_HOVER},
            "Button.Label": {"color": UIColors.TEXT_PRIMARY},
        }

    @staticmethod
    def get_state_button_style():
        return {
            **ButtonStyleManager._load_get_style(),
            "Button": {
                "background_color": UIColors.STATE_BUTTON_BG,
                "border_width": 2,
                "border_color": UIColors.STATE_BUTTON_BORDER,
                "border_radius": 4,
                "margin": UILayout.BUTTON_MARGIN,
                "padding": UILayout.BUTTON_PADDING,
            },
            "Button:hovered": {"background_color": UIColors.STATE_BUTTON_HOVER},
            "Button.Label": {"color": UIColors.TEXT_PRIMARY},
        }

    @staticmethod
    def apply_button_styles(load_btn, reset_btn, state_btn):
        """LoadButton / ResetButton / StateButton 의 내부 ui.Button 위젯에 스타일 주입."""
        try:
            if hasattr(load_btn, "_button") and load_btn._button:
                load_btn._button.style = ButtonStyleManager.get_load_button_style()
            if hasattr(reset_btn, "_button") and reset_btn._button:
                reset_btn._button.style = ButtonStyleManager.get_reset_button_style()
            if hasattr(state_btn, "_state_button") and state_btn._state_button:
                state_btn._state_button.style = ButtonStyleManager.get_state_button_style()
        except Exception as exc:
            print(f"[ALLEX][UI] could not apply button styles: {exc}")
