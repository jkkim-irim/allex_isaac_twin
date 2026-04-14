"""
재사용 가능한 UI 컴포넌트들
"""

import omni.ui as ui
from ..config.ui_config import UIColors, UILayout


class UIComponentFactory:
    """UI 컴포넌트 생성 팩토리"""

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
    def create_styled_button(text, callback=None, color_scheme='default', height=UILayout.BUTTON_HEIGHT):
        """스타일이 적용된 버튼 생성"""
        style_map = {
            'green': {
                "Button": {
                    "background_color": UIColors.STATE_BUTTON_BG,
                    "border_width": UILayout.BUTTON_BORDER_WIDTH,
                    "border_color": UIColors.STATE_BUTTON_BORDER,
                    "border_radius": UILayout.BUTTON_BORDER_RADIUS_LARGE,
                },
                "Button:hovered": {"background_color": UIColors.STATE_BUTTON_HOVER},
            },
            'yellow': {
                "Button": {
                    "background_color": UIColors.YELLOW_BUTTON_BG,
                    "border_width": UILayout.BUTTON_BORDER_WIDTH,
                    "border_color": UIColors.YELLOW_BUTTON_BORDER,
                    "border_radius": UILayout.BUTTON_BORDER_RADIUS_LARGE,
                },
                "Button:hovered": {"background_color": UIColors.YELLOW_BUTTON_HOVER},
            },
            'blue': {
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
