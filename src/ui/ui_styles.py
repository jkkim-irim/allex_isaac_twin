"""
UI 스타일링 관련 로직
"""

from isaacsim.gui.components.ui_utils import get_style
from ..config.ui_config import UIColors, UILayout


class ButtonStyleManager:
    """버튼 스타일 관리 클래스"""
    
    @staticmethod
    def get_load_button_style():
        """LOAD 버튼 스타일"""
        return {
            **get_style(),
            "Button": {
                "background_color": UIColors.LOAD_BUTTON_BG,
                "border_width": 2,
                "border_color": UIColors.LOAD_BUTTON_BORDER,
                "border_radius": 4,
                "margin": UILayout.BUTTON_MARGIN,
                "padding": UILayout.BUTTON_PADDING
            },
            "Button:hovered": {"background_color": UIColors.LOAD_BUTTON_HOVER},
            "Button.Label": {"color": UIColors.TEXT_PRIMARY}
        }
    
    @staticmethod
    def get_reset_button_style():
        """RESET 버튼 스타일"""
        return {
            **get_style(),
            "Button": {
                "background_color": UIColors.RED_BUTTON_BG,
                "border_width": 2,
                "border_color": UIColors.RED_BUTTON_BORDER,
                "border_radius": 4,
                "margin": UILayout.BUTTON_MARGIN,
                "padding": UILayout.BUTTON_PADDING
            },
            "Button:hovered": {"background_color": UIColors.RED_BUTTON_HOVER},
            "Button.Label": {"color": UIColors.TEXT_PRIMARY}
        }
    
    @staticmethod
    def get_state_button_style():
        """STATE 버튼 스타일"""
        return {
            **get_style(),
            "Button": {
                "background_color": UIColors.STATE_BUTTON_BG,
                "border_width": 2,
                "border_color": UIColors.STATE_BUTTON_BORDER,
                "border_radius": 4,
                "margin": UILayout.BUTTON_MARGIN,
                "padding": UILayout.BUTTON_PADDING
            },
            "Button:hovered": {"background_color": UIColors.STATE_BUTTON_HOVER},
            "Button.Label": {"color": UIColors.TEXT_PRIMARY}
        }
    
    @staticmethod
    def apply_button_styles(load_btn, reset_btn, state_btn):
        """버튼들에 스타일 적용"""
        try:
            # LOAD 버튼 스타일 적용
            if hasattr(load_btn, '_button') and load_btn._button:
                load_btn._button.style = ButtonStyleManager.get_load_button_style()
            
            # RESET 버튼 스타일 적용
            if hasattr(reset_btn, '_button') and reset_btn._button:
                reset_btn._button.style = ButtonStyleManager.get_reset_button_style()
            
            # STATE 버튼 스타일 적용
            if hasattr(state_btn, '_state_button') and state_btn._state_button:
                state_btn._state_button.style = ButtonStyleManager.get_state_button_style()
                        
        except Exception as e:
            print(f"⚠️ Could not apply button styles: {e}")