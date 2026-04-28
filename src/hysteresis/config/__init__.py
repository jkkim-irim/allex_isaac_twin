"""Hysteresis 전용 설정 패키지.

- `JOINT_CONFIG_PATH` — MJCF-style equality table (joint_config.json).
- `UIConfig`          — UI/lifecycle knobs.
- `PhysicsConfig`     — world/physics knobs (dt, gravity).
"""

from pathlib import Path

from .physics_config import PhysicsConfig
from .ui_config import UIConfig

JOINT_CONFIG_PATH = Path(__file__).resolve().parent / "joint_config.json"

__all__ = ["JOINT_CONFIG_PATH", "UIConfig", "PhysicsConfig"]
