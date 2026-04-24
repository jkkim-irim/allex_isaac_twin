"""
ALLEX Digital Twin 설정 모듈
"""
import json
from pathlib import Path

from .ui_config import UIConfig
from .ros2_config import ROS2Config, ROS2Topics, ROS2QoS
from . import physics_settings


JOINT_CONFIG_PATH = Path(__file__).resolve().parent / "joint_config.json"


def load_joint_config_json() -> dict:
    with open(JOINT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_drive_gains() -> dict[str, tuple[float, float]]:
    """Return {joint_name: (stiffness, damping)} from joint_config.json.

    Entries whose key starts with ``_`` (e.g. ``_notes``) are skipped so the
    JSON can carry inline documentation alongside the data.
    """
    raw = load_joint_config_json().get("drive_gains", {})
    return {
        name: (float(entry["stiffness"]), float(entry["damping"]))
        for name, entry in raw.items()
        if not name.startswith("_")
    }


_ui = load_joint_config_json().get("ui", {})


class JointConfig:
    """관절 제어 관련 설정 — values loaded from joint_config.json[ui]."""
    EFFECTIVE_JOINTS = _ui["effective_joints"]
    CONTROL_JOINTS = _ui["control_joints"]
    JOINT_MIN_ANGLE = _ui["joint_min_angle"]
    JOINT_MAX_ANGLE = _ui["joint_max_angle"]
    JOINT_STEP = _ui["joint_step"]
    JOINT_NAME_TEMPLATE = _ui["joint_name_template"]
    MAX_INITIALIZATION_ATTEMPTS = _ui["max_initialization_attempts"]
    INITIALIZATION_DELAY_STEPS = _ui["initialization_delay_steps"]


__all__ = [
    'UIConfig',
    'ROS2Config',
    'JointConfig',
    'ROS2Topics',
    'ROS2QoS',
    'physics_settings',
    'load_joint_config_json',
    'load_drive_gains',
    'JOINT_CONFIG_PATH',
]
