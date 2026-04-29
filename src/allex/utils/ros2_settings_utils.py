"""ROS2 settings — class wrappers backed by `src/allex/config/ros2_config.json`.

Preserves the same attribute API the codebase already uses
(``ROS2Config.NODE_NAME``, ``ROS2Config.R_HAND_JOINT_INDICES``,
``ROS2Config.get_outbound_topics_by_mode(...)`` etc.) so call sites only need
their import path swapped.

JSON structure
--------------
``node_name`` / ``bridge_extension`` / timeouts / booleans  — direct constants.

``default_domain_id`` / ``default_rmw_implementation``      — overridden by the
    ``ROS_DOMAIN_ID`` / ``RMW_IMPLEMENTATION`` env vars at import time.

``topic_mode``        — current/desired suffixes + display-name map.
``joint_groups``      — `{group: {indices: [...], names: [...]}}` flattened to
                         ``{R,L}_{HAND,ARM}_JOINT_{INDICES,NAMES}`` plus
                         ``ALL_JOINT_INDICES`` / ``ALL_JOINT_NAMES``.
``outbound_topic_to_joints`` — mapping for outbound publishers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "ros2_config.json"
)


def _load() -> dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_cfg = _load()
_groups = _cfg["joint_groups"]
_tmode = _cfg["topic_mode"]


class ROS2Topics:
    """ROS2 토픽 이름들"""
    ROBOT_STATE = _cfg["topics"]["robot_state"]


class ROS2QoS:
    """ROS2 QoS 설정"""
    HISTORY_DEPTH = int(_cfg["qos"]["history_depth"])


class ROS2Config:
    """ROS2 통신 전반적인 설정 (mutable: DOMAIN_ID can be reassigned at runtime)."""
    NODE_NAME          = _cfg["node_name"]
    BRIDGE_EXTENSION   = _cfg["bridge_extension"]
    DOMAIN_ID          = int(os.environ.get("ROS_DOMAIN_ID", _cfg["default_domain_id"]))
    RMW_IMPLEMENTATION = os.environ.get("RMW_IMPLEMENTATION", _cfg["default_rmw_implementation"])

    INIT_TIMEOUT     = float(_cfg["init_timeout"])
    SHUTDOWN_TIMEOUT = float(_cfg["shutdown_timeout"])
    THREAD_DAEMON    = bool(_cfg["thread_daemon"])
    EXECUTOR_TIMEOUT = float(_cfg["executor_timeout"])

    TOPIC_MODE_CURRENT  = _tmode["current"]
    TOPIC_MODE_DESIRED  = _tmode["desired"]
    DEFAULT_TOPIC_MODE  = _tmode["default"]

    # In the original schema TOPIC_SUFFIXES is an identity map keyed by the
    # mode string itself (used as `cfg.TOPIC_SUFFIXES[mode]` for lookups).
    TOPIC_SUFFIXES = {
        TOPIC_MODE_CURRENT: TOPIC_MODE_CURRENT,
        TOPIC_MODE_DESIRED: TOPIC_MODE_DESIRED,
    }
    TOPIC_MODE_DISPLAY_NAMES = dict(_tmode["display_names"])

    R_HAND_JOINT_INDICES = list(_groups["right_hand"]["indices"])
    L_HAND_JOINT_INDICES = list(_groups["left_hand"]["indices"])
    R_ARM_JOINT_INDICES  = list(_groups["right_arm"]["indices"])
    L_ARM_JOINT_INDICES  = list(_groups["left_arm"]["indices"])
    WAIST_JOINT_INDICES  = list(_groups["waist"]["indices"])
    NECK_JOINT_INDICES   = list(_groups["neck"]["indices"])

    R_HAND_JOINT_NAMES = list(_groups["right_hand"]["names"])
    L_HAND_JOINT_NAMES = list(_groups["left_hand"]["names"])
    R_ARM_JOINT_NAMES  = list(_groups["right_arm"]["names"])
    L_ARM_JOINT_NAMES  = list(_groups["left_arm"]["names"])
    WAIST_JOINT_NAMES  = list(_groups["waist"]["names"])
    NECK_JOINT_NAMES   = list(_groups["neck"]["names"])

    ALL_JOINT_INDICES = {g: list(d["indices"]) for g, d in _groups.items()}
    ALL_JOINT_NAMES   = {g: list(d["names"])   for g, d in _groups.items()}

    OUTBOUND_TOPIC_TO_JOINTS = {
        k: list(v) for k, v in _cfg["outbound_topic_to_joints"].items()
    }

    # Torque 토픽 (real2sim 토크 시각화용). rosbag torque 토픽 패턴 확정 시
    # ros2_config.json::outbound_torque_topic_to_joints 채울 것.
    OUTBOUND_TORQUE_TOPIC_TO_JOINTS = {
        k: list(v) for k, v in _cfg.get("outbound_torque_topic_to_joints", {}).items()
    }

    @classmethod
    def get_torque_topics(cls) -> list[tuple[str, dict[str, Any]]]:
        """Real torque 토픽 목록 반환.

        OUTBOUND_TORQUE_TOPIC_TO_JOINTS 가 비어있으면 빈 리스트.
        각 원소는 (topic_full_name, {joint_names, group_name}) 튜플.
        """
        if not cls.OUTBOUND_TORQUE_TOPIC_TO_JOINTS:
            return []

        entries: list[tuple[str, dict[str, Any]]] = []
        for group_name, joint_names in cls.OUTBOUND_TORQUE_TOPIC_TO_JOINTS.items():
            # TODO: 실제 토픽 네임스페이스/suffix 확정 시 아래 규칙 수정.
            topic = f"/robot_outbound_data/{group_name}/joint_torque"
            entries.append((topic, {
                "joint_names": list(joint_names),
                "group_name":  group_name,
            }))
        return entries

    @classmethod
    def get_outbound_topics_by_mode(cls, topic_mode: str) -> dict[str, dict[str, Any]]:
        """특정 모드에 맞는 14개 outbound 토픽 반환"""
        if topic_mode not in cls.TOPIC_SUFFIXES:
            raise ValueError(f"Invalid topic mode: {topic_mode}")
        suffix = cls.TOPIC_SUFFIXES[topic_mode]
        outbound_topics: dict[str, dict[str, Any]] = {}
        for group_name, joint_names in cls.OUTBOUND_TOPIC_TO_JOINTS.items():
            topic = f"/robot_outbound_data/{group_name}/{suffix}"
            outbound_topics[topic] = {
                "joint_names": joint_names,
                "group_name":  group_name,
            }
        return outbound_topics

    @classmethod
    def get_available_topic_modes(cls) -> list[str]:
        return list(cls.TOPIC_SUFFIXES.keys())

    @classmethod
    def is_valid_topic_mode(cls, topic_mode: str) -> bool:
        return topic_mode in cls.TOPIC_SUFFIXES

    @classmethod
    def get_topic_mode_display_name(cls, topic_mode: str) -> str:
        return cls.TOPIC_MODE_DISPLAY_NAMES.get(topic_mode, topic_mode)
