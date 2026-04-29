"""Force / Torque Visualization 설정 — JSON 로더.

데이터는 모두 ``viz_config.json`` 에 있고 이 모듈은 그걸 읽어서 module-level
상수들과 ``HAND_JOINT_TORQUE_RING_MAP`` / ``HAND_JOINT_TORQUE_RING_TUPLES`` /
``FORCE_VIZ_PARENT_LINKS`` 를 구성한다. 기존 import 경로 (``from
..config.viz_config import REAL_COLOR, TORQUE_GAIN, ...``) 는 그대로 동작.

다른 *_config.json 들 (physics, ros2, ui, joint, ...) 과 일관성 맞추기 위해
data ↔ logic 분리.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# JSON 로드
# ---------------------------------------------------------------------------
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH = os.path.join(_CONFIG_DIR, "viz_config.json")
EXT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "..", "..", ".."))

with open(_JSON_PATH, "r", encoding="utf-8") as _f:
    _cfg = json.load(_f)


def _tup3(v) -> tuple:
    return (float(v[0]), float(v[1]), float(v[2]))


# ---------------------------------------------------------------------------
# 색상
# ---------------------------------------------------------------------------
REAL_COLOR = _tup3(_cfg["colors"]["real"])
SIM_COLOR = _tup3(_cfg["colors"]["sim"])


# ---------------------------------------------------------------------------
# Torque 링 스케일
# ---------------------------------------------------------------------------
_t = _cfg["torque"]
TORQUE_GAIN          = float(_t["gain_default"])
TORQUE_GAIN_SHOULDER = float(_t["gain_shoulder"])
TORQUE_GAIN_ELBOW    = float(_t["gain_elbow"])
TORQUE_GAIN_WRIST    = float(_t["gain_wrist"])
TORQUE_GAIN_FINGER   = float(_t["gain_finger"])
TORQUE_MIN_SCALE     = float(_t["min_scale"])
TORQUE_MAX_SCALE     = float(_t["max_scale"])

RING_OFFSET_ALONG_AXIS_BY_JOINT: dict[str, float] = {
    k: float(v) for k, v in _t.get("ring_offset_along_axis", {}).items()
}


def _ring_offset_for(joint_name: str) -> float:
    """Lookup offset. Missing joints default to 0.0."""
    return float(RING_OFFSET_ALONG_AXIS_BY_JOINT.get(joint_name, 0.0))


# ---------------------------------------------------------------------------
# Force 화살표 스케일
# ---------------------------------------------------------------------------
_f = _cfg["force"]
FORCE_GAIN      = float(_f["gain"])
FORCE_MIN_SCALE = float(_f["min_scale"])
FORCE_MAX_SCALE = float(_f["max_scale"])
FORCE_VEC_HIDE_BELOW = float(_f["hide_below"])


# ---------------------------------------------------------------------------
# Prim 이름 상수
# ---------------------------------------------------------------------------
_p = _cfg["prim_names"]
FORCE_VIZ_REAL_NAME       = _p["force_viz_real"]
FORCE_VIZ_SIM_NAME        = _p["force_viz_sim"]
TORQUE_RING_REAL_NAME     = _p["torque_ring_real"]
TORQUE_RING_SIM_NAME      = _p["torque_ring_sim"]
FORCE_VIZ_VISUAL_SUBPATH  = _p["force_viz_visual_subpath"]


# ---------------------------------------------------------------------------
# 에셋 경로 (EXT_ROOT 기준 상대 → 절대)
# ---------------------------------------------------------------------------
_a = _cfg["assets"]
TORQUE_VIZ_USD_PATH       = os.path.join(EXT_ROOT, *_a["torque_viz_usd"].split("/"))
FORCE_VIZ_USD_PATH        = os.path.join(EXT_ROOT, *_a["force_viz_usd"].split("/"))
FORCE_VIZ_CUSTOM_USD_PATH = os.path.join(EXT_ROOT, *_a["force_viz_custom_usd"].split("/"))


# ---------------------------------------------------------------------------
# force_vec.usda 내부 구조
# ---------------------------------------------------------------------------
_fv = _cfg["force_vec"]
FORCE_VEC_SHAFT_NAME            = _fv["shaft_name"]
FORCE_VEC_HEAD_NAME             = _fv["head_name"]
FORCE_VEC_SHAFT_BASE_LENGTH     = float(_fv["shaft_base_length"])
FORCE_VEC_HEAD_BASE_TRANSLATE_Z = float(_fv["head_base_translate_z"])
FORCE_VEC_LENGTH_AXIS           = int(_fv["length_axis"])


# ---------------------------------------------------------------------------
# 손가락 / 팔 ring 테이블 — JSON 의 spec 을 양 사이드(L/R) 로 펼침
# ---------------------------------------------------------------------------
def _hand_side_entries(side: str) -> list[dict]:
    prefix = f"{side}_"
    out: list[dict] = []
    for joint_stem, link_name, abbr_suffix in _cfg["hand_joint_specs"]["specs"]:
        joint_name = f"{prefix}{joint_stem}_Joint"
        out.append({
            "usd_joint_name":          joint_name,
            "child_link_path":         f"/ALLEX/{prefix}{link_name}",
            "dof_abbr":                f"{side}{abbr_suffix}",
            "base_scale":              (1, 1, 1),
            "ring_offset_along_axis":  _ring_offset_for(joint_name),
        })
    return out


def _arm_side_entries(side: str) -> list[dict]:
    prefix = f"{side}_"
    out: list[dict] = []
    for joint_stem, link_name, abbr_suffix in _cfg["arm_joint_specs"]["specs"]:
        full_joint_name = f"{prefix}{joint_stem}"
        out.append({
            "usd_joint_name":          full_joint_name,
            "child_link_path":         f"/ALLEX/{prefix}{link_name}",
            "dof_abbr":                f"{side}{abbr_suffix}",
            "base_scale":              (1, 1, 1),
            "ring_offset_along_axis":  _ring_offset_for(full_joint_name),
        })
    return out


HAND_JOINT_TORQUE_RING_MAP = (
    _hand_side_entries("L") + _hand_side_entries("R")
    + _arm_side_entries("L") + _arm_side_entries("R")
)


# 핫 패스용 평탄화 튜플 리스트.
HAND_JOINT_TORQUE_RING_TUPLES = tuple(
    (e["dof_abbr"], e["usd_joint_name"], e["child_link_path"])
    for e in HAND_JOINT_TORQUE_RING_MAP
)


# ---------------------------------------------------------------------------
# Force_viz parent links (좌우 Palm + 5 finger distal × 2 = 12)
# ---------------------------------------------------------------------------
def _force_entries() -> list[dict]:
    fp = _cfg["force_viz_parent"]
    finger_to_joint_stem: dict[str, str] = dict(fp["finger_to_joint_stem"])
    orient_table: dict[str, list[float]] = fp["orient_euler_deg"]

    def _orient_for(finger: str):
        v = orient_table.get(finger, orient_table.get("Palm", [0.0, 90.0, 0.0]))
        return (float(v[0]), float(v[1]), float(v[2]))

    out: list[dict] = []
    for side in ("L", "R"):
        out.append({
            "link_path":         f"/ALLEX/{side}_Palm_Link",
            "usd_joint_name":    f"{side}_Wrist_Pitch_Joint",
            "side":              side,
            "finger":            "Palm",
            "orient_euler_deg":  _orient_for("Palm"),
        })
        for finger, joint_stem in finger_to_joint_stem.items():
            out.append({
                "link_path":        f"/ALLEX/{side}_{finger}_Distal_Link",
                "usd_joint_name":   f"{side}_{joint_stem}_Joint",
                "side":             side,
                "finger":           finger,
                "orient_euler_deg": _orient_for(finger),
            })
    return out


FORCE_VIZ_PARENT_LINKS = _force_entries()
