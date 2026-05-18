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
# hand_joint_specs / arm_joint_specs 는 robot kinematic topology + ROS abbr
# 매핑이라 showcase 무관. viz_config_*.json 변형들 사이에서 중복되지 않도록
# 별도 joint_specs.json 으로 분리.
_JOINT_SPECS_PATH = os.path.join(_CONFIG_DIR, "joint_specs.json")
EXT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "..", "..", ".."))

with open(_JSON_PATH, "r", encoding="utf-8") as _f:
    _cfg = json.load(_f)
with open(_JOINT_SPECS_PATH, "r", encoding="utf-8") as _f:
    _joint_specs = json.load(_f)


def _tup3(v) -> tuple:
    return (float(v[0]), float(v[1]), float(v[2]))


# ---------------------------------------------------------------------------
# 색상
# ---------------------------------------------------------------------------
REAL_COLOR = _tup3(_cfg["colors"]["real"])
# SIM_COLOR = (0.0, 0.8, 0.0)  # green override — force vector + torque ring 둘 다 적용
SIM_COLOR = _tup3(_cfg["colors"]["sim"])

# Trail (force origin trajectory BasisCurves) 전용 색상. 없으면 force 색 fallback.
TRAIL_REAL_COLOR = _tup3(_cfg["colors"].get("trail_real", _cfg["colors"]["real"]))
TRAIL_SIM_COLOR = _tup3(_cfg["colors"].get("trail_sim", _cfg["colors"]["sim"]))


# ---------------------------------------------------------------------------
# Torque 링 스케일
#
# group gain 들은 단일 float (양쪽 공통) 또는 ``{"L": float, "R": float}`` 형태로
# JSON 에 작성 가능. 내부적으론 항상 dict ``{"L": ..., "R": ...}`` 로 normalize.
# ``_torque_gain_for(abbr)`` 가 dof_abbr 의 side prefix (L/R) 보고 골라 줌.
# ---------------------------------------------------------------------------
_t = _cfg["torque"]


def _normalize_side_gain(raw, label: str) -> dict:
    """float → {'L': f, 'R': f}; dict → {'L': L, 'R': R} (대소문자 무관). 실패 시 0.0."""
    if isinstance(raw, dict):
        try:
            l = float(raw.get("L", raw.get("l", 0.0)))
            r = float(raw.get("R", raw.get("r", 0.0)))
            return {"L": l, "R": r}
        except (TypeError, ValueError):
            pass
    try:
        f = float(raw)
        return {"L": f, "R": f}
    except (TypeError, ValueError):
        print(f"[viz_config] {label}: invalid value {raw!r}, fallback to 0.0")
        return {"L": 0.0, "R": 0.0}


TORQUE_GAIN          = float(_t["gain_default"])
TORQUE_GAIN_SHOULDER = _normalize_side_gain(_t["gain_shoulder"], "gain_shoulder")
TORQUE_GAIN_ELBOW    = _normalize_side_gain(_t["gain_elbow"],    "gain_elbow")
TORQUE_GAIN_WRIST    = _normalize_side_gain(_t["gain_wrist"],    "gain_wrist")
TORQUE_GAIN_FINGER   = _normalize_side_gain(_t["gain_finger"],   "gain_finger")
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
# Ext-torque 화살표 스케일 (Wrench 의 torque 시각화 전용)
# JSON 에 ext_torque 섹션 없으면 force 와 동일값으로 fallback.
# ---------------------------------------------------------------------------
def _tup3_or_none(v):
    if v is None:
        return None
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except (TypeError, ValueError, IndexError):
        return None


_et = _cfg.get("ext_torque", {})
EXT_TORQUE_GAIN       = float(_et.get("gain",       FORCE_GAIN))
EXT_TORQUE_MIN_SCALE  = float(_et.get("min_scale",  FORCE_MIN_SCALE))
EXT_TORQUE_MAX_SCALE  = float(_et.get("max_scale",  FORCE_MAX_SCALE))
EXT_TORQUE_HIDE_BELOW = float(_et.get("hide_below", FORCE_VEC_HIDE_BELOW))
# None 이면 visualizer 가 source color (real/sim) 로 fallback.
EXT_TORQUE_COLOR_REAL = _tup3_or_none(_et.get("color_real"))
EXT_TORQUE_COLOR_SIM  = _tup3_or_none(_et.get("color_sim"))


# ---------------------------------------------------------------------------
# Per-joint external torque ring scaling.
# `/result/<group>/joint/ext_torque` 의 값을 ring 에 substitute 할 때 기존
# per_joint_torque_gain 위에 곱해지는 보조 스케일. 별도 prim 안 만들고 기존
# ring 재사용이라 색은 source(real) 색 그대로 — 별도 색 변수 없음.
# ---------------------------------------------------------------------------
_ejt = _cfg.get("ext_joint_torque", {})
EXT_JOINT_TORQUE_GAIN_MULT = float(_ejt.get("gain_multiplier", 1.0))


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
# Torque plot (DataPlotter rolling window / y-axis)
# ---------------------------------------------------------------------------
_tp = _cfg.get("torque_plot", {})


def _parse_y_lim(v, label: str):
    """null → None (autoscale). [ymin, ymax] (ymin<ymax) → (float, float). 그 외 → None."""
    if v is None:
        return None
    try:
        a = float(v[0])
        b = float(v[1])
    except (TypeError, ValueError, IndexError, KeyError):
        print(f"[viz_config] {label}: invalid value {v!r}, fallback to autoscale")
        return None
    if not (a < b):
        print(f"[viz_config] {label}: ymin>=ymax in {v!r}, fallback to autoscale")
        return None
    return (a, b)


TORQUE_PLOT_WINDOW_SECONDS = float(_tp.get("window_seconds", 20.0))
TORQUE_PLOT_Y_LIM_BODY = _parse_y_lim(_tp.get("y_lim_body"), "torque_plot.y_lim_body")
TORQUE_PLOT_Y_LIM_HAND = _parse_y_lim(_tp.get("y_lim_hand"), "torque_plot.y_lim_hand")


def _parse_subsets(raw) -> dict:
    """``{subset_name: [{"name": str, "joints": [str, ...], "y_lim"?: [ymin, ymax]}, ...]}`` 검증.

    각 group 은 한 subplot 단위. 잘못된 entry 는 silently drop. subset 자체가
    비어있거나 형식이 깨졌으면 그 subset 은 key 자체가 안 생기고,
    ``DataPlotter._build_groups`` 가 regex 기반 fallback 으로 동작.

    group.y_lim 은 optional — 있으면 그 subplot 만 fixed y-axis, 없으면 subset
    레벨 ``y_lim_<subset>`` (있으면) fallback, 그것도 없으면 autoscale.

    group.channel 은 optional — ``"torque"`` (default, N m) 또는 ``"pos"`` (deg).
    잘못된 값이면 ``"torque"`` 로 fallback.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for subset_name, groups in raw.items():
        if not isinstance(groups, list):
            print(f"[viz_config] torque_plot.subsets.{subset_name}: not a list, skip")
            continue
        clean: list[dict] = []
        for g in groups:
            if not isinstance(g, dict):
                continue
            name = g.get("name")
            joints = g.get("joints")
            if not isinstance(name, str) or not isinstance(joints, list):
                continue
            jl = [str(j) for j in joints if isinstance(j, str) and j]
            if not jl:
                continue
            ch_raw = g.get("channel", "torque")
            ch = ch_raw if ch_raw in ("torque", "pos") else "torque"
            entry: dict = {"name": name, "joints": jl, "channel": ch}
            if "y_lim" in g:
                yl = _parse_y_lim(
                    g["y_lim"],
                    f"torque_plot.subsets.{subset_name}.{name}.y_lim",
                )
                if yl is not None:
                    entry["y_lim"] = [yl[0], yl[1]]
            if "highlights" in g and isinstance(g["highlights"], list):
                parsed_highlights: list[dict] = []
                for h in g["highlights"]:
                    if not isinstance(h, dict):
                        continue
                    t0 = h.get("t0")
                    t1 = h.get("t1")
                    if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
                        continue
                    if float(t1) <= float(t0):
                        continue
                    entry_h: dict = {
                        "t0": float(t0),
                        "t1": float(t1),
                        "color": str(h.get("color", "#ff8c00")),
                        "alpha": max(0.0, min(1.0, float(h.get("alpha", 0.15)))),
                    }
                    if "edgecolor" in h:
                        entry_h["edgecolor"] = str(h["edgecolor"])
                    if "linewidth" in h:
                        entry_h["linewidth"] = float(h["linewidth"])
                    parsed_highlights.append(entry_h)
                if parsed_highlights:
                    entry["highlights"] = parsed_highlights
            clean.append(entry)
        if clean:
            out[str(subset_name)] = clean
    return out


TORQUE_PLOT_SUBSETS = _parse_subsets(_tp.get("subsets"))


# ---------------------------------------------------------------------------
# 손가락 / 팔 ring 테이블 — joint_specs.json 의 spec 을 양 사이드(L/R) 로 펼침
# ---------------------------------------------------------------------------
def _hand_side_entries(side: str) -> list[dict]:
    prefix = f"{side}_"
    out: list[dict] = []
    for joint_stem, link_name, abbr_suffix in _joint_specs["hand_joint_specs"]["specs"]:
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
    for joint_stem, link_name, abbr_suffix in _joint_specs["arm_joint_specs"]["specs"]:
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
