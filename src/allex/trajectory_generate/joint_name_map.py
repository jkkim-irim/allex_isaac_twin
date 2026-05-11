"""CSV filename (without .csv) → ordered list of ALLEX joint names.

Each CSV contains columns `duration, joint_1, ..., joint_N`; the k-th joint
column maps to the k-th entry in the list below. Joint names follow the MJCF
(and USD) convention in `asset/ALLEX/mjcf/ALLEX.xml` and are resolved against
`articulation.dof_names` at runtime.

Follower joints governed by Newton equality constraints (e.g. L_Thumb_IP_Joint,
*_DIP_Joint, Waist_Upper_Pitch_Joint, Waist_Pitch_Dummy_Joint) are intentionally
NOT listed here — they're driven by their master joints, not the CSV.
"""
from __future__ import annotations


ALLEX_CSV_JOINT_NAMES: dict[str, list[str]] = {
    "theOne_waist": ["Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint"],
    "theOne_neck": ["Neck_Pitch_Joint", "Neck_Yaw_Joint"],
    "Arm_L_theOne": [
        "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
        "L_Elbow_Joint",
        "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    ],
    "Arm_R_theOne": [
        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
        "R_Elbow_Joint",
        "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    ],
    "Hand_L_thumb_wir":  ["L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint"],
    "Hand_L_index_wir":  ["L_Index_ABAD_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint"],
    "Hand_L_middle_wir": ["L_Middle_ABAD_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint"],
    "Hand_L_ring_wir":   ["L_Ring_ABAD_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint"],
    "Hand_L_little_wir": ["L_Little_ABAD_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint"],
    "Hand_R_thumb_wir":  ["R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint"],
    "Hand_R_index_wir":  ["R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint"],
    "Hand_R_middle_wir": ["R_Middle_ABAD_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint"],
    "Hand_R_ring_wir":   ["R_Ring_ABAD_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint"],
    "Hand_R_little_wir": ["R_Little_ABAD_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint"],
}


# ext_torque 토픽 (`/result/<group>/joint/ext_torque`) 의 element 순서.
#
# ALLEX_CSV_JOINT_NAMES 와 다른 점:
#   - finger/thumb: follower joint (DIP / IP) 가 끝에 포함 → 4 elements
#   - neck: 순서가 (Yaw, Pitch) 로 ALLEX_CSV_JOINT_NAMES (Pitch, Yaw) 와 반대
#   - waist / arm: ALLEX_CSV_JOINT_NAMES 와 동일 순서
#
# 사용자 명세 (per element):
#   finger : 0=MCP roll(ABAD), 1=MCP pitch, 2=PIP, 3=DIP
#   thumb  : 0=CMC yaw(Thumb_Yaw), 1=CMC pitch(Thumb_CMC), 2=MCP, 3=IP
#   arm    : 0..6 = SP, SR, SY, EP, WY, WR, WP
#   neck   : 0=Yaw, 1=Pitch
#   waist  : 0=Yaw, 1=Pitch
EXT_TORQUE_JOINT_NAMES: dict[str, list[str]] = {
    "theOne_waist": ["Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint"],
    "theOne_neck":  ["Neck_Yaw_Joint", "Neck_Pitch_Joint"],
    "Arm_L_theOne": [
        "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
        "L_Elbow_Joint",
        "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    ],
    "Arm_R_theOne": [
        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
        "R_Elbow_Joint",
        "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    ],
    "Hand_L_thumb_wir":  ["L_Thumb_Yaw_Joint",  "L_Thumb_CMC_Joint",
                          "L_Thumb_MCP_Joint",  "L_Thumb_IP_Joint"],
    "Hand_L_index_wir":  ["L_Index_ABAD_Joint", "L_Index_MCP_Joint",
                          "L_Index_PIP_Joint",  "L_Index_DIP_Joint"],
    "Hand_L_middle_wir": ["L_Middle_ABAD_Joint", "L_Middle_MCP_Joint",
                          "L_Middle_PIP_Joint",  "L_Middle_DIP_Joint"],
    "Hand_L_ring_wir":   ["L_Ring_ABAD_Joint", "L_Ring_MCP_Joint",
                          "L_Ring_PIP_Joint",  "L_Ring_DIP_Joint"],
    "Hand_L_little_wir": ["L_Little_ABAD_Joint", "L_Little_MCP_Joint",
                          "L_Little_PIP_Joint",  "L_Little_DIP_Joint"],
    "Hand_R_thumb_wir":  ["R_Thumb_Yaw_Joint",  "R_Thumb_CMC_Joint",
                          "R_Thumb_MCP_Joint",  "R_Thumb_IP_Joint"],
    "Hand_R_index_wir":  ["R_Index_ABAD_Joint", "R_Index_MCP_Joint",
                          "R_Index_PIP_Joint",  "R_Index_DIP_Joint"],
    "Hand_R_middle_wir": ["R_Middle_ABAD_Joint", "R_Middle_MCP_Joint",
                          "R_Middle_PIP_Joint",  "R_Middle_DIP_Joint"],
    "Hand_R_ring_wir":   ["R_Ring_ABAD_Joint", "R_Ring_MCP_Joint",
                          "R_Ring_PIP_Joint",  "R_Ring_DIP_Joint"],
    "Hand_R_little_wir": ["R_Little_ABAD_Joint", "R_Little_MCP_Joint",
                          "R_Little_PIP_Joint",  "R_Little_DIP_Joint"],
}


# ---------------------------------------------------------------------------
# Per-joint external torque CSV column key (`ext_joint_torque_<short>`).
#
# Wrench-torque (`ext_torque_<topic_id>_x/y/z`) 와 column prefix 가 명확히 구분되도록
# per-joint scalar 컬럼은 `ext_joint_torque_` 로 시작한다.
#
# 명명 규칙: ``<group_short>_<joint_stem_lower>`` (e.g. hand_l_index_abad).
#   group_short:  group → 짧은 lowercase 식별자
#   joint_stem:   joint_full 에서 side/group prefix 와 "_Joint" suffix 를 떼고 lowercase
#
# 예시:
#   L_Index_ABAD_Joint      (Hand_L_index_wir)  → "hand_l_index_abad"
#   L_Thumb_IP_Joint        (Hand_L_thumb_wir)  → "hand_l_thumb_ip"
#   L_Shoulder_Pitch_Joint  (Arm_L_theOne)      → "arm_l_shoulder_pitch"
#   L_Elbow_Joint           (Arm_L_theOne)      → "arm_l_elbow"
#   Neck_Yaw_Joint          (theOne_neck)       → "neck_yaw"
#   Waist_Lower_Pitch_Joint (theOne_waist)      → "waist_lower_pitch"
# ---------------------------------------------------------------------------
_EXT_TORQUE_GROUP_SHORT: dict[str, str] = {
    "theOne_waist":      "waist",
    "theOne_neck":       "neck",
    "Arm_L_theOne":      "arm_l",
    "Arm_R_theOne":      "arm_r",
    "Hand_L_thumb_wir":  "hand_l_thumb",
    "Hand_L_index_wir":  "hand_l_index",
    "Hand_L_middle_wir": "hand_l_middle",
    "Hand_L_ring_wir":   "hand_l_ring",
    "Hand_L_little_wir": "hand_l_little",
    "Hand_R_thumb_wir":  "hand_r_thumb",
    "Hand_R_index_wir":  "hand_r_index",
    "Hand_R_middle_wir": "hand_r_middle",
    "Hand_R_ring_wir":   "hand_r_ring",
    "Hand_R_little_wir": "hand_r_little",
}

_EXT_TORQUE_GROUP_JOINT_PREFIX: dict[str, str] = {
    "theOne_waist":      "Waist_",
    "theOne_neck":       "Neck_",
    "Arm_L_theOne":      "L_",
    "Arm_R_theOne":      "R_",
    "Hand_L_thumb_wir":  "L_Thumb_",
    "Hand_L_index_wir":  "L_Index_",
    "Hand_L_middle_wir": "L_Middle_",
    "Hand_L_ring_wir":   "L_Ring_",
    "Hand_L_little_wir": "L_Little_",
    "Hand_R_thumb_wir":  "R_Thumb_",
    "Hand_R_index_wir":  "R_Index_",
    "Hand_R_middle_wir": "R_Middle_",
    "Hand_R_ring_wir":   "R_Ring_",
    "Hand_R_little_wir": "R_Little_",
}


def _build_ext_torque_csv_keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for group, joints in EXT_TORQUE_JOINT_NAMES.items():
        gs = _EXT_TORQUE_GROUP_SHORT[group]
        prefix = _EXT_TORQUE_GROUP_JOINT_PREFIX[group]
        for jn in joints:
            stem = jn
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
            if stem.endswith("_Joint"):
                stem = stem[: -len("_Joint")]
            out[jn] = f"{gs}_{stem.lower()}"
    return out


# joint_full_name → CSV column short key (without "ext_joint_torque_" prefix).
EXT_TORQUE_JOINT_CSV_KEYS: dict[str, str] = _build_ext_torque_csv_keys()


# ---------------------------------------------------------------------------
# 일반 CSV 컬럼용 joint 이름 변환 (pos_<short>, torque_<short>).
#
# 규칙: ``joint_full`` 에서 ``_Joint`` suffix 제거 + 전체 lowercase.
#   R_Ring_ABAD_Joint       → r_ring_abad
#   L_Elbow_Joint           → l_elbow
#   L_Shoulder_Pitch_Joint  → l_shoulder_pitch
#   Neck_Yaw_Joint          → neck_yaw
#   Waist_Lower_Pitch_Joint → waist_lower_pitch
#
# 변별력 없는 ``_Joint`` postfix 제거 + 대소문자 통일을 통해 ext_joint_torque_*
# 명명 규칙과 일관성 유지. showcase_logger / rosbag_to_csv (emitter) 와
# showcase_reader (parser) 양쪽에서 사용.
#
# Reverse map (JOINT_CSV_SHORT_TO_FULL) 은 reader 가 short_key → full 으로 복원할 때.
# 알려지지 않은 short 는 fallback 으로 column key 그대로 사용 (구버전 CSV 호환).
# ---------------------------------------------------------------------------
def _build_joint_csv_short_map() -> dict[str, str]:
    all_joints: set[str] = set()
    for joints in ALLEX_CSV_JOINT_NAMES.values():
        all_joints.update(joints)
    for joints in EXT_TORQUE_JOINT_NAMES.values():
        all_joints.update(joints)
    # 위 두 dict 에 없는 추가 follower / dummy joint (joint_config.json 의 equality
    # follower 등). showcase_logger 가 articulation.dof_names 기반이라 이들도 등장.
    all_joints.update([
        "Waist_Upper_Pitch_Joint",
        "Waist_Pitch_Dummy_Joint",
    ])
    out: dict[str, str] = {}
    for jn in all_joints:
        s = jn[: -len("_Joint")] if jn.endswith("_Joint") else jn
        out[jn] = s.lower()
    return out


JOINT_FULL_TO_CSV_SHORT: dict[str, str] = _build_joint_csv_short_map()
JOINT_CSV_SHORT_TO_FULL: dict[str, str] = {v: k for k, v in JOINT_FULL_TO_CSV_SHORT.items()}


def joint_to_csv_short(joint_full: str) -> str:
    """joint_full → short CSV column name (lowercase, no `_Joint`).

    알려지지 않은 joint 도 동일 규칙으로 변환 (mapping 없어도 deterministic).
    """
    cached = JOINT_FULL_TO_CSV_SHORT.get(joint_full)
    if cached is not None:
        return cached
    s = joint_full[: -len("_Joint")] if joint_full.endswith("_Joint") else joint_full
    return s.lower()
