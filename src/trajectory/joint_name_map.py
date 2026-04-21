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
