"""
Force / Torque Visualization 설정

- torque 링: 40개 손가락 joint 각각의 child_link 아래에 생성
- force 화살표: 12개 손/손가락 distal + palm link. 기존 `force_viz`는 real 역할,
  런타임에 같은 부모 밑에 `force_viz_sim`을 추가 생성하여 겹침 표현
- 색상: REAL=빨강, SIM=시안계열
"""

import os


# ========================================
# 색상 (RGB, 0~1)
# ========================================
REAL_COLOR = (1.0, 0.0, 0.0)
SIM_COLOR = (0.0, 0.8, 1.0)


# ========================================
# Torque 링 스케일 — 공통 기본값
# ========================================
TORQUE_GAIN = 0.1
TORQUE_MIN_SCALE = 0.05   # 최소 표시 크기 (토크 0일 때). 음수는 clip 오류 유발.
TORQUE_MAX_SCALE = 3.0


# ========================================
# Torque 링 스케일 — 그룹별 gain 오버라이드
#
# dof_abbr 분류:
#   SHOULDER : *SP, *SR, *SY  (어깨 3축)
#   ELBOW    : *EP             (팔꿈치)
#   WRIST    : *WY, *WR, *WP  (손목 3축)
#   FINGER   : L11~L54, R11~R54 (손가락 40개)
#
# gain 을 낮추면 링이 작아지고, 높이면 커진다.
# ========================================
TORQUE_GAIN_SHOULDER = 0.10   # 어깨 — 토크가 커서 기본값보다 줄임
TORQUE_GAIN_ELBOW    = 0.10   # 팔꿈치
TORQUE_GAIN_WRIST    = 0.40   # 손목 — 토크가 작아서 기본값보다 키움
TORQUE_GAIN_FINGER   = 0.40   # 손가락 — 토크가 매우 작음


# ========================================
# Torque ring 위치 offset — **ring 회전축 방향** (+/-) m 단위
#
# 기본 위치는 joint.localPos1 (child link 원점 기준 joint 좌표).
# 여기에 axis 방향으로 추가 offset 을 줘서 ring 을 link 안쪽/바깥쪽으로
# 밀 수 있음. 양수 = 회전축 +방향 (joint axis 방향), 음수 = 반대.
#
# Per-joint override 테이블. dict 에 없는 joint 는 **0.0** (offset 없음).
# 좌우 비대칭 튜닝이 필요하면 L_* / R_* 를 각각 다르게 세팅하면 됨.
# 키: `usd_joint_name` (예: "L_Shoulder_Pitch_Joint", "R_Elbow_Joint",
#                        "L_Thumb_MCP_Joint" …)
# ========================================
RING_OFFSET_ALONG_AXIS_BY_JOINT: dict[str, float] = {
    # --- Shoulder ---
    "L_Shoulder_Pitch_Joint": 0.01,
    "L_Shoulder_Roll_Joint":  -0.035,
    "L_Shoulder_Yaw_Joint":   0.07,
    "R_Shoulder_Pitch_Joint": -0.01,
    "R_Shoulder_Roll_Joint":  -0.035,
    "R_Shoulder_Yaw_Joint":   0.07,
    # --- Elbow ---
    "L_Elbow_Joint": 0.06,
    "R_Elbow_Joint": -0.06,
    # --- Wrist / Finger --- 필요 시 추가 (없으면 0.0 적용)
}


def _ring_offset_for(joint_name: str) -> float:
    """Lookup offset. Missing joints default to 0.0."""
    return float(RING_OFFSET_ALONG_AXIS_BY_JOINT.get(joint_name, 0.0))


# ========================================
# Force 화살표 스케일
# ========================================
FORCE_GAIN = 0.5
FORCE_MIN_SCALE = -3.0
FORCE_MAX_SCALE = 3.0


# ========================================
# Prim 이름 상수
# ========================================
FORCE_VIZ_REAL_NAME = "force_viz_real"   # 새로 생성할 real prim (red)
FORCE_VIZ_SIM_NAME = "force_viz"          # 기존 ALLEX.usd prim 재사용 (sim, cyan)
TORQUE_RING_REAL_NAME = "torque_ring_real"
TORQUE_RING_SIM_NAME = "torque_ring_sim"


# ========================================
# 에셋 경로
# ========================================
# 이 파일: <EXT_ROOT>/src/config/viz_config.py
# 확장 루트: <EXT_ROOT>
# torque_viz.usd / force_viz.usd 위치: <EXT_ROOT>/asset/utils/
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "..", ".."))
TORQUE_VIZ_USD_PATH = os.path.join(EXT_ROOT, "asset", "utils", "torque_viz.usd")
# force 화살표 asset 교체: 정적 force_viz.usd → 길이 가변 force_vec.usda (Arrow/Shaft + Arrow/Head)
FORCE_VIZ_USD_PATH = os.path.join(EXT_ROOT, "asset", "utils", "force_vec.usda")


# ========================================
# force_vec.usda 내부 구조
# (_create_ref_xform 이 레퍼런스 prim 을 만들면 내부에 Shaft / Head Mesh 가 자식으로 존재)
# ========================================
FORCE_VEC_SHAFT_NAME = "Shaft"
FORCE_VEC_HEAD_NAME = "Head"
# force_vec.usda 원본 Shaft 길이 (Z=0..0.005)
FORCE_VEC_SHAFT_BASE_LENGTH = 0.005
# Head 원본 translate Z (Head 는 본래 Z=0.005 부터 시작하는 메쉬)
FORCE_VEC_HEAD_BASE_TRANSLATE_Z = 0.0
# force 길이 스케일을 적용할 축 (force_vec 의 shaft 축은 +Z)
FORCE_VEC_LENGTH_AXIS = 2  # 0=X, 1=Y, 2=Z


# ========================================
# 손가락 40 joint 테이블
#
# 공통 필드:
#   usd_joint_name       : USD 상의 PhysicsRevoluteJoint prim 이름
#   child_link_path      : 토크 링이 attach 될 body link 경로
#   dof_abbr             : joint_config.json 약어 (L11, R54 …)
#   axis_local           : joint 축 (로컬). 전부 +X
#   translate_offset     : 링 Xform local translate offset
#   rotate_offset_euler_deg : 링 Xform local rotate offset (XYZ deg)
#   base_scale           : 기본 scale (토크 0일 때 가시성)
# ========================================
def _hand_side_entries(side):
    """side = "L" or "R" — 20 entries."""
    prefix = f"{side}_"
    abbr_prefix = side  # "L11" ...
    # (joint_name_stem, link_name, abbr_suffix)
    specs = [
        ("Thumb_Yaw",   "Thumb_Yaw_Link",       "11"),
        ("Thumb_CMC",   "Thumb_Proximal_Link",  "12"),
        ("Thumb_MCP",   "Thumb_Middle_Link",    "13"),
        ("Thumb_IP",    "Thumb_Distal_Link",    "14"),  # mimic slave
        ("Index_ABAD",  "Index_Roll_Link",      "21"),
        ("Index_MCP",   "Index_Proximal_Link",  "22"),
        ("Index_PIP",   "Index_Middle_Link",    "23"),
        ("Index_DIP",   "Index_Distal_Link",    "24"),  # mimic
        ("Middle_ABAD", "Middle_Roll_Link",     "31"),
        ("Middle_MCP",  "Middle_Proximal_Link", "32"),
        ("Middle_PIP",  "Middle_Middle_Link",   "33"),
        ("Middle_DIP",  "Middle_Distal_Link",   "34"),  # mimic
        ("Ring_ABAD",   "Ring_Roll_Link",       "41"),
        ("Ring_MCP",    "Ring_Proximal_Link",   "42"),
        ("Ring_PIP",    "Ring_Middle_Link",     "43"),
        ("Ring_DIP",    "Ring_Distal_Link",     "44"),  # mimic
        ("Little_ABAD", "Little_Roll_Link",     "51"),
        ("Little_MCP",  "Little_Proximal_Link", "52"),
        ("Little_PIP",  "Little_Middle_Link",   "53"),
        ("Little_DIP",  "Little_Distal_Link",   "54"),  # mimic
    ]
    out = []
    for joint_stem, link_name, abbr_suffix in specs:
        joint_name = f"{prefix}{joint_stem}_Joint"
        out.append({
            "usd_joint_name": joint_name,
            "child_link_path": f"/ALLEX/{prefix}{link_name}",
            "dof_abbr": f"{abbr_prefix}{abbr_suffix}",
            "base_scale": (1, 1, 1),
            "ring_offset_along_axis": _ring_offset_for(joint_name),
        })
    return out


def _arm_side_entries(side):
    """side = "L" or "R" — 3 shoulder + elbow + 3 wrist joints. dof_abbr는 ros2_config와 일치."""
    prefix = f"{side}_"
    # (joint_name, link_name, dof_abbr)
    specs = [
        ("Shoulder_Pitch_Joint", "Shoulder_Pitch_Link", f"{side}SP"),
        ("Shoulder_Roll_Joint",  "Shoulder_Roll_Link",  f"{side}SR"),
        ("Shoulder_Yaw_Joint",   "Shoulder_Yaw_Link",   f"{side}SY"),
        ("Elbow_Joint",          "Elbow_Link",          f"{side}EP"),
        ("Wrist_Yaw_Joint",      "Wrist_Yaw_Link",      f"{side}WY"),
        ("Wrist_Roll_Joint",     "Wrist_Roll_Link",     f"{side}WR"),
        ("Wrist_Pitch_Joint",    "Wrist_Pitch_Link",    f"{side}WP"),
    ]
    out = []
    for joint_name, link_name, dof_abbr in specs:
        full_joint_name = f"{prefix}{joint_name}"
        out.append({
            "usd_joint_name": full_joint_name,
            "child_link_path": f"/ALLEX/{prefix}{link_name}",
            "dof_abbr": dof_abbr,
            "base_scale": (1, 1, 1),
            "ring_offset_along_axis": _ring_offset_for(full_joint_name),
        })
    return out


HAND_JOINT_TORQUE_RING_MAP = (
    _hand_side_entries("L") + _hand_side_entries("R")
    + _arm_side_entries("L") + _arm_side_entries("R")
)


# 핫 패스용 평탄화 튜플 리스트: (dof_abbr, usd_joint_name, child_link_path)
# visualizer.update() 매 step 호출되므로 dict lookup 비용 제거용.
HAND_JOINT_TORQUE_RING_TUPLES = tuple(
    (e["dof_abbr"], e["usd_joint_name"], e["child_link_path"])
    for e in HAND_JOINT_TORQUE_RING_MAP
)


# ========================================
# Force_viz 12개 부모 link 테이블
# ========================================
def _force_entries():
    """Force arrow 부모 link 정보. joint_name 은 arrow 의 axis 정렬용.

    orient_euler_deg (XYZ intrinsic deg):
      force_vec.usda 화살표는 local +Z 방향. 각 링크 local frame 기준
      올바른 force 방향에 맞추기 위한 회전.
      - Thumb distal: local -Z 방향 → X축 180° → (180, 0, 0)
      - 그 외 (Palm, Index/Middle/Ring/Little distal): local +X 방향
        → Y축 +90° → (0, 90, 0)
    """
    finger_to_joint_stem = {
        "Thumb":  "Thumb_IP",
        "Index":  "Index_DIP",
        "Middle": "Middle_DIP",
        "Ring":   "Ring_DIP",
        "Little": "Little_DIP",
    }

    def _orient_for(finger):
        if finger == "Thumb":
            return (180.0, 0.0, 0.0)
        # Palm + Index/Middle/Ring/Little
        return (0.0, 90.0, 0.0)

    out = []
    for side in ("L", "R"):
        # Palm: wrist pitch joint axis 로 정렬
        out.append({
            "link_path": f"/ALLEX/{side}_Palm_Link",
            "usd_joint_name": f"{side}_Wrist_Pitch_Joint",
            "side": side,
            "finger": "Palm",
            "orient_euler_deg": _orient_for("Palm"),
        })
        for finger, joint_stem in finger_to_joint_stem.items():
            out.append({
                "link_path": f"/ALLEX/{side}_{finger}_Distal_Link",
                "usd_joint_name": f"{side}_{joint_stem}_Joint",
                "side": side,
                "finger": finger,
                "orient_euler_deg": _orient_for(finger),
            })
    return out


FORCE_VIZ_PARENT_LINKS = _force_entries()


# Force_viz 기존 prim 의 서브 경로 (`/visuals/force_viz`)
FORCE_VIZ_VISUAL_SUBPATH = "visuals"
