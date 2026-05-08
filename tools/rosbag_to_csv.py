"""
rosbag2 → CSV 변환기. 두 가지 출력 포맷 지원.

기본 output dir = source rosbag 의 parent (즉 source 와 같은 디렉토리).
--output 으로 override 가능.

[--format trajstudio] (default)
  Extension 의 TrajStudio 가 읽는 14-group CSV 포맷.
  - 재생용 (joint_ang_target_deg) 은 루트에, 부가 데이터 (torque, velocity 등) 는
    sub-folder 에 저장.
  Output:
    <bag_parent>/
      Arm_L_theOne.csv          ← 재생용 (joint_ang_target_deg)
      ...
      _current/  _torque/  _velocity/  _torque_target/  ← 분석용 sub-folder

[--format showcase]
  Real/Sim Replay 패널이 읽는 단일 full-robot CSV (showcase_logger 와 동일 포맷).
  - 행 = 시점 (rosbag 절대 timestamp 기준 상대시간)
  - 컬럼: time, pos_<joint_full_name> [rad], torque_<joint_full_name> [Nm]
  - Position 출처: joint_positions_deg (current = measured, deg→rad)
  - Torque 출처:   /result/<group>/joint/torque  (필터된 joint torque — outbound 대신)
  - 14 그룹 토픽을 union 시간축으로 합치고, 각 시점에서 해당 토픽의 최신 값 사용.
  - Contact / force_vec 컬럼은 rosbag 에 그 토픽이 없으면 자동 생략.
  Output:
    <output_dir>/real_<bag_name>.csv
    (--output 미지정 시 <bag_parent>/real_<bag_name>.csv)

[--time-offset SEC] (모든 포맷 공통)
  + : 첫 sample 값을 그만큼 앞에 복제해서 패딩 (전체 길이 +offset).
  - : 앞에서 |offset| 초 만큼 trim (전체 길이 -|offset|).

Usage:
  python3 tools/rosbag_to_csv.py <bag_dir>                               # trajstudio (default)
  python3 tools/rosbag_to_csv.py <bag_dir> --format showcase             # showcase real_*.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float64MultiArray

# joint_name_map.py 만 직접 로드 (allex/__init__.py 를 우회 — isaacsim 의존성 회피).
import importlib.util as _ilu
_REPO_ROOT = Path(__file__).resolve().parent.parent
_jnm_path = _REPO_ROOT / "src" / "allex" / "trajectory_generate" / "joint_name_map.py"
_spec = _ilu.spec_from_file_location("joint_name_map", _jnm_path)
_jnm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_jnm)
ALLEX_CSV_JOINT_NAMES = _jnm.ALLEX_CSV_JOINT_NAMES


# TrajStudio 가 스캔하는 14 개 그룹 (src/trajectory/joint_name_map.py 와 일치)
CSV_GROUPS = (
    "Arm_L_theOne",
    "Arm_R_theOne",
    "Hand_L_thumb_wir", "Hand_L_index_wir", "Hand_L_middle_wir",
    "Hand_L_ring_wir", "Hand_L_little_wir",
    "Hand_R_thumb_wir", "Hand_R_index_wir", "Hand_R_middle_wir",
    "Hand_R_ring_wir", "Hand_R_little_wir",
    "theOne_waist",
    "theOne_neck",
)

# 변환 대상 topic suffix → 저장 sub-folder ("" = 루트 = 재생용)
# 재생은 joint_ang_target_deg(desired) 기본. 나머지는 추후 분석/비교용.
TOPIC_TO_SUBDIR = {
    "joint_ang_target_deg":   "",                 # 재생 대상
    "joint_positions_deg":    "_current",         # 실측 위치
    "joint_torque":           "_torque",          # 실측 토크
    "joint_velocities_deg_s": "_velocity",        # 실측 속도
    "joint_trq_target":       "_torque_target",   # 목표 토크
}
# 변환 스킵 (재생·분석 모두 불필요): joint_output_deg, motor_*, servo_status,
# control_mode, now_traj_file, traj_status, articulation_now, poses,
# communication_code, 그리고 /rosout, /parameter_events 등 전역 topic

# ──────────────────────────────────────────────────────────────────────────────
# External force / contact_pos 토픽 (showcase 모드 전용 — Real/Sim Replay overlay)
#
# 모든 ext_force / position 데이터는 **Chest_Origin_Link frame** 으로 publish 된다.
# CSV 에는 chest frame 그대로 저장하고, runtime 에 CsvReplayer 가 sim 의
# Chest_Origin_Link USD prim 의 world matrix 를 Fabric API 로 query 해서 변환한다.
# (waist joint 에 따라 chest 위치가 변하므로 정적 offset 근사 불가 — pitch 4-bar
#  linkage 의 평행이동까지 잡으려면 runtime sim transform 이 정답.)
#
# 컬럼명은 토픽 출처를 그대로 반영하는 단순한 id 로 통일. CSV 에 추가된 모든 ext_force
# topic 은 replay 시 ``ext_force_<topic_id>`` 키로 force_trigger.json 에서 게이팅 가능
# (csv_replayer.py::_push_real_force_routed 참조). routing dict 와 무관하게 임의 추가 OK.
#
# topic id 매핑:
#   /result/Arm_L_theOne/ee/ext_force         → arm_l           (L_Palm 외력 / contact_pos)
#   /result/Arm_R_theOne/ee/ext_force         → arm_r           (R_Palm 외력 / contact_pos)
#   /result/frames/left_shoulder/ee/ext_force → shoulder_l      (3rd-law mirror)
#   /result/frames/right_shoulder/ee/ext_force→ shoulder_r      (3rd-law mirror)
#   /result/Hand_L_<finger>_wir/ee/ext_force  → hand_l_<finger> (손가락별 외력 + 접촉위치)
#   /result/Hand_R_<finger>_wir/ee/ext_force  → hand_r_<finger>
#     finger ∈ {thumb, index, middle, ring, little}
EXT_FORCE_TOPICS: dict[str, str] = {
    "/result/Arm_L_theOne/ee/ext_force":          "arm_l",
    "/result/Arm_R_theOne/ee/ext_force":          "arm_r",
    "/result/frames/left_shoulder/ee/ext_force":  "shoulder_l",
    "/result/frames/right_shoulder/ee/ext_force": "shoulder_r",
}

EXT_CONTACT_POS_TOPICS: dict[str, str] = {
    "/result/Arm_L_theOne/ee/position": "arm_l",
    "/result/Arm_R_theOne/ee/position": "arm_r",
}

# 손가락별 ext_force / position — Hand_<L|R>_<finger>_wir 그룹 (5 finger × 2 hand = 10 ea).
# topic_id = hand_<l|r>_<finger> → CSV 컬럼: ext_force_hand_l_thumb_x ... etc.
# trigger.json 에서 "ext_force_hand_l_thumb": [[t_on, t_off]] 로 visualize 게이팅.
for _hand_up, _hand_lo in (("L", "l"), ("R", "r")):
    for _finger in ("thumb", "index", "middle", "ring", "little"):
        _tid = f"hand_{_hand_lo}_{_finger}"
        EXT_FORCE_TOPICS[f"/result/Hand_{_hand_up}_{_finger}_wir/ee/ext_force"] = _tid
        EXT_CONTACT_POS_TOPICS[f"/result/Hand_{_hand_up}_{_finger}_wir/ee/position"] = _tid
del _hand_up, _hand_lo, _finger, _tid


def _parse_bag(input_dir: str):
    """bag 을 읽어 {topic_name: [(t_ns, [values])...] } 리턴."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # 관심 topic 필터:
    #
    # 1) /robot_outbound_data/<group>/<suffix>  — pos / vel / target 등.
    # 2) /result/<group>/joint/torque           — joint_torque 의 **필터된** 버전.
    #    /robot_outbound_data/<group>/joint_torque (raw motor cmd 잔차 포함) 대신
    #    이쪽을 쓴다. 동일 group / 동일 dispatch 경로 (suffix="joint_torque") 로
    #    내부 처리 → showcase CSV 의 torque_* 컬럼, _torque/<group>.csv 둘 다
    #    필터된 데이터를 사용하게 됨.
    # 3) ext_force / contact_pos — 위 dict 매핑 그대로.
    keep = []
    topic_meta = {}  # topic → (group, suffix)
    for name in type_map:
        parts = name.strip("/").split("/")
        # /result/<group>/joint/torque (필터된 joint torque)
        if (len(parts) == 4 and parts[0] == "result"
                and parts[2] == "joint" and parts[3] == "torque"):
            group = parts[1]
            if group not in CSV_GROUPS:
                continue
            keep.append(name)
            topic_meta[name] = (group, "joint_torque")
            continue
        # /robot_outbound_data/<group>/<suffix>  (단 joint_torque 는 위에서 처리)
        if len(parts) == 3 and parts[0] == "robot_outbound_data":
            group, suffix = parts[1], parts[2]
            if group not in CSV_GROUPS:
                continue
            if suffix == "joint_torque":
                # raw torque skip — 필터된 /result/<group>/joint/torque 만 쓴다.
                continue
            if suffix not in TOPIC_TO_SUBDIR:
                continue
            keep.append(name)
            topic_meta[name] = (group, suffix)
            continue
        # ext_force 토픽 — topic_id 가 그대로 CSV 컬럼 suffix.
        if name in EXT_FORCE_TOPICS:
            keep.append(name)
            topic_meta[name] = ("_ext_force", EXT_FORCE_TOPICS[name])
            continue
        # contact_pos 토픽 — 1:1 매핑 (각 팔의 EE position).
        if name in EXT_CONTACT_POS_TOPICS:
            keep.append(name)
            topic_meta[name] = ("_contact_pos", EXT_CONTACT_POS_TOPICS[name])
            continue

    if not keep:
        print("[ERROR] 대상 topic 을 찾지 못했어요. bag 경로를 확인하세요.")
        return {}, {}

    reader.set_filter(rosbag2_py.StorageFilter(topics=keep))

    # Per-topic deserializer cache. ext_force 토픽은 Vector3 / Wrench / Float64MultiArray
    # 중 하나일 수 있어서 type_map 기반으로 분기.
    def _deserializer_for(topic_name: str):
        type_str = type_map.get(topic_name, "")
        if type_str.endswith("Float64MultiArray"):
            return ("Float64MultiArray",)
        if "Vector3" in type_str:
            return ("Vector3",)
        if "Wrench" in type_str:
            # WrenchStamped vs Wrench 둘 다 force.x/y/z 가짐 (Stamped 는 wrench.force.x 경로).
            return ("WrenchStamped" if "Stamped" in type_str else "Wrench",)
        # 미상 타입 — 첫 시도는 Float64MultiArray 로 fallback.
        return ("Float64MultiArray",)

    data = {t: [] for t in keep}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        kind = _deserializer_for(topic)[0]
        try:
            if kind == "Float64MultiArray":
                msg = deserialize_message(raw, Float64MultiArray)
                vals = list(msg.data)
            elif kind == "Vector3":
                from geometry_msgs.msg import Vector3
                msg = deserialize_message(raw, Vector3)
                vals = [msg.x, msg.y, msg.z]
            elif kind == "Wrench":
                from geometry_msgs.msg import Wrench
                msg = deserialize_message(raw, Wrench)
                vals = [msg.force.x, msg.force.y, msg.force.z,
                        msg.torque.x, msg.torque.y, msg.torque.z]
            elif kind == "WrenchStamped":
                from geometry_msgs.msg import WrenchStamped
                msg = deserialize_message(raw, WrenchStamped)
                vals = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
            else:
                continue
        except Exception as exc:
            print(f"[WARN] deserialize 실패 ({topic}, kind={kind}): {exc}")
            continue
        data[topic].append((t_ns, vals))

    for t in data:
        data[t].sort(key=lambda x: x[0])

    return data, topic_meta


def _downsample(samples, target_hz: float):
    """시간 간격 >= 1/hz 인 샘플만 남김. hz <= 0 이면 원본 그대로.

    주의: rosbag timestamp 에는 DDS/recorder jitter 가 있어 실제 간격이
    1/hz 보다 미세하게 짧은 경우가 있다. 원본 보존을 원하면 hz=0 으로 호출.
    """
    if not samples or target_hz <= 0:
        return samples
    interval_ns = int(1e9 / target_hz)
    if interval_ns <= 0:
        return samples
    # "다음 목표 시점을 지났는가" 방식 — jitter 누적 drift 없음.
    t_base = samples[0][0]
    next_target = 0
    out = []
    for t_ns, vals in samples:
        rel = t_ns - t_base
        if rel >= next_target:
            out.append((t_ns, vals))
            next_target = rel + interval_ns
    return out


def _write_csv(out_path: Path, samples, initial_hold: float, time_offset: float = 0.0):
    """TrajStudio 호환 CSV 작성.

    Row 0: duration = initial_hold, 값 = 첫 sample 값 (초기 hold)
    Row k (k>=1): duration = (t_k - t_{k-1}), 값 = t_k 의 sample 값

    time_offset > 0: initial_hold 를 그만큼 늘려 첫 sample 값을 앞에 더 hold.
    time_offset < 0: 앞에서 |offset| 초 만큼의 sample 을 drop (그 시점 이후로 첫 sample 재배치).
    """
    if not samples:
        return 0
    if time_offset > 0:
        initial_hold = initial_hold + time_offset
    elif time_offset < 0:
        cutoff_ns = samples[0][0] + int(-time_offset * 1e9)
        samples = [s for s in samples if s[0] >= cutoff_ns]
        if not samples:
            return 0
    n_joints = len(samples[0][1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["duration"] + [f"joint_{i + 1}" for i in range(n_joints)]
        w.writerow(header)
        # 초기 hold
        w.writerow([f"{initial_hold:.3f}"] + [f"{v:.6f}" for v in samples[0][1]])
        prev_ns = samples[0][0]
        for t_ns, vals in samples[1:]:
            dt_s = (t_ns - prev_ns) / 1e9
            if dt_s <= 0:
                continue  # parse_via_csv 에서 duration<=0 은 무시됨
            w.writerow([f"{dt_s:.6f}"] + [f"{v:.6f}" for v in vals])
            prev_ns = t_ns
    return len(samples)


def _write_showcase_csv(out_path: Path, data: dict, topic_meta: dict,
                        target_hz: float, time_offset: float = 0.0) -> int:
    """rosbag → 단일 full-robot showcase CSV (uniform grid, intersection range).

    Columns: time, pos_<joint> [rad], torque_<joint> [Nm].
    Master clock = arange(t_start, t_end, 1/target_hz) where
        t_start = max(first_t over all active streams)
        t_end   = min(last_t  over all active streams)
    이 범위는 모든 publisher 가 publish 중인 구간이라 전 row 가 빈 셀 없이 채워짐.
    각 컬럼은 해당 시점 기준 nearest-prior lookup.
    """
    import math

    # 1) joint full name 등장 순서 보존.
    column_joint_order: list[str] = []
    for group, names in ALLEX_CSV_JOINT_NAMES.items():
        column_joint_order.extend(names)

    # 2) Per-joint stream: full_name → list[(t_ns, value)] (pos / torque 따로).
    #
    # 토픽 → 컬럼 매핑 표 (group 별 N 개 joint, ALLEX_CSV_JOINT_NAMES 순서대로 인덱싱):
    #   /robot_outbound_data/<group>/joint_positions_deg [deg]
    #       → pos_<full_joint_name>   (deg → rad 변환)
    #   /result/<group>/joint/torque                     [Nm]   ← 필터된 joint torque
    #       → torque_<full_joint_name> (단위 그대로)
    # _parse_bag 가 /result/<group>/joint/torque 를 (group, "joint_torque") 로 매핑하므로
    # 아래 `suffix == "joint_torque"` 분기는 그대로 동작. raw 잔차 포함된
    # /robot_outbound_data/<group>/joint_torque 는 _parse_bag 에서 skip 됨.
    pos_streams: dict[str, list] = {n: [] for n in column_joint_order}
    trq_streams: dict[str, list] = {n: [] for n in column_joint_order}
    # ext_force streams: topic_id → list[(t_ns, [fx, fy, fz])]   (chest_origin frame)
    ext_force_streams: dict[str, list] = {
        tid: [] for tid in EXT_FORCE_TOPICS.values()
    }
    # contact_pos streams: topic_id → list[(t_ns, [px, py, pz])] (chest_origin frame)
    contact_pos_streams: dict[str, list] = {
        tid: [] for tid in EXT_CONTACT_POS_TOPICS.values()
    }

    for topic, samples in data.items():
        group, suffix = topic_meta[topic]
        if group == "_ext_force":
            topic_id = suffix
            for t_ns, vals in samples:
                if len(vals) < 3:
                    continue
                ext_force_streams[topic_id].append(
                    (t_ns, [float(vals[0]), float(vals[1]), float(vals[2])])
                )
            continue
        if group == "_contact_pos":
            topic_id = suffix
            for t_ns, vals in samples:
                if len(vals) < 3:
                    continue
                contact_pos_streams[topic_id].append(
                    (t_ns, [float(vals[0]), float(vals[1]), float(vals[2])])
                )
            continue
        joint_names = ALLEX_CSV_JOINT_NAMES.get(group)
        if joint_names is None:
            continue
        if suffix == "joint_positions_deg":
            target = pos_streams
            scale = math.pi / 180.0  # deg → rad
        elif suffix == "joint_torque":
            target = trq_streams
            scale = 1.0
        else:
            continue
        for t_ns, vals in samples:
            for j_idx, name in enumerate(joint_names):
                if j_idx >= len(vals):
                    continue
                target[name].append((t_ns, vals[j_idx] * scale))

    for d in (pos_streams, trq_streams):
        for k in d:
            d[k].sort(key=lambda x: x[0])
    for k in ext_force_streams:
        ext_force_streams[k].sort(key=lambda x: x[0])
    for k in contact_pos_streams:
        contact_pos_streams[k].sort(key=lambda x: x[0])

    # 3) Active joint set (실제 데이터 있는 것만).
    active_pos_joints = [n for n in column_joint_order if pos_streams[n]]
    active_trq_joints = [n for n in column_joint_order if trq_streams[n]]
    if not active_pos_joints:
        print("[ERROR] showcase 모드: joint_positions_deg 데이터가 없습니다.")
        return 0

    # 4) Intersection range = max(first_t) ~ min(last_t) — 모든 stream 이 valid 한 구간.
    # ext_force / contact_pos stream 도 있으면 포함시켜 전 컬럼 채워지는 구간만 남김.
    active_ext_ids = [tid for tid, s in ext_force_streams.items() if s]
    active_cpos_ids = [tid for tid, s in contact_pos_streams.items() if s]
    all_streams = ([pos_streams[n] for n in active_pos_joints]
                   + [trq_streams[n] for n in active_trq_joints]
                   + [ext_force_streams[tid] for tid in active_ext_ids]
                   + [contact_pos_streams[tid] for tid in active_cpos_ids])
    t_start_ns = max(s[0][0] for s in all_streams)
    t_end_ns = min(s[-1][0] for s in all_streams)
    if t_end_ns <= t_start_ns:
        print(f"[ERROR] showcase 모드: 모든 stream 이 겹치는 시간 구간이 없습니다.")
        return 0

    interval_ns = int(1e9 / target_hz)

    # time_offset 적용:
    #   > 0 : 첫 sample 값을 그만큼 앞에 패딩 (data 시작이 csv_time = offset 으로 밀림).
    #   < 0 : 앞에서 |offset| 초 만큼 trim (그만큼 늦은 t_ns 부터 시작).
    n_pad_rows = 0
    if time_offset > 0:
        n_pad_rows = int(round(time_offset * target_hz))
    elif time_offset < 0:
        skip_ns = int(round(-time_offset * 1e9))
        new_t_start_ns = t_start_ns + skip_ns
        if new_t_start_ns >= t_end_ns:
            print(f"[ERROR] time_offset={time_offset:+.3f}s trim 후 데이터가 남지 않습니다.")
            return 0
        t_start_ns = new_t_start_ns

    t0_offset_ns = t_start_ns  # CSV time=0 기준점 (offset 적용 후)

    # 5) Per-stream nearest-prior lookup (sorted streams 가정, monotonic advance).
    def _build_lookup(stream):
        idx = [0]
        def _at(t_ns: int) -> float:
            while idx[0] + 1 < len(stream) and stream[idx[0] + 1][0] <= t_ns:
                idx[0] += 1
            return stream[idx[0]][1]
        return _at

    pos_lookup = {n: _build_lookup(pos_streams[n]) for n in active_pos_joints}
    trq_lookup = {n: _build_lookup(trq_streams[n]) for n in active_trq_joints}

    # ─────────────────────────────────────────────────────────────────────
    # 5b) ext_force / contact_pos — chest_origin frame 으로 RAW 저장.
    # CSV 컬럼:
    #   ext_force_<topic_id>_{x,y,z}    — chest_origin frame 의 force vector
    #   contact_pos_<topic_id>_{x,y,z}  — chest_origin frame 의 contact point
    # runtime 에서 sim chest_origin USD transform 으로 world 로 변환.
    ext_lookup = {tid: _build_lookup(ext_force_streams[tid]) for tid in active_ext_ids}
    cpos_lookup = {tid: _build_lookup(contact_pos_streams[tid]) for tid in active_cpos_ids}

    # 6) Uniform grid 위에 한 줄씩 emit.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = (["time"]
                  + [f"pos_{n}" for n in active_pos_joints]
                  + [f"torque_{n}" for n in active_trq_joints])
        # ext_force / contact_pos 컬럼 — ShowcaseReader 가 prefix 매칭으로 가져감.
        for tid in active_ext_ids:
            header += [f"ext_force_{tid}_{ax}" for ax in ("x", "y", "z")]
        for tid in active_cpos_ids:
            header += [f"contact_pos_{tid}_{ax}" for ax in ("x", "y", "z")]
        w.writerow(header)

        # +offset: pad 행 들 (모두 t_start_ns 시점 nearest-prior 값).
        # csv_time = 0, dt, 2dt, ..., (n_pad_rows-1)*dt.
        # 이후 main loop 의 첫 행은 csv_time = n_pad_rows*dt = time_offset.
        for i in range(n_pad_rows):
            t_s = i / target_hz
            row = [f"{t_s:.6f}"]
            for n in active_pos_joints:
                row.append(f"{pos_lookup[n](t_start_ns):.6f}")
            for n in active_trq_joints:
                row.append(f"{trq_lookup[n](t_start_ns):.6f}")
            for tid in active_ext_ids:
                fx, fy, fz = ext_lookup[tid](t_start_ns)
                row += [f"{fx:.6f}", f"{fy:.6f}", f"{fz:.6f}"]
            for tid in active_cpos_ids:
                px, py, pz = cpos_lookup[tid](t_start_ns)
                row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"]
            w.writerow(row)
            n_rows += 1

        # t_s base shift: +offset 일 때 main rows 는 offset 만큼 더해서 출력 (pad 와 연속).
        t_s_shift = time_offset if time_offset > 0 else 0.0

        t_ns = t_start_ns
        while t_ns <= t_end_ns:
            t_s = (t_ns - t0_offset_ns) / 1e9 + t_s_shift
            row = [f"{t_s:.6f}"]
            for n in active_pos_joints:
                row.append(f"{pos_lookup[n](t_ns):.6f}")
            for n in active_trq_joints:
                row.append(f"{trq_lookup[n](t_ns):.6f}")
            # ext_force / contact_pos 는 chest_origin frame 그대로 — runtime 에서 변환.
            for tid in active_ext_ids:
                fx, fy, fz = ext_lookup[tid](t_ns)
                row += [f"{fx:.6f}", f"{fy:.6f}", f"{fz:.6f}"]
            for tid in active_cpos_ids:
                px, py, pz = cpos_lookup[tid](t_ns)
                row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"]
            w.writerow(row)
            n_rows += 1
            t_ns += interval_ns

    print(f"  grid:            {target_hz:.0f} Hz × {(t_end_ns - t_start_ns) / 1e9:.3f} s")
    print(f"  pos columns:     {len(active_pos_joints)} / {len(column_joint_order)} joints")
    print(f"  torque columns:  {len(active_trq_joints)} / {len(column_joint_order)} joints")
    if active_ext_ids:
        print(f"  ext_force:       {len(active_ext_ids)} ({', '.join(active_ext_ids)})")
    if active_cpos_ids:
        print(f"  contact_pos:     {len(active_cpos_ids)} ({', '.join(active_cpos_ids)})")
    if active_ext_ids or active_cpos_ids:
        print(f"                   frame: chest_origin (CsvReplayer 가 runtime 에 world 변환)")
    if not active_ext_ids and not active_cpos_ids:
        print(f"  ext_force/pos:   없음")
    print(f"  intersection:    {(t_start_ns - min(s[0][0] for s in all_streams)) / 1e6:.1f} ms 늦은 시작 / "
          f"{(max(s[-1][0] for s in all_streams) - t_end_ns) / 1e6:.1f} ms 일찍 끝")
    if time_offset != 0.0:
        action = (f"앞에 {time_offset:.3f} s pad 추가 (첫 sample 값 복제)" if time_offset > 0
                  else f"앞에서 {-time_offset:.3f} s trim")
        print(f"  time_offset:     {time_offset:+.3f} s — {action}")
    return n_rows


def main():
    parser = argparse.ArgumentParser(description="rosbag2 → TrajStudio / Showcase CSV 변환")
    parser.add_argument("input", help="입력 rosbag2 디렉토리")
    parser.add_argument("--output", default=None,
                        help="출력 디렉토리 (기본: <bag_parent>_csv)")
    parser.add_argument("--format", choices=("trajstudio", "showcase"), default="trajstudio",
                        help="trajstudio (default): 14-group CSV, TrajStudio 재생용. "
                             "showcase: 단일 full-robot CSV, Real/Sim Replay 패널의 real_*.csv 용.")
    parser.add_argument("--hz", type=float, default=0.0,
                        help="[trajstudio] 재생용 via-point sampling rate [Hz]. 0=원본 그대로.")
    parser.add_argument("--torque-hz", type=float, default=0.0,
                        help="[trajstudio] torque/속도 CSV sampling rate [Hz]. 0=원본 그대로.")
    parser.add_argument("--initial-hold", type=float, default=1.0,
                        help="[trajstudio] 첫 행 initial hold [s] (default 1.0). "
                             "rosbag 첫 sample 에 duration 이 없어 0 으로 두면 "
                             "player 가 첫 sample 을 스킵하므로 1 s 정도 hold 권장.")
    parser.add_argument("--showcase-hz", type=float, default=1000.0,
                        help="[showcase] uniform grid sampling rate [Hz] (default 1000). "
                             "출력 row = 모든 stream 이 valid 한 intersection 시간 구간에서 "
                             "이 rate 로 균등 샘플링 (전 row 빈 셀 없음).")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="시간 offset [s]. + 값: 첫 sample 값을 그만큼 앞에 복제 패딩 "
                             "(전체 길이 +offset). - 값: 앞에서 |offset| 초 만큼 trim "
                             "(전체 길이 -|offset|). default 0 = 변형 없음.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[ERROR] 입력 경로가 없어요: {input_path}")
        return

    # rosbag2 storage 는 metadata.yaml 이 같이 있는 디렉토리를 기대.
    # metadata.yaml 없이 .mcap 만 있는 형태도 흔해서 자동 감지:
    #   - input 이 .mcap 파일       → 그대로 reader 에 넘김
    #   - input 이 dir + metadata.yaml 있음 → dir 그대로
    #   - input 이 dir + metadata 없음 + 단일 .mcap 있음 → 그 .mcap 으로 fallback
    bag_uri = input_path
    if input_path.is_dir():
        if (input_path / "metadata.yaml").exists():
            pass  # 표준 rosbag2 layout
        else:
            mcaps = sorted(input_path.glob("*.mcap"))
            if len(mcaps) == 1:
                bag_uri = mcaps[0]
                print(f"[INFO] metadata.yaml 없음 → 단일 .mcap 으로 fallback: {bag_uri.name}")
            elif len(mcaps) == 0:
                print(f"[ERROR] '{input_path}' 에 metadata.yaml 도 .mcap 도 없습니다.")
                return
            else:
                print(f"[ERROR] '{input_path}' 에 .mcap 파일이 여러 개 ({len(mcaps)} 개) 입니다. "
                      f"파일 한 개를 직접 인자로 넘겨주세요.")
                return

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        # source 와 같은 디렉토리에 출력 (rosbag dir 의 parent).
        # input  : trajectory/dumbbell_20260415/rosbag2_xxx/
        # output : trajectory/dumbbell_20260415/
        output_dir = input_path.parent

    print(f"입력: {input_path}")
    print(f"출력: {output_dir}")
    print(f"포맷: {args.format}")
    if args.format == "trajstudio":
        print(f"재생용 Hz: {args.hz}, 분석용 Hz: {args.torque_hz}")
    print()

    data, topic_meta = _parse_bag(str(bag_uri))
    if not data:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "showcase":
        out_path = output_dir / f"real_{input_path.name}.csv"
        n = _write_showcase_csv(out_path, data, topic_meta, args.showcase_hz,
                                time_offset=args.time_offset)
        print(f"\n=== 변환 결과 ===")
        print(f"  {out_path.relative_to(output_dir)}: {n} rows")
        print(f"\nReal/Sim Replay 패널에서 'trajectory/<group>/real_*.csv' 로 인식되려면 "
              f"trajectory/<group>/ 하위로 이동/심볼릭 링크 필요.")
        return

    # ---- trajstudio format ----
    written = {}
    for topic, samples in data.items():
        if not samples:
            continue
        group, suffix = topic_meta[topic]
        subdir_name = TOPIC_TO_SUBDIR[suffix]
        target_hz = args.torque_hz if suffix in ("joint_torque", "joint_trq_target",
                                                  "joint_velocities_deg_s") else args.hz
        down = _downsample(samples, target_hz)
        out_path = output_dir / subdir_name / f"{group}.csv"
        n = _write_csv(out_path, down, args.initial_hold,
                       time_offset=args.time_offset)
        written[str(out_path.relative_to(output_dir))] = n

    # 요약
    print("=== 변환 결과 ===")
    for path, n in sorted(written.items()):
        print(f"  {path}: {n} rows")
    print(f"\n총 {len(written)} 개 CSV 생성. TrajStudio 에서 그룹 선택 후 Run 하면 재생 가능.")


if __name__ == "__main__":
    main()
