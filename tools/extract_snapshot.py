"""rosbag 한 시점의 robot joint pos + PointCloud2 동시 추출 (PoC test 용).

이 bag 형식 전용:
    - storage: sqlite3 (.db3)
    - topic prefix: /allex/p001/robot_outbound_data/<group>/joint_positions_deg
    - group naming: short ("waist", "neck", "left_arm", "left_thumb", ...)
    - /result/ 토픽 없음 → outbound joint_positions_deg (deg, unfiltered) 만 사용

ROS2 sourced 별도 프로세스에서 실행:
    source /opt/ros/jazzy/setup.bash
    python3 tools/extract_snapshot.py /home/asher12/Downloads/rosbag2_2026_05_19-14_01_31

Output (output_dir, default = bag_dir/snapshot_<frame_index>/):
    pc.npz          (xyz Nx3, rgb Nx3 [0..1])
    joints.json     ({joint_full_name: angle_rad, ...})
    meta.json       (bag, frame_index, pc_timestamp_ns, ...)
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# 이번 bag 의 group 이름 → ALLEX_CSV_JOINT_NAMES 의 group key
GROUP_ALIAS = {
    "waist":         "theOne_waist",
    "neck":          "theOne_neck",
    "left_arm":      "Arm_L_theOne",
    "right_arm":     "Arm_R_theOne",
    "left_thumb":    "Hand_L_thumb_wir",
    "left_index":    "Hand_L_index_wir",
    "left_middle":   "Hand_L_middle_wir",
    "left_ring":     "Hand_L_ring_wir",
    "left_little":   "Hand_L_little_wir",
    "right_thumb":   "Hand_R_thumb_wir",
    "right_index":   "Hand_R_index_wir",
    "right_middle":  "Hand_R_middle_wir",
    "right_ring":    "Hand_R_ring_wir",
    "right_little":  "Hand_R_little_wir",
}

TOPIC_PREFIX = "/allex/p001/robot_outbound_data"
PC_TOPIC = "/orbbec/depth_registered/points"


def _load_allex_joint_names():
    """joint_name_map.py 에서 ALLEX_CSV_JOINT_NAMES dict 가져오기."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    jnm_path = repo_root / "src" / "allex" / "trajectory_generate" / "joint_name_map.py"
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("joint_name_map", jnm_path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ALLEX_CSV_JOINT_NAMES


def _open_reader(bag_dir, topics):
    reader = rosbag2_py.SequentialReader()
    storage = "mcap" if list(pathlib.Path(bag_dir).glob("*.mcap")) else "sqlite3"
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def _read_pc_frame(bag_dir, frame_index):
    """N-번째 PC 메시지 가져오기. (timestamp_ns, decoded_msg) 반환."""
    reader = _open_reader(bag_dir, [PC_TOPIC])
    PC2 = get_message("sensor_msgs/msg/PointCloud2")
    count = 0
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic != PC_TOPIC:
            continue
        if count == frame_index:
            return ts_ns, deserialize_message(data, PC2)
        count += 1
    raise RuntimeError(f"frame {frame_index} not found on {PC_TOPIC} (only {count} msgs)")


def _read_latest_joint_positions(bag_dir, target_ts_ns, allex_groups):
    """target_ts_ns 이하 마지막 joint_positions_deg 를 그룹별로 모음.

    반환: {joint_full_name: angle_rad}
    """
    topics = []
    bag_group_to_allex = {}
    for bag_grp, allex_grp in GROUP_ALIAS.items():
        if allex_grp not in allex_groups:
            continue
        t = f"{TOPIC_PREFIX}/{bag_grp}/joint_positions_deg"
        topics.append(t)
        bag_group_to_allex[t] = allex_grp

    reader = _open_reader(bag_dir, topics)
    FloatArr = get_message("std_msgs/msg/Float64MultiArray")
    latest: dict[str, tuple[int, list[float]]] = {}  # topic → (ts, values)
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if ts_ns > target_ts_ns:
            continue
        msg = deserialize_message(data, FloatArr)
        latest[topic] = (ts_ns, list(msg.data))

    out_rad: dict[str, float] = {}
    missing_groups = []
    for topic, allex_grp in bag_group_to_allex.items():
        if topic not in latest:
            missing_groups.append(allex_grp)
            continue
        _, values = latest[topic]
        joint_names = allex_groups[allex_grp]
        if len(values) < len(joint_names):
            print(f"WARN {topic}: only {len(values)} values, expected {len(joint_names)}")
        for i, jn in enumerate(joint_names):
            if i >= len(values):
                break
            out_rad[jn] = float(values[i]) * math.pi / 180.0
    if missing_groups:
        print(f"WARN no joint_positions_deg msg before target_ts for groups: {missing_groups}")
    return out_rad


def _decode_pc(msg, sys_path_extra):
    sys.path.insert(0, sys_path_extra)
    import pc2_reader as r
    return r.decode_pointcloud2(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bag_dir")
    p.add_argument("--frame-index", type=int, default=0,
                   help="N-번째 PC 메시지 기준 (default 0)")
    p.add_argument("--output-dir", default=None,
                   help="default: <bag_dir>/snapshot_<frame_index>/")
    args = p.parse_args()

    bag = pathlib.Path(args.bag_dir)
    out = pathlib.Path(args.output_dir) if args.output_dir else (bag / f"snapshot_{args.frame_index}")
    out.mkdir(parents=True, exist_ok=True)

    tools_dir = str(pathlib.Path(__file__).resolve().parent)

    # 1) PC 추출
    print(f"[1/2] PC frame #{args.frame_index} from {PC_TOPIC} ...")
    pc_ts_ns, msg = _read_pc_frame(bag, args.frame_index)
    print(f"      timestamp = {pc_ts_ns} ns ({pc_ts_ns / 1e9:.3f}s epoch), "
          f"frame_id={msg.header.frame_id!r}, width={msg.width}")
    xyz, rgb = _decode_pc(msg, tools_dir)
    np.savez(out / "pc.npz", xyz=xyz, rgb=rgb)
    print(f"      → {out/'pc.npz'} (xyz={xyz.shape}, rgb={rgb.shape})")

    # 2) joint position 추출
    print(f"[2/2] joint_positions_deg at ts <= {pc_ts_ns} ns ...")
    allex_groups = _load_allex_joint_names()
    joints_rad = _read_latest_joint_positions(bag, pc_ts_ns, allex_groups)
    print(f"      {len(joints_rad)} joints collected")
    with open(out / "joints.json", "w") as f:
        json.dump(joints_rad, f, indent=2, sort_keys=True)
    print(f"      → {out/'joints.json'}")

    # meta
    meta = {
        "bag": str(bag),
        "pc_topic": PC_TOPIC,
        "pc_frame_index": args.frame_index,
        "pc_timestamp_ns": pc_ts_ns,
        "pc_frame_id": msg.header.frame_id,
        "n_points": int(xyz.shape[0]),
        "n_joints": len(joints_rad),
        "topic_prefix": TOPIC_PREFIX,
        "group_alias": GROUP_ALIAS,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
