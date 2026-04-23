"""
rosbag2 → TrajStudio CSV 변환기.

Extension 의 TrajStudio 가 읽는 14-group CSV 포맷으로 bag 을 변환한다.
- 재생용 (joint_ang_target_deg) 은 루트에, 부가 데이터 (torque, velocity 등) 는
  sub-folder 에 저장. TrajStudio 는 루트의 14 개 CSV 만 스캔하므로 sub-folder
  파일들은 재생에 영향 없이 분석용으로 남는다.

Output 경로:
  <bag_parent>_csv/
    Arm_L_theOne.csv          ← 재생용 (joint_ang_target_deg, downsampled)
    Arm_R_theOne.csv
    ...
    theOne_neck.csv
    _current/                 ← joint_positions_deg (실측 위치)
      Arm_L_theOne.csv
      ...
    _torque/                  ← joint_torque (원본 1kHz 유지)
      Arm_L_theOne.csv
      ...
    _velocity/                ← joint_velocities_deg_s
      ...
    _torque_target/           ← joint_trq_target
      ...

Usage:
  python3 tools/rosbag_to_csv.py <input_bag_dir> [--hz 50] [--torque-hz 1000]
"""

import argparse
import csv
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float64MultiArray


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


def _parse_bag(input_dir: str):
    """bag 을 읽어 {topic_name: [(t_ns, [values])...] } 리턴."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # 관심 topic 필터: /robot_outbound_data/<group>/<suffix>
    keep = []
    topic_meta = {}  # topic → (group, suffix)
    for name in type_map:
        parts = name.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "robot_outbound_data":
            continue
        group, suffix = parts[1], parts[2]
        if group not in CSV_GROUPS:
            continue
        if suffix not in TOPIC_TO_SUBDIR:
            continue
        keep.append(name)
        topic_meta[name] = (group, suffix)

    if not keep:
        print("[ERROR] 대상 topic 을 찾지 못했어요. bag 경로를 확인하세요.")
        return {}, {}

    reader.set_filter(rosbag2_py.StorageFilter(topics=keep))

    data = {t: [] for t in keep}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        try:
            msg = deserialize_message(raw, Float64MultiArray)
        except Exception as exc:
            print(f"[WARN] deserialize 실패 ({topic}): {exc}")
            continue
        data[topic].append((t_ns, list(msg.data)))

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


def _write_csv(out_path: Path, samples, initial_hold: float):
    """TrajStudio 호환 CSV 작성.

    Row 0: duration = initial_hold, 값 = 첫 sample 값 (초기 hold)
    Row k (k>=1): duration = (t_k - t_{k-1}), 값 = t_k 의 sample 값
    """
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


def main():
    parser = argparse.ArgumentParser(description="rosbag2 → TrajStudio CSV 변환")
    parser.add_argument("input", help="입력 rosbag2 디렉토리")
    parser.add_argument("--output", default=None,
                        help="출력 디렉토리 (기본: <bag_parent>_csv)")
    parser.add_argument("--hz", type=float, default=0.0,
                        help="재생용 via-point sampling rate [Hz]. 0=원본 그대로 (default).")
    parser.add_argument("--torque-hz", type=float, default=0.0,
                        help="torque/속도 CSV sampling rate [Hz]. 0=원본 그대로 (default).")
    parser.add_argument("--initial-hold", type=float, default=1.0,
                        help="첫 행 initial hold [s] (default 1.0)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[ERROR] 입력 경로가 없어요: {input_path}")
        return

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        # input: trajectory/dumbbell_20260415/rosbag2_...
        # default output: trajectory/dumbbell_20260415_csv/
        output_dir = input_path.parent.parent / f"{input_path.parent.name}_csv"

    print(f"입력: {input_path}")
    print(f"출력: {output_dir}")
    print(f"재생용 Hz: {args.hz}, 분석용 Hz: {args.torque_hz}\n")

    data, topic_meta = _parse_bag(str(input_path))
    if not data:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

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
        n = _write_csv(out_path, down, args.initial_hold)
        written[str(out_path.relative_to(output_dir))] = n

    # 요약
    print("=== 변환 결과 ===")
    for path, n in sorted(written.items()):
        print(f"  {path}: {n} rows")
    print(f"\n총 {len(written)} 개 CSV 생성. TrajStudio 에서 그룹 선택 후 Run 하면 재생 가능.")


if __name__ == "__main__":
    main()
