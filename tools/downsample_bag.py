"""
rosbag2 downsampler — 지정 토픽을 target_hz로 다운샘플링해서 새 bag 생성.

Usage:
    python3 downsample_bag.py <input_bag_dir> [--hz 200] [--output <out_dir>]

Example:
    python3 downsample_bag.py trajectory/dumbbell_20260415/rosbag2_2026_04_15-23_32_17 --hz 200
"""

import argparse
import os
from pathlib import Path

import rosbag2_py


JOINT_POSITION_SUFFIX = "joint_positions_deg"


def downsample(input_dir: str, target_hz: float, output_dir: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    # joint_positions_deg 토픽만 선택
    target_topics = [t.name for t in topic_types if t.name.endswith(JOINT_POSITION_SUFFIX)]
    if not target_topics:
        print("[ERROR] joint_positions_deg 토픽을 찾을 수 없어요.")
        return

    print(f"대상 토픽 {len(target_topics)}개:")
    for t in sorted(target_topics):
        print(f"  {t}")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    for i, topic in enumerate(target_topics):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=i,
            name=topic,
            type=type_map[topic],
            serialization_format="cdr",
        ))

    # 토픽별 마지막 기록 시각 추적 (ns)
    interval_ns = int(1e9 / target_hz)
    last_written: dict[str, int] = {}

    reader.set_filter(rosbag2_py.StorageFilter(topics=target_topics))

    written = 0
    skipped = 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        last = last_written.get(topic, -interval_ns)
        if t_ns - last >= interval_ns:
            writer.write(topic, data, t_ns)
            last_written[topic] = t_ns
            written += 1
        else:
            skipped += 1

    print(f"\n완료: written={written}, skipped={skipped}")
    print(f"출력: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="rosbag2 joint_positions_deg downsampler")
    parser.add_argument("input", help="입력 bag 디렉토리")
    parser.add_argument("--hz", type=float, default=200.0, help="목표 주파수 (기본 200)")
    parser.add_argument("--output", default=None, help="출력 bag 디렉토리 (기본: input_200hz)")
    args = parser.parse_args()

    input_dir = str(Path(args.input).resolve())
    if args.output:
        output_dir = str(Path(args.output).resolve())
    else:
        output_dir = input_dir.rstrip("/") + f"_{int(args.hz)}hz"

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"기존 출력 삭제: {output_dir}")

    print(f"입력: {input_dir}")
    print(f"목표: {args.hz} Hz → {output_dir}\n")
    downsample(input_dir, args.hz, output_dir)


if __name__ == "__main__":
    main()
