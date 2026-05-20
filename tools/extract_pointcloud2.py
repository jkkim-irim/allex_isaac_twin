"""rosbag2 (sqlite3 .db3 또는 .mcap) 에서 PointCloud2 첫 프레임 추출 → .npz.

Isaac Sim 의 conda env (rosbag2_py 포함) 에서 실행:
    python3 tools/extract_pointcloud2.py /home/.../rosbag2_2026_05_19-14_01_31
    python3 tools/extract_pointcloud2.py <bag> --list                  # 토픽 나열만
    python3 tools/extract_pointcloud2.py <bag> --filter orbbec         # substr 매칭
    python3 tools/extract_pointcloud2.py <bag> --topic /camera/depth   # 정확한 topic
    python3 tools/extract_pointcloud2.py <bag> --frame-index 30        # 30번째 프레임

Output: <bag_dir>/<topic_safe>_frame<i>.npz (xyz Nx3 float32, rgb Nx3 [0..1] optional)

Script Editor 에서 로드:
    import numpy as np
    d = np.load("<path>.npz")
    xyz = d["xyz"]
    rgb = d["rgb"] if "rgb" in d.files else None
    pc.update(xyz, rgb)
"""

import argparse
import pathlib
import sys

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def detect_storage(bag_dir: pathlib.Path) -> str:
    if list(bag_dir.glob("*.mcap")):
        return "mcap"
    if list(bag_dir.glob("*.db3")):
        return "sqlite3"
    raise FileNotFoundError(f"no .mcap or .db3 in {bag_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bag_dir")
    p.add_argument("--topic", default=None, help="정확한 토픽 이름")
    p.add_argument("--filter", default="orbbec", help="substr 매칭 (--topic 미지정 시)")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--output", default=None, help="출력 .npz 경로")
    p.add_argument("--list", action="store_true", help="토픽 나열 후 종료")
    args = p.parse_args()

    bag_dir = pathlib.Path(args.bag_dir)
    if not bag_dir.exists():
        sys.exit(f"not found: {bag_dir}")
    storage = detect_storage(bag_dir)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
        rosbag2_py.ConverterOptions("", ""),
    )

    topics = reader.get_all_topics_and_types()
    pc_topics = [t for t in topics if t.type == "sensor_msgs/msg/PointCloud2"]

    if args.list:
        print(f"Bag: {bag_dir} ({storage})")
        print(f"All topics ({len(topics)}):")
        for t in topics:
            print(f"  {t.name}  [{t.type}]")
        print(f"\nPointCloud2 topics: {len(pc_topics)}")
        for t in pc_topics:
            print(f"  {t.name}")
        return

    if not pc_topics:
        print("no PointCloud2 topics in bag, all topics:")
        for t in topics:
            print(f"  {t.name}  [{t.type}]")
        sys.exit(1)

    if args.topic:
        target = args.topic
    else:
        matches = [t.name for t in pc_topics if args.filter.lower() in t.name.lower()]
        if not matches:
            print(f"no PointCloud2 topic matching {args.filter!r}")
            print("available PointCloud2 topics:")
            for t in pc_topics:
                print(f"  {t.name}")
            sys.exit(1)
        target = matches[0]
        if len(matches) > 1:
            print(f"WARN multiple matches, using first: {target}")
            for m in matches:
                print(f"  - {m}")

    print(f"Extracting frame #{args.frame_index} from {target} ({storage} bag)")

    msg_class = get_message("sensor_msgs/msg/PointCloud2")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[target]))

    count = 0
    decoded = None
    while reader.has_next():
        topic_name, data, _ts = reader.read_next()
        if topic_name != target:
            continue
        if count == args.frame_index:
            decoded = deserialize_message(data, msg_class)
            break
        count += 1

    if decoded is None:
        sys.exit(f"frame {args.frame_index} not found on {target} (only {count} message(s))")

    print(
        f"  height={decoded.height} width={decoded.width} "
        f"point_step={decoded.point_step} row_step={decoded.row_step} "
        f"is_dense={decoded.is_dense} frame_id={decoded.header.frame_id!r}"
    )
    print(f"  fields:")
    for f in decoded.fields:
        print(f"    name={f.name} offset={f.offset} datatype={f.datatype} count={f.count}")

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import pc2_reader as r

    xyz, rgb = r.decode_pointcloud2(decoded)

    if args.output:
        out = pathlib.Path(args.output)
    else:
        safe = target.strip("/").replace("/", "_")
        out = bag_dir / f"{safe}_frame{args.frame_index}.npz"

    save_kwargs = {"xyz": xyz}
    if rgb is not None:
        save_kwargs["rgb"] = rgb
    np.savez(out, **save_kwargs)
    print(
        f"saved {out}: xyz={xyz.shape}, "
        f"rgb={tuple(rgb.shape) if rgb is not None else 'none'}, "
        f"valid_pts={xyz.shape[0]}"
    )


if __name__ == "__main__":
    main()
