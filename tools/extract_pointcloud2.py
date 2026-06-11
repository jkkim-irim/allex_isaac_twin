"""mcap bag → per-frame .npz dump for PointCloud2 replay.

Reads ROS2 sensor_msgs/PointCloud2 messages from .mcap bag files using the
``mcap`` + ``mcap_ros2`` packages (no rosbag2_py, no ROS source). Each frame
is optionally voxel-downsampled and saved as ``frame_<idx>.npz`` plus a
``manifest.json`` describing the sequence — consumed at runtime by
``allex.replay.pc_replayer.PcReplayer``.

Examples:
    # single frame (legacy behavior)
    python tools/extract_pointcloud2.py <bag_or_mcap> --frame-index 0

    # full bag, 5 mm voxel, default out dir = <bag_dir>/pc_frames_5mm/
    python tools/extract_pointcloud2.py <bag_or_mcap> --all-frames

    # every 10th frame, 1 cm voxel
    python tools/extract_pointcloud2.py <bag_or_mcap> --all-frames --every 10 --voxel 0.01

    # verify a previously-dumped sequence
    python tools/extract_pointcloud2.py --check <out_dir>

The bag argument accepts either a directory containing .mcap files or a
single .mcap file path.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pc2_reader as r


DEFAULT_TOPIC = "/orbbec/depth_registered/points"


def _voxel_downsample(xyz: np.ndarray, rgb: np.ndarray | None, voxel_m: float):
    """Average points falling into the same voxel. Returns (xyz_ds, rgb_ds)."""
    if voxel_m <= 0.0 or xyz.shape[0] == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    # Pack 3D key into single int64 for fast unique. Range guard: roughly ±10 m
    # / 0.001 m → ±1e4 fits in 21 bits per axis ⇒ safe in 63 bits.
    k = (keys[:, 0] + (1 << 20)) \
        | ((keys[:, 1] + (1 << 20)) << 21) \
        | ((keys[:, 2] + (1 << 20)) << 42)
    order = np.argsort(k, kind="stable")
    k_sorted = k[order]
    xyz_sorted = xyz[order]
    uniq, start = np.unique(k_sorted, return_index=True)
    counts = np.diff(np.append(start, k_sorted.shape[0]))
    # Group-sum via reduceat then divide by counts.
    sum_xyz = np.add.reduceat(xyz_sorted, start, axis=0)
    xyz_ds = (sum_xyz / counts[:, None]).astype(np.float32)
    rgb_ds = None
    if rgb is not None:
        rgb_sorted = rgb[order]
        sum_rgb = np.add.reduceat(rgb_sorted, start, axis=0)
        rgb_ds = (sum_rgb / counts[:, None]).astype(np.float32)
    return xyz_ds, rgb_ds


def _bag_dir_for_default_out(bag_arg: pathlib.Path) -> pathlib.Path:
    """Choose default <out> location: bag dir (if dir) or parent dir (if file)."""
    p = bag_arg
    if p.is_file():
        return p.parent
    return p


def _extract_all(
    bag: pathlib.Path,
    topic: str,
    out_dir: pathlib.Path,
    voxel_m: float,
    every: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_meta: list[dict] = []
    t0_ns: int | None = None
    last_t_ns: int = 0
    n_kept = 0
    t_start = time.monotonic()

    for i_msg, (t_ns, xyz, rgb) in enumerate(r.iter_pointcloud2(bag, topic)):
        if i_msg % every != 0:
            continue
        if t0_ns is None:
            t0_ns = int(t_ns)
        last_t_ns = int(t_ns)
        t_rel_s = (last_t_ns - t0_ns) * 1e-9
        xyz_ds, rgb_ds = _voxel_downsample(xyz, rgb, voxel_m)
        npz_name = f"frame_{n_kept:05d}.npz"
        save_kwargs = {"xyz": xyz_ds.astype(np.float32)}
        if rgb_ds is not None:
            save_kwargs["rgb"] = rgb_ds.astype(np.float32)
        np.savez(out_dir / npz_name, **save_kwargs)
        frames_meta.append({
            "idx": n_kept,
            "t_rel_s": float(t_rel_s),
            "npz": npz_name,
            "n_pts": int(xyz_ds.shape[0]),
        })
        n_kept += 1
        if n_kept % 10 == 0:
            elapsed = time.monotonic() - t_start
            print(
                f"[extract_pc2] kept {n_kept} frames (msg {i_msg+1}), "
                f"last n_pts={xyz_ds.shape[0]}, {elapsed:.1f}s elapsed",
                file=sys.stderr,
            )

    if n_kept == 0:
        sys.exit(f"no messages on topic {topic!r} in {bag}")

    duration_s = (last_t_ns - (t0_ns or 0)) * 1e-9
    # Probe frame_id from the first decoded message (re-open lightly).
    frame_id = ""
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
        for mcap_path in r._iter_mcap_files(bag):
            with open(mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for _s, ch, _m, dec in reader.iter_decoded_messages(topics=[topic]):
                    if ch.topic == topic:
                        frame_id = str(dec.header.frame_id)
                        break
            if frame_id:
                break
    except Exception:
        pass

    manifest = {
        "topic": topic,
        "frame_id": frame_id,
        "voxel_m": float(voxel_m),
        "every": int(every),
        "n_frames": n_kept,
        "t0_ns": int(t0_ns or 0),
        "duration_s": float(duration_s),
        "frames": frames_meta,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[extract_pc2] wrote {n_kept} frames + manifest.json to {out_dir} "
        f"(duration={duration_s:.2f}s, voxel={voxel_m*1000:.1f}mm)"
    )


def _extract_single(
    bag: pathlib.Path,
    topic: str,
    frame_index: int,
    out_path: pathlib.Path | None,
) -> None:
    n = 0
    decoded = None
    for t_ns, xyz, rgb in r.iter_pointcloud2(bag, topic):
        if n == frame_index:
            decoded = (t_ns, xyz, rgb)
            break
        n += 1
    if decoded is None:
        sys.exit(f"frame {frame_index} not found on {topic} (only {n} msgs)")
    _t_ns, xyz, rgb = decoded
    if out_path is None:
        safe = topic.strip("/").replace("/", "_")
        out_path = _bag_dir_for_default_out(bag) / f"{safe}_frame{frame_index}.npz"
    save_kwargs = {"xyz": xyz}
    if rgb is not None:
        save_kwargs["rgb"] = rgb
    np.savez(out_path, **save_kwargs)
    print(f"saved {out_path}: xyz={xyz.shape}, rgb={'none' if rgb is None else tuple(rgb.shape)}")


def _check_manifest(out_dir: pathlib.Path) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"manifest.json not found in {out_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    frames = manifest.get("frames", [])
    missing: list[str] = []
    bad_shape: list[str] = []
    for fm in frames:
        p = out_dir / fm["npz"]
        if not p.is_file():
            missing.append(fm["npz"])
            continue
        try:
            d = np.load(p)
            xyz = d["xyz"]
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                bad_shape.append(f"{fm['npz']} {xyz.shape}")
        except Exception as exc:
            bad_shape.append(f"{fm['npz']} load fail: {exc}")
    print(
        f"[check] {out_dir}: n_frames={len(frames)}, "
        f"voxel={manifest.get('voxel_m')}m, duration={manifest.get('duration_s')}s, "
        f"frame_id={manifest.get('frame_id')!r}"
    )
    if missing:
        print(f"[check] MISSING ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if bad_shape:
        print(f"[check] BAD ({len(bad_shape)}): {bad_shape[:5]}{'...' if len(bad_shape) > 5 else ''}")
    if not missing and not bad_shape:
        print("[check] OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bag", nargs="?",
                   help="dir containing .mcap files or a single .mcap path")
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--frame-index", type=int, default=None,
                   help="single-frame mode: extract this frame only")
    p.add_argument("--all-frames", action="store_true",
                   help="dump every frame (subject to --every)")
    p.add_argument("--every", type=int, default=1,
                   help="keep 1 of every N frames in --all-frames mode")
    p.add_argument("--voxel", type=float, default=0.005,
                   help="voxel size in meters (<=0 disables downsampling)")
    p.add_argument("--out", default=None,
                   help="output dir (--all-frames) or .npz path (--frame-index). "
                        "Default for --all-frames: <bag_dir>/pc_frames_<voxelmm>mm/")
    p.add_argument("--check", action="store_true",
                   help="verify an existing dump dir (manifest + npz files). "
                        "Pass the dir as the positional arg.")
    args = p.parse_args()

    if args.check:
        if args.bag is None:
            sys.exit("--check requires a dir positional arg")
        _check_manifest(pathlib.Path(args.bag))
        return

    if args.bag is None:
        sys.exit("missing bag (dir or .mcap file)")
    bag = pathlib.Path(args.bag)
    if not bag.exists():
        sys.exit(f"not found: {bag}")

    if args.all_frames:
        if args.out is not None:
            out_dir = pathlib.Path(args.out)
        else:
            voxel_mm = max(0, int(round(args.voxel * 1000)))
            sub = f"pc_frames_{voxel_mm}mm" if args.voxel > 0 else "pc_frames_raw"
            out_dir = _bag_dir_for_default_out(bag) / sub
        _extract_all(bag, args.topic, out_dir, args.voxel, max(1, args.every))
        return

    # single-frame mode
    idx = args.frame_index if args.frame_index is not None else 0
    out_path = pathlib.Path(args.out) if args.out else None
    _extract_single(bag, args.topic, idx, out_path)


if __name__ == "__main__":
    main()
