"""sensor_msgs/PointCloud2 mcap bag → numpy 변환 + SE(3) extrinsic util.

Isaac Sim 의존성 없음. conda env 의 mcap / mcap_ros2 만 사용.

사용 예 (Script Editor):
    from tools import pc2_reader as r
    xyz, rgb = r.read_first_pointcloud2(
        "/path/to/bag",
        topic="/camera/depth_registered/points",
    )
    # 일단 extrinsic 모르면 identity 로 → camera frame == world 로 가정

bag 가 dir 면 dir 안 *.mcap 다 훑음. .mcap 파일 직접 줘도 됨.
"""

from __future__ import annotations

import pathlib

import numpy as np


# --- PointCloud2 decode ------------------------------------------------------

# datatype enum (sensor_msgs/PointField)
_PF_TO_NP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def decode_pointcloud2(msg) -> tuple[np.ndarray, np.ndarray | None]:
    """sensor_msgs/PointCloud2 → (xyz Nx3 float32, rgb Nx3 float[0..1] or None).

    invalid (NaN/inf) point 자동 제거. organized 정보는 잃음 (flat N).
    """
    n = msg.height * msg.width
    point_step = msg.point_step
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, point_step)

    field_map = {f.name: f for f in msg.fields}

    def _read_field(name: str) -> np.ndarray:
        f = field_map[name]
        dt = _PF_TO_NP[f.datatype]
        size = np.dtype(dt).itemsize
        col = raw[:, f.offset : f.offset + size].copy()
        return col.view(dt).reshape(n).astype(np.float32)

    xyz = np.stack([_read_field("x"), _read_field("y"), _read_field("z")], axis=1)

    rgb: np.ndarray | None = None
    if "rgb" in field_map:
        f = field_map["rgb"]
        col = raw[:, f.offset : f.offset + 4].copy().view(np.uint32).reshape(n)
        r = ((col >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((col >> 8) & 0xFF).astype(np.float32) / 255.0
        b = (col & 0xFF).astype(np.float32) / 255.0
        rgb = np.stack([r, g, b], axis=1)
    elif "rgba" in field_map:
        f = field_map["rgba"]
        col = raw[:, f.offset : f.offset + 4].copy()
        rgb = (col[:, :3].astype(np.float32)) / 255.0

    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    if rgb is not None:
        rgb = rgb[mask]
    return xyz, rgb


# --- mcap bag reader ---------------------------------------------------------


def _iter_mcap_files(path: str | pathlib.Path) -> list[pathlib.Path]:
    p = pathlib.Path(path)
    if p.is_file() and p.suffix == ".mcap":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.mcap"))
        if not files:
            raise FileNotFoundError(f"no .mcap files under {p}")
        return files
    raise FileNotFoundError(p)


def read_first_pointcloud2(bag: str | pathlib.Path, topic: str):
    """첫 메시지 1개 → decode 결과."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    for mcap_path in _iter_mcap_files(bag):
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, channel, _msg, decoded in reader.iter_decoded_messages(topics=[topic]):
                if channel.topic == topic:
                    return decode_pointcloud2(decoded)
    raise RuntimeError(f"no message on topic {topic!r} in {bag}")


def iter_pointcloud2(bag: str | pathlib.Path, topic: str):
    """모든 frame iter — (t_nanosec, xyz, rgb)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    for mcap_path in _iter_mcap_files(bag):
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, channel, msg, decoded in reader.iter_decoded_messages(topics=[topic]):
                if channel.topic != topic:
                    continue
                xyz, rgb = decode_pointcloud2(decoded)
                yield msg.log_time, xyz, rgb


# --- SE(3) extrinsic --------------------------------------------------------


def apply_se3(xyz: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    """xyz Nx3 → T_4x4 @ [xyz, 1]^T 의 위 3행. 결과 Nx3."""
    homo = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)])
    return (homo @ T_4x4.T.astype(xyz.dtype))[:, :3]


def make_se3(R_3x3: np.ndarray, t_3: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_3x3
    T[:3, 3] = t_3
    return T


# --- 합성 PC (bag 없이 pipeline 검증) ----------------------------------------


def synthetic_wall(width: int = 320, height: int = 240, z: float = 1.0, half_size: float = 0.5):
    """camera optical frame 의 z=1 m 평면에 깔린 격자.

    색은 (u,v) gradient. 합성 PC 가 한 번 보이면 decode + apply_se3 + show_array 경로 검증 끝.
    """
    us = np.linspace(-half_size, half_size, width, dtype=np.float32)
    vs = np.linspace(-half_size, half_size, height, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    zz = np.full_like(uu, z, dtype=np.float32)
    xyz = np.stack([uu, vv, zz], axis=-1).reshape(-1, 3)

    r = ((uu + half_size) / (2 * half_size)).reshape(-1)
    g = ((vv + half_size) / (2 * half_size)).reshape(-1)
    b = np.full_like(r, 0.5, dtype=np.float32)
    rgb = np.stack([r, g, b], axis=1).astype(np.float32)
    return xyz, rgb
