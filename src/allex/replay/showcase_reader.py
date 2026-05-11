"""Showcase CSV reader.

Parses a `showcase_*` style CSV (1 kHz log of joint pos/torque + per-pair
contact pos/force vectors + R_Palm aggregate net force) into per-channel
numpy arrays for kinematic replay overlay.

Column conventions:

Sim CSV (from showcase_logger):
  time                                       — seconds
  pos_<joint_short>                          — rad. joint_short = joint_full 의 lowercase
                                              minus `_Joint` (e.g. r_ring_abad). reader 가
                                              reverse-map 해서 self.pos[joint_full] 로 저장.
  torque_<joint_short>                       — Nm. qfrc_actuator + qfrc_gravcomp.
  contact_pos_<linkA> <-> <linkB>_{x,y,z}    — m, world frame. pair → pair_contact_pos.
  aggregate_origin_{x,y,z}                   — m, world frame.
  force_<linkA> <-> <linkB>                  — scalar, ignored (not loaded).
  force_aggregate_R_Palm_Net_Force           — scalar magnitude, N.
  force_vec_<linkA> <-> <linkB>_{x,y,z}      — N, world frame. → pair_force_vec.
  normal_R_Palm_Net_Force_{x,y,z}            — unit normal.

Real CSV (from rosbag_to_csv.py --format showcase):
  time, pos_<joint_short>, torque_<joint_short>   — same naming as sim
                                              (lowercase, no `_Joint` suffix).
  ext_joint_torque_<short_key>               — Nm, per-joint external torque (스칼라).
                                              `/result/<group>/joint/ext_torque` 토픽
                                              에서 풀어 쓴 값. follower joint (DIP/IP)
                                              도 포함. short_key 는
                                              `EXT_TORQUE_JOINT_CSV_KEYS[joint_full]`
                                              (e.g. hand_l_index_abad). 내부적으로는
                                              joint_full → self.ext_joint_torque[joint_full]
                                              로 reverse-map 해서 저장.
  ext_force_<topic_id>_{x,y,z}               — N, chest_origin frame. → topic_force_vec.
  ext_torque_<topic_id>_{x,y,z}              — Nm, chest_origin frame. → topic_torque_vec.
                                              (Wrench 의 torque.x/y/z. per-joint 스칼라는
                                               별도 `ext_joint_torque_` prefix 라 충돌 없음.)
  contact_pos_<topic_id>_{x,y,z}             — m, chest_origin frame. → topic_contact_pos.
  (topic_id 예: arm_l, arm_r, shoulder_l, shoulder_r — ext_force_topics 매핑)

Sim 의 pair name 은 ``<->`` / 공백 포함 → sanitize 해서 ``<linkA>__<linkB>`` 형태로
저장. real 의 topic_id 는 ``__`` 없는 단순 식별자라 별도 dict 로 분리한다.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger("allex.replay.reader")


_PAIR_SEP = " <-> "
_SANITIZED_SEP = "__"


def sanitize_pair_name(raw: str) -> str:
    """``"L_Elbow <-> R_Palm_Back"`` → ``"L_Elbow__R_Palm_Back"``.

    Idempotent: re-sanitising returns the same string.
    """
    return raw.replace(_PAIR_SEP, _SANITIZED_SEP).replace(" ", "_")


class ShowcaseReader:
    """In-memory CSV slice. Loads once, indexed by physical time."""

    def __init__(self, csv_path: Path):
        self.path = Path(csv_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"showcase csv missing: {self.path}")

        with self.path.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"empty csv: {self.path}") from exc
            rows = list(reader)

        if not rows:
            raise ValueError(f"csv has no data rows: {self.path}")

        # NaN-tolerant float parse — empty / non-numeric cells become NaN.
        data = np.full((len(rows), len(header)), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            for j, cell in enumerate(row[: len(header)]):
                if cell == "" or cell is None:
                    continue
                try:
                    data[i, j] = float(cell)
                except ValueError:
                    pass

        # 일부 logger 가 컬럼 사이에 trailing/leading 공백을 넣음 ("time  , pos_..."). strip.
        col_idx: Dict[str, int] = {name.strip(): i for i, name in enumerate(header)}

        # --- time -------------------------------------------------------
        if "time" not in col_idx:
            raise ValueError(f"csv missing 'time' column: {self.path}")
        self.t: np.ndarray = data[:, col_idx["time"]].astype(np.float64)

        # --- pos / torque / ext_joint_torque 컬럼 ------------------------
        # 모든 joint scalar 컬럼은 short form (lowercase, no `_Joint`) 으로 emit 된다.
        # reader 는 short → joint_full reverse-map 으로 dict key 를 joint_full 로 복원
        # 한다 (csv_replayer 가 articulation.dof_names 기반 full 이름으로 lookup).
        #
        # 두 종류의 ext_torque 컬럼:
        #   (a) ext_joint_torque_<short> — per-joint scalar. EXT_TORQUE_JOINT_CSV_KEYS
        #       reverse 로 full 복원 → self.ext_joint_torque[joint_full].
        #   (b) ext_torque_<topic_id>_{x,y,z} — Wrench 벡터 (별도 prefix). 아래 axis 그룹.
        #
        # Backward compat: 구버전 CSV (대문자 + `_Joint` 포함 컬럼) 도 그대로 받음 —
        # reverse-map miss 시 column key 그대로를 dict key 로 사용.
        try:
            from ..trajectory_generate.joint_name_map import (
                EXT_TORQUE_JOINT_CSV_KEYS,
                JOINT_CSV_SHORT_TO_FULL,
            )
            _ejt_short_to_full = {v: k for k, v in EXT_TORQUE_JOINT_CSV_KEYS.items()}
            _joint_short_to_full = JOINT_CSV_SHORT_TO_FULL
        except Exception:
            _ejt_short_to_full = {}
            _joint_short_to_full = {}

        self.pos: Dict[str, np.ndarray] = {}
        self.torque: Dict[str, np.ndarray] = {}
        self.ext_joint_torque: Dict[str, np.ndarray] = {}
        for name, idx in col_idx.items():
            if name.startswith("pos_"):
                key = name[len("pos_"):]
                full = _joint_short_to_full.get(key, key)
                self.pos[full] = data[:, idx].astype(np.float32)
            elif name.startswith("ext_joint_torque_"):
                short = name[len("ext_joint_torque_"):]
                full = _ejt_short_to_full.get(short, short)
                self.ext_joint_torque[full] = data[:, idx].astype(np.float32)
            elif name.startswith("torque_"):
                key = name[len("torque_"):]
                full = _joint_short_to_full.get(key, key)
                self.torque[full] = data[:, idx].astype(np.float32)

        # --- force_vec / contact_pos / ext_force ---------------------
        # Sim CSV (pair):
        #   pair_force_vec[sanitized_pair]    ← force_vec_<linkA> <-> <linkB>_*  (world)
        #   pair_contact_pos[sanitized_pair]  ← contact_pos_<linkA> <-> <linkB>_*  (world)
        # Real CSV (topic):
        #   topic_force_vec[topic_id]         ← ext_force_<id>_*  (chest_origin)
        #   topic_contact_pos[topic_id]       ← contact_pos_<id>_*  (chest_origin)
        # contact_pos_ 컬럼은 sim/real 모두 같은 prefix 라 raw key 안에 ``<->`` 또는
        # ``__`` 가 포함됐는지로 분기.
        self.pair_force_vec: Dict[str, np.ndarray] = {}
        self.pair_contact_pos: Dict[str, np.ndarray] = {}
        self.topic_force_vec: Dict[str, np.ndarray] = {}
        self.topic_torque_vec: Dict[str, np.ndarray] = {}
        self.topic_contact_pos: Dict[str, np.ndarray] = {}

        # First, group columns by raw key & axis.
        force_vec_axes: Dict[str, Dict[str, int]] = {}      # sim only
        ext_force_axes: Dict[str, Dict[str, int]] = {}      # real only
        ext_torque_axes: Dict[str, Dict[str, int]] = {}     # real only (Wrench.torque)
        contact_pos_axes: Dict[str, Dict[str, int]] = {}    # sim + real (분기)
        for name, idx in col_idx.items():
            if not (name[-2:] in ("_x", "_y", "_z")):
                continue
            axis = name[-1]
            if name.startswith("force_vec_"):
                raw = name[len("force_vec_"):-2]
                force_vec_axes.setdefault(raw, {})[axis] = idx
            elif name.startswith("ext_force_"):
                raw = name[len("ext_force_"):-2]
                ext_force_axes.setdefault(raw, {})[axis] = idx
            elif name.startswith("ext_torque_"):
                raw = name[len("ext_torque_"):-2]
                ext_torque_axes.setdefault(raw, {})[axis] = idx
            elif name.startswith("contact_pos_"):
                raw = name[len("contact_pos_"):-2]
                contact_pos_axes.setdefault(raw, {})[axis] = idx

        def _stack_xyz(axes_idx):
            return np.stack(
                [data[:, axes_idx["x"]], data[:, axes_idx["y"]], data[:, axes_idx["z"]]],
                axis=1,
            ).astype(np.float32)

        for raw_pair, axes in force_vec_axes.items():
            if not all(a in axes for a in ("x", "y", "z")):
                continue
            self.pair_force_vec[sanitize_pair_name(raw_pair)] = _stack_xyz(axes)

        for raw_id, axes in ext_force_axes.items():
            if not all(a in axes for a in ("x", "y", "z")):
                continue
            self.topic_force_vec[raw_id] = _stack_xyz(axes)

        for raw_id, axes in ext_torque_axes.items():
            if not all(a in axes for a in ("x", "y", "z")):
                continue
            self.topic_torque_vec[raw_id] = _stack_xyz(axes)

        for raw, axes in contact_pos_axes.items():
            if not all(a in axes for a in ("x", "y", "z")):
                continue
            arr = _stack_xyz(axes)
            # raw 가 sim pair (``<->``/공백) 면 pair_contact_pos, 아니면 topic.
            if (_PAIR_SEP in raw) or (" " in raw):
                self.pair_contact_pos[sanitize_pair_name(raw)] = arr
            else:
                self.topic_contact_pos[raw] = arr

        # --- aggregate (R_Palm_Net_Force) ------------------------------
        # name 은 "R_Palm_Net_Force" (이미 sanitize 안전 — 공백/<->없음).
        self.aggregate: Dict[str, Dict[str, np.ndarray]] = {}
        agg_origin_axes = {a: col_idx.get(f"aggregate_origin_{a}") for a in ("x", "y", "z")}
        if all(v is not None for v in agg_origin_axes.values()):
            origin = np.stack(
                [data[:, agg_origin_axes["x"]],
                 data[:, agg_origin_axes["y"]],
                 data[:, agg_origin_axes["z"]]],
                axis=1,
            ).astype(np.float32)
        else:
            origin = None

        # 현재 logger 가 R_Palm_Net_Force 한 종류만 내보냄. 추후 다른 tag 가
        # 추가되면 prefix 스캔으로 일반화.
        for name, idx in col_idx.items():
            if not name.startswith("force_aggregate_"):
                continue
            tag = name[len("force_aggregate_"):]   # 예: "R_Palm_Net_Force"
            mag = data[:, idx].astype(np.float32)
            normal_axes = {
                a: col_idx.get(f"normal_{tag}_{a}") for a in ("x", "y", "z")
            }
            if not all(v is not None for v in normal_axes.values()):
                logger.warning(
                    f"[replay] aggregate '{tag}' missing normal columns — skipping"
                )
                continue
            normal = np.stack(
                [data[:, normal_axes["x"]],
                 data[:, normal_axes["y"]],
                 data[:, normal_axes["z"]]],
                axis=1,
            ).astype(np.float32)
            self.aggregate[tag] = {
                "origin": origin if origin is not None else np.zeros((len(self.t), 3), dtype=np.float32),
                "mag": mag,
                "normal": normal,
            }

        logger.info(
            f"[replay] loaded {self.path.name}: T={len(self.t)}, "
            f"pos_joints={len(self.pos)}, torque_joints={len(self.torque)}, "
            f"ext_joint_torque_joints={len(self.ext_joint_torque)}, "
            f"pairs={len(self.pair_force_vec)}, aggregates={len(self.aggregate)}, "
            f"ext_force_topics={len(self.topic_force_vec)}, "
            f"ext_torque_topics={len(self.topic_torque_vec)}"
        )

    # ------------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        if len(self.t) == 0:
            return 0.0
        return float(self.t[-1] - self.t[0])

    @property
    def num_samples(self) -> int:
        return int(len(self.t))

    def index_at(self, t: float) -> int:
        """Largest sample index whose timestamp ≤ t. Clipped to [0, T-1]."""
        if len(self.t) == 0:
            return 0
        # searchsorted on the absolute time column. CSV t may not start at 0,
        # so caller is expected to add t[0] offset (or pass relative+t[0]).
        idx = int(np.searchsorted(self.t, t, side="right")) - 1
        if idx < 0:
            return 0
        if idx >= len(self.t):
            return len(self.t) - 1
        return idx
