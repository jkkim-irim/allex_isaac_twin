"""Showcase CSV reader.

Parses a `showcase_*` style CSV (1 kHz log of joint pos/torque + per-pair
contact pos/force vectors + R_Palm aggregate net force) into per-channel
numpy arrays for kinematic replay overlay.

Column conventions (as produced by the showcase logging pipeline):
  time                                       — seconds
  pos_<joint>                                — rad. follower joints excluded.
  torque_<joint>                             — Nm. qfrc_actuator (subset of pos joints).
  contact_pos_<linkA> <-> <linkB>_{x,y,z}    — m, world frame.
  aggregate_origin_{x,y,z}                   — m, world frame.
  force_<linkA> <-> <linkB>                  — scalar, ignored (not loaded).
  force_aggregate_R_Palm_Net_Force           — scalar magnitude, N.
  force_vec_<linkA> <-> <linkB>_{x,y,z}      — N, world frame.
  normal_R_Palm_Net_Force_{x,y,z}            — unit normal.

Pair names contain spaces and "<->" separators. They are sanitized to
``"<linkA>__<linkB>"`` (USD-prim-path safe) and that sanitized form is the
key used everywhere downstream.
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

        # --- pos_<joint> / torque_<joint> ------------------------------
        self.pos: Dict[str, np.ndarray] = {}
        self.torque: Dict[str, np.ndarray] = {}
        for name, idx in col_idx.items():
            if name.startswith("pos_"):
                self.pos[name[len("pos_"):]] = data[:, idx].astype(np.float32)
            elif name.startswith("torque_"):
                self.torque[name[len("torque_"):]] = data[:, idx].astype(np.float32)

        # --- pair-keyed force_vec / contact_pos -----------------------
        # pair_force_vec / pair_contact_pos: sanitized_pair → (T, 3)
        self.pair_force_vec: Dict[str, np.ndarray] = {}
        self.pair_contact_pos: Dict[str, np.ndarray] = {}

        # First, group columns by raw pair name & axis.
        force_vec_axes: Dict[str, Dict[str, int]] = {}
        contact_pos_axes: Dict[str, Dict[str, int]] = {}
        for name, idx in col_idx.items():
            if name.startswith("force_vec_") and name[-2:] in ("_x", "_y", "_z"):
                axis = name[-1]
                raw_pair = name[len("force_vec_"):-2]
                force_vec_axes.setdefault(raw_pair, {})[axis] = idx
            elif name.startswith("contact_pos_") and name[-2:] in ("_x", "_y", "_z"):
                axis = name[-1]
                raw_pair = name[len("contact_pos_"):-2]
                contact_pos_axes.setdefault(raw_pair, {})[axis] = idx

        for raw_pair, axes in force_vec_axes.items():
            if not all(a in axes for a in ("x", "y", "z")):
                continue
            sanitized = sanitize_pair_name(raw_pair)
            arr = np.stack(
                [data[:, axes["x"]], data[:, axes["y"]], data[:, axes["z"]]],
                axis=1,
            ).astype(np.float32)
            self.pair_force_vec[sanitized] = arr

        for raw_pair, axes in contact_pos_axes.items():
            if not all(a in axes for a in ("x", "y", "z")):
                continue
            sanitized = sanitize_pair_name(raw_pair)
            arr = np.stack(
                [data[:, axes["x"]], data[:, axes["y"]], data[:, axes["z"]]],
                axis=1,
            ).astype(np.float32)
            self.pair_contact_pos[sanitized] = arr

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
            f"pairs={len(self.pair_force_vec)}, aggregates={len(self.aggregate)}"
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
