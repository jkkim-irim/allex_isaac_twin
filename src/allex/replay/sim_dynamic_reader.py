"""Sim dynamic (long-format) reader.

Sim 측 새 데이터 형식 — 디렉토리 안에 세 개의 CSV 가 분리돼 있음:
    contact.csv         (long  : t, pair, fx, fy, fz, f_mag)
    joint_position.csv  (wide  : t, <joint_full_name>, ...)
    joint_torque.csv    (wide  : t, <joint_full_name>, ...)

`ShowcaseReader` 와 동일 public attribute 를 제공해서 `csv_replayer.CsvReplayer`
가 sim/real 구분 없이 동일 코드 경로로 받게 함.

Long-format contact:
    - 한 timestep 당 최대 1 개 pair 활성 (또는 ``pair=="None"``).
    - pair 이름 = ``<linkA>_<linkB>`` 형태 (e.g. ``L_Elbow_R_Palm``).
    - origin 정보 없음 → viz_scenario_config 의 ``force_origin`` 으로 link 지정,
      또는 push 단계에서 pair 의 suffix (`_(L|R)_Palm`) 로 자동 매핑.
    - reader 는 wide-pivot 만 수행하고 contact_pos 는 빈 dict 유지.

Wide joint CSV:
    - 컬럼 이름이 곧 ``joint_full_name`` (예: ``Waist_Yaw_Joint``). ShowcaseReader 의
      short-form 컨벤션 (``pos_<short>``) 과 달리 직접 매칭 가능.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .showcase_reader import sanitize_pair_name


logger = logging.getLogger("allex.replay.sim_dynamic")


REQUIRED_FILES: tuple[str, ...] = (
    "contact.csv",
    "joint_position.csv",
    "joint_torque.csv",
)


def _load_wide_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """wide CSV (t, col_1, col_2, ...) → (t_array, {col_name: array}).

    빈 셀 / 비숫자 셀은 NaN. ShowcaseReader 와 같은 NaN-tolerant parse.
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty csv: {path}") from exc
        rows = list(reader)

    if not rows:
        raise ValueError(f"csv has no data rows: {path}")

    header = [h.strip() for h in header]
    if not header or header[0].lower() != "t":
        raise ValueError(f"{path}: first column must be 't', got header={header[:3]}")

    n_rows = len(rows)
    n_cols = len(header)
    data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row[:n_cols]):
            if cell == "" or cell is None:
                continue
            try:
                data[i, j] = float(cell)
            except ValueError:
                pass

    t = data[:, 0].astype(np.float64)
    cols = {
        name: data[:, j].astype(np.float32)
        for j, name in enumerate(header)
        if j > 0 and name
    }
    return t, cols


def _load_contact_csv(path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """long CSV → {pair: [(t, fx, fy, fz), ...] sorted by t}.

    ``pair=="None"`` 또는 빈 row 는 무시.
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty csv: {path}") from exc
        header = [h.strip() for h in header]
        required = ["t", "pair", "fx", "fy", "fz"]
        for col in required:
            if col not in header:
                raise ValueError(
                    f"{path}: missing column {col!r} in header {header}"
                )
        idx = {col: header.index(col) for col in required}

        out: dict[str, list[tuple[float, float, float, float]]] = {}
        for row in reader:
            if not row or len(row) <= idx["fz"]:
                continue
            pair = row[idx["pair"]].strip()
            if not pair or pair == "None":
                continue
            try:
                t = float(row[idx["t"]])
                fx = float(row[idx["fx"]])
                fy = float(row[idx["fy"]])
                fz = float(row[idx["fz"]])
            except ValueError:
                continue
            out.setdefault(pair, []).append((t, fx, fy, fz))

    return out


class SimDynamicReader:
    """Long-format sim 데이터 디렉토리 reader.

    `ShowcaseReader` 와 같은 public attribute 제공:
        path, t, pos, torque, pair_force_vec, pair_contact_pos,
        topic_force_vec, topic_torque_vec, topic_contact_pos,
        ext_joint_torque, aggregate, duration_s, num_samples, index_at()

    Real-only 채널 (topic_*, ext_joint_torque, aggregate) 은 빈 dict —
    csv_replayer 가 자동 fallback.
    """

    def __init__(self, dir_path: Path):
        self.path = Path(dir_path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"sim dynamic dir missing: {self.path}")

        missing = [f for f in REQUIRED_FILES if not (self.path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"sim dynamic dir partial: {self.path} — missing {missing}. "
                f"필수 파일: {list(REQUIRED_FILES)}"
            )

        # --- joint_position.csv (wide) — master time grid ---
        t_pos, pos_cols = _load_wide_csv(self.path / "joint_position.csv")
        self.t: np.ndarray = t_pos
        self.pos: Dict[str, np.ndarray] = pos_cols

        # --- joint_torque.csv (wide) — t grid 검증 ---
        t_tor, tor_cols = _load_wide_csv(self.path / "joint_torque.csv")
        if len(t_tor) != len(self.t) or not np.allclose(
            t_tor[: min(len(t_tor), len(self.t))],
            self.t[: min(len(t_tor), len(self.t))],
            atol=1e-6,
        ):
            logger.warning(
                f"[sim_dynamic] joint_torque.csv t grid differs from "
                f"joint_position.csv ({len(t_tor)} vs {len(self.t)}). "
                f"Using master from joint_position; mismatched samples may misalign."
            )
        self.torque: Dict[str, np.ndarray] = tor_cols

        # --- contact.csv (long → wide pivot) — pair_force_vec ---
        # 각 pair 마다 N×3 zeros 배열. 활성 row 의 t 위치에 (fx, fy, fz) 채움.
        self.pair_force_vec: Dict[str, np.ndarray] = {}
        contact_rows_by_pair = _load_contact_csv(self.path / "contact.csv")
        n = len(self.t)
        for pair_raw, rows in contact_rows_by_pair.items():
            if not rows:
                continue
            ts = np.array([r[0] for r in rows], dtype=np.float64)
            fx = np.array([r[1] for r in rows], dtype=np.float32)
            fy = np.array([r[2] for r in rows], dtype=np.float32)
            fz = np.array([r[3] for r in rows], dtype=np.float32)
            indices = np.searchsorted(self.t, ts, side="left")
            valid = (indices >= 0) & (indices < n)
            arr = np.zeros((n, 3), dtype=np.float32)
            arr[indices[valid], 0] = fx[valid]
            arr[indices[valid], 1] = fy[valid]
            arr[indices[valid], 2] = fz[valid]
            self.pair_force_vec[sanitize_pair_name(pair_raw)] = arr

        # --- contact_pos 정보는 long CSV 에 없음 — viz 단계에서 link origin 사용 ---
        self.pair_contact_pos: Dict[str, np.ndarray] = {}

        # --- real-only 채널 (sim 에 없음) — 빈 dict 으로 채워 csv_replayer 가 자동 skip ---
        self.topic_force_vec: Dict[str, np.ndarray] = {}
        self.topic_torque_vec: Dict[str, np.ndarray] = {}
        self.topic_contact_pos: Dict[str, np.ndarray] = {}
        self.ext_joint_torque: Dict[str, np.ndarray] = {}
        self.aggregate: Dict[str, Dict[str, np.ndarray]] = {}

        logger.info(
            f"[replay] loaded {self.path.name} (sim_dynamic): T={len(self.t)}, "
            f"pos_joints={len(self.pos)}, torque_joints={len(self.torque)}, "
            f"pairs={len(self.pair_force_vec)}"
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
        idx = int(np.searchsorted(self.t, t, side="right")) - 1
        if idx < 0:
            return 0
        if idx >= len(self.t):
            return len(self.t) - 1
        return idx
