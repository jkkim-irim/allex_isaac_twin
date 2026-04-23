"""Per-step kp/kd/tau_ff CSV reader for runtime PD gain + feedforward injection.

CSV format:
    time, kp_<dof_name>, kd_<dof_name>, tau_ff_<dof_name>

Column names use ``articulation.dof_names`` directly. Each ``kp_``/``kd_``/
``tau_ff_`` column produces a per-DOF override for the matching joint; DOFs
without a matching column are left untouched (mask=False) by the controller.

Sample indexing is aligned with TrajectoryPlayer._sample_idx via
``get_sample(sample_idx)``; the caller is responsible for advancing the index
at the same pace as the physics step.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .joint_name_map import ALLEX_CSV_JOINT_NAMES


@dataclass(frozen=True)
class GainFFSample:
    """Per-step override snapshot.

    Each array has shape (num_dof,). The *_mask arrays mark which DOFs carry a
    meaningful value; masked-off entries hold 0.0 as a neutral placeholder and
    must not be applied by the controller.
    """
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray
    kp_mask: np.ndarray
    kd_mask: np.ndarray
    ff_mask: np.ndarray


def _follower_joint_names() -> set[str]:
    """Return the set of joint names that appear in ALLEX_CSV_JOINT_NAMES.

    Columns for these joints are allowed; columns outside this map are
    tolerated but logged once. Newton equality followers (e.g. IP/DIP/
    Upper_Pitch) are NOT in this map and will be warned on.
    """
    active: set[str] = set()
    for names in ALLEX_CSV_JOINT_NAMES.values():
        active.update(names)
    return active


class GainFFCsvReader:
    def __init__(
        self,
        csv_path: Path,
        dof_names: list[str],
    ) -> None:
        self._csv_path = Path(csv_path)
        self._dof_names = list(dof_names)
        self._num_dof = len(dof_names)
        self._name_to_idx = {n: i for i, n in enumerate(dof_names)}

        self._kp: np.ndarray | None = None   # (n_samples, num_dof)
        self._kd: np.ndarray | None = None
        self._tau_ff: np.ndarray | None = None
        # Per-DOF masks are constant across the whole CSV (column presence).
        self._kp_mask: np.ndarray = np.zeros(self._num_dof, dtype=bool)
        self._kd_mask: np.ndarray = np.zeros(self._num_dof, dtype=bool)
        self._ff_mask: np.ndarray = np.zeros(self._num_dof, dtype=bool)
        self._n_samples: int = 0

        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        if not self._csv_path.exists():
            # Debug-level: CSV is optional, absence is normal.
            print(f"[ALLEX][GainFF] csv not found: {self._csv_path} (skip)")
            return
        try:
            with open(self._csv_path, "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                rows: list[list[str]] = list(reader)
        except Exception as exc:
            print(f"[ALLEX][GainFF] failed to read {self._csv_path}: {exc}")
            return
        if len(rows) < 2:
            print(f"[ALLEX][GainFF] {self._csv_path} has no data rows")
            return

        header = [c.strip() for c in rows[0]]
        data_rows = rows[1:]

        kp_cols: list[tuple[int, int]] = []   # (col_idx, dof_idx)
        kd_cols: list[tuple[int, int]] = []
        ff_cols: list[tuple[int, int]] = []
        ignored: list[str] = []
        followers: list[str] = []
        active_set = _follower_joint_names()

        for col_idx, name in enumerate(header):
            if name in ("time", "duration"):
                continue
            if name.startswith("tau_ff_"):
                dof_name = name[len("tau_ff_"):]
                dof_idx = self._name_to_idx.get(dof_name)
                if dof_idx is None:
                    ignored.append(name)
                    continue
                if dof_name not in active_set:
                    followers.append(dof_name)
                    continue
                ff_cols.append((col_idx, dof_idx))
                continue
            prefix, sep, dof = name.partition("_")
            if not sep:
                ignored.append(name)
                continue
            if prefix in ("kp", "kd"):
                dof_name = dof
                dof_idx = self._name_to_idx.get(dof_name)
                if dof_idx is None:
                    ignored.append(name)
                    continue
                if dof_name not in active_set:
                    followers.append(dof_name)
                    continue
                (kp_cols if prefix == "kp" else kd_cols).append((col_idx, dof_idx))
            else:
                ignored.append(name)

        if ignored:
            print(f"[ALLEX][GainFF] ignored columns (unknown dof or format): {sorted(set(ignored))}")
        if followers:
            print(f"[ALLEX][GainFF] follower joint columns ignored (equality-driven): {sorted(set(followers))}")

        if not (kp_cols or kd_cols or ff_cols):
            print(f"[ALLEX][GainFF] no kp_/kd_/tau_ff_ columns found in {self._csv_path.name}; nothing to apply")
            return

        n = len(data_rows)
        # Masked-off entries stay at 0.0 — controller skips them via the mask,
        # so the placeholder value is irrelevant.
        kp_buf = np.zeros((n, self._num_dof), dtype=np.float32)
        kd_buf = np.zeros((n, self._num_dof), dtype=np.float32)
        ff_buf = np.zeros((n, self._num_dof), dtype=np.float32)

        for r, row in enumerate(data_rows):
            if not row:
                continue
            for col_idx, dof_idx in kp_cols:
                if col_idx < len(row):
                    try:
                        kp_buf[r, dof_idx] = float(row[col_idx])
                    except ValueError:
                        pass
            for col_idx, dof_idx in kd_cols:
                if col_idx < len(row):
                    try:
                        kd_buf[r, dof_idx] = float(row[col_idx])
                    except ValueError:
                        pass
            for col_idx, dof_idx in ff_cols:
                if col_idx < len(row):
                    try:
                        ff_buf[r, dof_idx] = float(row[col_idx])
                    except ValueError:
                        pass

        kp_mask = np.zeros(self._num_dof, dtype=bool)
        kd_mask = np.zeros(self._num_dof, dtype=bool)
        ff_mask = np.zeros(self._num_dof, dtype=bool)
        for _c, dof_idx in kp_cols:
            kp_mask[dof_idx] = True
        for _c, dof_idx in kd_cols:
            kd_mask[dof_idx] = True
        for _c, dof_idx in ff_cols:
            ff_mask[dof_idx] = True

        self._kp = kp_buf
        self._kd = kd_buf
        self._tau_ff = ff_buf
        self._kp_mask = kp_mask
        self._kd_mask = kd_mask
        self._ff_mask = ff_mask
        self._n_samples = n
        print(
            f"[ALLEX][GainFF] {self._csv_path.name} loaded: "
            f"kp={len(kp_cols)} kd={len(kd_cols)} ff={len(ff_cols)} samples={n}"
        )

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return self._n_samples > 0 and bool(
            self._kp_mask.any() or self._kd_mask.any() or self._ff_mask.any()
        )

    @property
    def num_samples(self) -> int:
        return self._n_samples

    def get_sample(self, sample_idx: int) -> GainFFSample | None:
        """Return a GainFFSample for the given sample index, or None if not ready.

        Holds the last row when sample_idx exceeds num_samples.
        """
        if (
            self._kp is None
            or self._kd is None
            or self._tau_ff is None
            or self._n_samples == 0
        ):
            return None
        last = self._n_samples - 1
        idx = max(0, min(int(sample_idx), last))
        return GainFFSample(
            kp=self._kp[idx].copy(),
            kd=self._kd[idx].copy(),
            tau_ff=self._tau_ff[idx].copy(),
            kp_mask=self._kp_mask.copy(),
            kd_mask=self._kd_mask.copy(),
            ff_mask=self._ff_mask.copy(),
        )
