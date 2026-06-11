"""PointCloud2 frame-sequence replayer synced to CsvReplayer timeline.

Loads a manifest.json produced by ``tools/extract_pointcloud2.py``, exposes
``advance(t_rel)`` which nearest-snaps to the closest dumped frame and pushes
its xyz/rgb into a ``PcViz`` Points prim. Heavy npz loads are cached (small
LRU) to avoid re-reading the same frame across consecutive ticks.

The replayer is driven by ``CsvReplayer``: lifecycle (start/advance/stop) is
delegated. No standalone subscription.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..core.pc_viz import PcViz

logger = logging.getLogger("allex.replay.pc")

_CACHE_MAX = 10


class PcReplayer:
    def __init__(
        self,
        manifest_path: Path | str,
        parent_xform_path: str = "/ALLEX/orbbec_link",
        optical_frame: bool = True,
        point_size_m: float = 0.005,
    ):
        self._manifest_path = Path(manifest_path)
        self._parent_xform_path = parent_xform_path
        self._optical_frame = bool(optical_frame)
        self._point_size = float(point_size_m)

        self._frames: list[dict] = []
        self._t_rel: np.ndarray = np.zeros(0, dtype=np.float64)
        self._dir: Path = self._manifest_path.parent
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        self._last_idx: int = -1
        self._pc_viz: PcViz | None = None
        self._paused: bool = False
        self._started: bool = False

        self._load_manifest()

    # ------------------------------------------------------------------
    def _load_manifest(self) -> None:
        if not self._manifest_path.is_file():
            logger.warning(f"[pc_replay] manifest not found: {self._manifest_path}")
            return
        try:
            with open(self._manifest_path) as f:
                m = json.load(f)
        except Exception as exc:
            logger.warning(f"[pc_replay] manifest load failed: {exc}")
            return
        self._frames = list(m.get("frames", []))
        if not self._frames:
            logger.warning("[pc_replay] manifest has no frames")
            return
        self._t_rel = np.asarray(
            [float(fr["t_rel_s"]) for fr in self._frames], dtype=np.float64
        )
        logger.info(
            f"[pc_replay] manifest loaded: {len(self._frames)} frames, "
            f"duration={float(m.get('duration_s', 0.0)):.2f}s, "
            f"voxel={m.get('voxel_m')}m, frame_id={m.get('frame_id')!r}"
        )

    # ------------------------------------------------------------------
    def _load_frame(self, idx: int):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        npz_name = self._frames[idx]["npz"]
        path = self._dir / npz_name
        try:
            d = np.load(path)
            xyz = d["xyz"]
            rgb = d["rgb"] if "rgb" in d.files else None
        except Exception as exc:
            logger.warning(f"[pc_replay] frame {idx} ({npz_name}) load fail: {exc}")
            return None
        self._cache[idx] = (xyz, rgb)
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
        return xyz, rgb

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._frames:
            return
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        except Exception as exc:
            logger.warning(f"[pc_replay] stage fetch failed: {exc}")
            return
        self._pc_viz = PcViz(
            stage,
            parent_xform_path=self._parent_xform_path,
            optical_frame=self._optical_frame,
            point_size_m=self._point_size,
        )
        if not self._pc_viz.setup():
            self._pc_viz = None
            return
        self._last_idx = -1
        self._paused = False
        self._started = True
        # 첫 프레임 즉시 push.
        first = self._load_frame(0)
        if first is not None:
            xyz, rgb = first
            self._pc_viz.update(xyz, rgb)
            self._last_idx = 0
        logger.info(f"[pc_replay] start ({len(self._frames)} frames)")

    # ------------------------------------------------------------------
    def advance(self, t_rel: float) -> None:
        if not self._started or self._paused or self._pc_viz is None:
            return
        if self._t_rel.size == 0:
            return
        # np.searchsorted → 가장 가까운 (왼쪽 우선) 프레임 선택.
        i = int(np.searchsorted(self._t_rel, t_rel, side="right")) - 1
        if i < 0:
            i = 0
        elif i >= self._t_rel.size:
            i = self._t_rel.size - 1
        if i == self._last_idx:
            return
        frame = self._load_frame(i)
        if frame is None:
            return
        xyz, rgb = frame
        self._pc_viz.update(xyz, rgb)
        self._last_idx = i

    # ------------------------------------------------------------------
    def pause(self, paused: bool = True) -> None:
        self._paused = bool(paused)

    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self._pc_viz is not None:
            try:
                self._pc_viz.teardown()
            except Exception as exc:
                logger.debug(f"[pc_replay] teardown warn: {exc}")
            self._pc_viz = None
        self._cache.clear()
        self._last_idx = -1
        self._started = False
        logger.info("[pc_replay] stop")
