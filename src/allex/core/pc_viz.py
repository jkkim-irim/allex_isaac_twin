"""UsdGeom.Points authoring for PointCloud2 replay.

Defines a single Points prim as a child of an existing xform (default
``/ALLEX/orbbec_link``) and exposes setup / update / clear / teardown. The
parent xform's fabric world matrix is inherited by Hydra automatically, so
the cloud follows the robot's head without per-step transform math.

All USD writes happen on the stage session layer (see CLAUDE.md — viz prim
edits must stay in session to avoid root/session strength conflicts).
"""
from __future__ import annotations

import contextlib
import logging

import numpy as np

logger = logging.getLogger("allex.core.pc_viz")


class PcViz:
    """Authors a single UsdGeom.Points prim under a parent xform."""

    def __init__(
        self,
        stage,
        parent_xform_path: str = "/ALLEX/orbbec_link",
        prim_name: str = "PCReplay",
        optical_frame: bool = True,
        point_size_m: float = 0.005,
        default_color=(1.0, 0.8, 0.0),
    ):
        self._stage = stage
        self._parent_path = parent_xform_path
        self._prim_name = prim_name
        self._optical_frame = bool(optical_frame)
        self._point_size = float(point_size_m)
        self._default_color = tuple(default_color)
        self._prim_path: str | None = None
        self._points = None
        self._disp_primvar = None

    # ------------------------------------------------------------------
    def _session_edit(self):
        if self._stage is None:
            return contextlib.nullcontext()
        try:
            from pxr import Usd
            return Usd.EditContext(self._stage, self._stage.GetSessionLayer())
        except Exception:
            return contextlib.nullcontext()

    # ------------------------------------------------------------------
    def setup(self) -> bool:
        """Define Points prim under parent. Returns True on success."""
        if self._stage is None:
            logger.warning("[pc_viz] no stage")
            return False
        from pxr import Gf, Sdf, UsdGeom, Vt

        parent = self._stage.GetPrimAtPath(self._parent_path)
        if not parent or not parent.IsValid():
            logger.warning(f"[pc_viz] parent prim not found: {self._parent_path}")
            return False

        self._prim_path = f"{self._parent_path}/{self._prim_name}"
        with self._session_edit():
            if self._stage.GetPrimAtPath(Sdf.Path(self._prim_path)):
                self._stage.RemovePrim(Sdf.Path(self._prim_path))

            points = UsdGeom.Points.Define(self._stage, Sdf.Path(self._prim_path))
            points.CreatePointsAttr().Set(Vt.Vec3fArray())
            widths = points.CreateWidthsAttr()
            widths.Set(Vt.FloatArray([self._point_size]))
            points.SetWidthsInterpolation(UsdGeom.Tokens.constant)

            if self._optical_frame:
                # ROS REP-103 optical → body, q(w,x,y,z) = (0.5, -0.5, 0.5, -0.5).
                xform = UsdGeom.Xformable(points)
                xform.ClearXformOpOrder()
                op = xform.AddOrientOp()
                op.Set(Gf.Quatf(0.5, Gf.Vec3f(-0.5, 0.5, -0.5)))

            primvars = UsdGeom.PrimvarsAPI(points.GetPrim())
            self._disp_primvar = primvars.CreatePrimvar(
                "displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.vertex,
            )
            self._disp_primvar.Set(Vt.Vec3fArray())

        self._points = points
        logger.info(
            f"[pc_viz] setup {self._prim_path} (width={self._point_size}, "
            f"optical_frame={self._optical_frame})"
        )
        return True

    # ------------------------------------------------------------------
    def update(self, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
        if self._points is None or self._disp_primvar is None:
            return
        from pxr import Vt

        xyz_np = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
        n = xyz_np.shape[0]
        if rgb is None:
            rgb_np = np.tile(np.asarray(self._default_color, dtype=np.float32), (n, 1))
        else:
            rgb_np = np.clip(np.asarray(rgb, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
            rgb_np = np.ascontiguousarray(rgb_np)
            if rgb_np.shape[0] != n:
                logger.warning(f"[pc_viz] rgb size {rgb_np.shape[0]} != xyz size {n}")
                return

        with self._session_edit():
            self._points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz_np))
            self._disp_primvar.Set(Vt.Vec3fArray.FromNumpy(rgb_np))

    # ------------------------------------------------------------------
    def clear(self) -> None:
        if self._points is None or self._disp_primvar is None:
            return
        from pxr import Vt
        with self._session_edit():
            self._points.GetPointsAttr().Set(Vt.Vec3fArray())
            self._disp_primvar.Set(Vt.Vec3fArray())

    # ------------------------------------------------------------------
    def teardown(self) -> None:
        if self._stage is None or self._prim_path is None:
            return
        from pxr import Sdf
        with self._session_edit():
            self._stage.RemovePrim(Sdf.Path(self._prim_path))
        logger.info(f"[pc_viz] teardown {self._prim_path}")
        self._prim_path = None
        self._points = None
        self._disp_primvar = None
