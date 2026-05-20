"""Phase 2 v2 — UsdGeom.Points as child of /ALLEX/orbbec_link.

debug_draw 대신 정식 Points prim 사용. parent xform (orbbec_link 의 fabric worldMatrix) 을
Hydra renderer 가 자동 inherit 하므로 head 움직임이 자동 반영됨.

- 매 frame update callback 없음
- 점 갯수/위치/색 매번 바뀌어도 OK (`update(xyz, rgb)` 만 호출)
- session layer 에 author (CLAUDE.md USD layer 규약)

사용 예 (Script Editor):
    import sys, importlib
    sys.path.insert(0, "/home/asher12/workspace/isaacsim/_build/linux-x86_64/release/extsUser/allex_isaac_twin/tools")
    import pc2_reader, poc_pc_usd_points as pc
    importlib.reload(pc2_reader); importlib.reload(pc)

    pc.setup(parent_path="/ALLEX/orbbec_link", point_size=0.005)
    xyz_cam, rgb = pc2_reader.synthetic_wall(width=160, height=120, z=1.0, half_size=0.4)
    pc.update(xyz_cam, rgb)
    # trajectory replay 시작 → wall 이 head 따라 자동 이동/회전

    # 다른 frame 들어오면 size 달라도 OK:
    pc.update(new_xyz, new_rgb)

    pc.teardown()    # 완전 제거
"""

from __future__ import annotations

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

_prim_path: str | None = None
_points_prim = None
_disp_primvar = None


def _stage():
    return omni.usd.get_context().get_stage()


def _session_edit(stage):
    return Usd.EditContext(stage, stage.GetSessionLayer())


def setup(
    parent_path: str = "/ALLEX/orbbec_link",
    prim_name: str = "PCViz",
    point_size: float = 0.005,
    optical_frame: bool = True,
):
    """parent_path 아래 prim_name 으로 Points prim 생성 (session layer).

    optical_frame=True 면 ROS REP-103 standard quat 을 prim local xform 으로 author —
    PC 가 ROS optical 컨벤션 (+Z forward, +X right, +Y down) 인 채로 들어와도 정합 맞음.
    body-frame PC 면 False.
    """
    global _prim_path, _points_prim, _disp_primvar
    stage = _stage()
    if stage is None:
        print("[poc_pc_usd] no stage")
        return
    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        print(f"[poc_pc_usd] ERROR parent not found: {parent_path}")
        return

    _prim_path = f"{parent_path}/{prim_name}"
    with _session_edit(stage):
        # 이전 setup 잔재 정리 (orient op 등이 남아있을 수 있음)
        if stage.GetPrimAtPath(Sdf.Path(_prim_path)):
            stage.RemovePrim(Sdf.Path(_prim_path))

        points = UsdGeom.Points.Define(stage, Sdf.Path(_prim_path))
        points.CreatePointsAttr().Set(Vt.Vec3fArray())
        widths = points.CreateWidthsAttr()
        widths.Set(Vt.FloatArray([float(point_size)]))
        points.SetWidthsInterpolation(UsdGeom.Tokens.constant)

        if optical_frame:
            # ROS REP-103: optical → body rotation, q(w,x,y,z) = (0.5, -0.5, 0.5, -0.5)
            xform = UsdGeom.Xformable(points)
            xform.ClearXformOpOrder()
            op = xform.AddOrientOp()
            op.Set(Gf.Quatf(0.5, Gf.Vec3f(-0.5, 0.5, -0.5)))

        primvars = UsdGeom.PrimvarsAPI(points.GetPrim())
        _disp_primvar = primvars.CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        _disp_primvar.Set(Vt.Vec3fArray())

    _points_prim = points
    print(f"[poc_pc_usd] setup {_prim_path}, width={point_size}, "
          f"optical_frame={optical_frame}")


def update(xyz_cam, rgb=None, default_color=(1.0, 0.8, 0.0)):
    """현재 PC 갱신. N 가변. xyz_cam 은 cam-local frame (parent xform 자동 적용)."""
    if _points_prim is None or _disp_primvar is None:
        print("[poc_pc_usd] not set up, call setup() first")
        return
    stage = _stage()

    xyz = np.ascontiguousarray(np.asarray(xyz_cam, dtype=np.float32).reshape(-1, 3))
    n = xyz.shape[0]

    if rgb is None:
        rgb_np = np.tile(np.asarray(default_color, dtype=np.float32), (n, 1))
    else:
        rgb_np = np.clip(np.asarray(rgb, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
        rgb_np = np.ascontiguousarray(rgb_np)
        if rgb_np.shape[0] != n:
            print(f"[poc_pc_usd] WARN rgb size {rgb_np.shape[0]} != xyz size {n}")
            return

    with _session_edit(stage):
        _points_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz))
        _disp_primvar.Set(Vt.Vec3fArray.FromNumpy(rgb_np))


def clear():
    """빈 PC. prim 은 유지."""
    if _points_prim is None:
        return
    stage = _stage()
    with _session_edit(stage):
        _points_prim.GetPointsAttr().Set(Vt.Vec3fArray())
        _disp_primvar.Set(Vt.Vec3fArray())
    print("[poc_pc_usd] cleared (empty points)")


def teardown():
    """prim 자체 제거."""
    global _prim_path, _points_prim, _disp_primvar
    stage = _stage()
    if stage is not None and _prim_path is not None:
        with _session_edit(stage):
            stage.RemovePrim(Sdf.Path(_prim_path))
        print(f"[poc_pc_usd] teardown {_prim_path}")
    _prim_path = None
    _points_prim = None
    _disp_primvar = None
