"""Phase 2 PoC — camera (orbbec_link) frame PC → world frame 매 frame redraw.

USD 의 link world transform 가 매 physics step 마다 갱신되므로, update event 안에서
ComputeLocalToWorldTransform 호출하면 현재 robot 자세가 반영된 PC 가 그려진다.

사용 예 (Script Editor):
    import sys, importlib
    sys.path.insert(0, "/home/asher12/workspace/isaacsim/_build/linux-x86_64/release/extsUser/allex_isaac_twin/tools")
    import pc2_reader, poc_pc_head_frame as head
    importlib.reload(pc2_reader); importlib.reload(head)

    # camera (orbbec_link) frame 의 합성 PC — z=1m 앞 정면 1m square wall
    xyz_cam, rgb = pc2_reader.synthetic_wall(width=160, height=120, z=1.0, half_size=0.4)
    head.enable(xyz_cam, rgb, link_name="orbbec_link", point_size=4.0)

    # robot 움직이기:
    #   - Traj Studio UI 에서 trajectory replay 시작, 또는
    #   - Isaac Sim ▶ 후 ROS2 미러링 켜고 실제 로봇 움직임
    # → PC 가 head 움직임 따라 같이 회전·평행이동 해야 OK.

    head.disable()
"""

from __future__ import annotations

import numpy as np

import omni.kit.app
import omni.usd
import usdrt
from isaacsim.util.debug_draw import _debug_draw

import pc2_reader as r


_draw = _debug_draw.acquire_debug_draw_interface()

_subscription = None
_xyz_cam: np.ndarray | None = None
_rgb: np.ndarray | None = None
_link_prim_path: str | None = None
_rt_stage = None
_point_size: float = 4.0


def _find_prim_by_name(stage, name: str):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def _cam_world_T() -> np.ndarray | None:
    """Newton FabricManager 가 매 step writeback 하는 omni:fabric:worldMatrix 에서 live pose 읽음.

    USD `ComputeLocalToWorldTransform(Default)` 은 bind pose 만 줘서 안 됨.
    """
    if _rt_stage is None or _link_prim_path is None:
        return None
    rt_prim = _rt_stage.GetPrimAtPath(usdrt.Sdf.Path(_link_prim_path))
    if not rt_prim:
        return None
    attr = rt_prim.GetAttribute("omni:fabric:worldMatrix")
    if not attr or not attr.IsValid():
        return None
    mat = attr.Get()
    if mat is None:
        return None
    # usdrt.Gf.Matrix4d — row-major, row-vector convention (p_row @ M).
    # 4x4 numpy 로 추출 후 transpose → standard column-vector T (T @ p_col).
    arr = np.array([[float(mat[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    return arr.T


def _on_update(_event):
    if _xyz_cam is None:
        return
    T = _cam_world_T()
    if T is None:
        return
    xyz_w = r.apply_se3(_xyz_cam, T)
    pts = [tuple(map(float, p)) for p in xyz_w]
    if _rgb is not None:
        cols = [(float(c[0]), float(c[1]), float(c[2]), 1.0) for c in _rgb]
    else:
        cols = [(1.0, 0.8, 0.0, 1.0)] * len(pts)
    szs = [_point_size] * len(pts)
    _draw.clear_points()
    _draw.draw_points(pts, cols, szs)


def enable(
    xyz_cam: np.ndarray,
    rgb: np.ndarray | None = None,
    link_name: str = "orbbec_link",
    link_prim_path: str | None = None,
    point_size: float = 4.0,
):
    """xyz_cam (Nx3, camera frame) 을 link 의 world transform 으로 매 frame 변환.

    link_prim_path 우선, 없으면 link_name 으로 stage 검색.
    """
    global _subscription, _xyz_cam, _rgb, _link_prim_path, _rt_stage, _point_size
    disable()

    _xyz_cam = np.asarray(xyz_cam, dtype=np.float32).reshape(-1, 3)
    if rgb is not None:
        _rgb = np.clip(np.asarray(rgb, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
    else:
        _rgb = None
    _point_size = float(point_size)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[poc_head_pc] ERROR no stage")
        return

    if link_prim_path:
        prim = stage.GetPrimAtPath(link_prim_path)
        if not prim or not prim.IsValid():
            print(f"[poc_head_pc] ERROR prim not found at {link_prim_path}")
            return
    else:
        prim = _find_prim_by_name(stage, link_name)
        if prim is None:
            print(f"[poc_head_pc] ERROR no prim named {link_name!r} in stage")
            return

    _link_prim_path = prim.GetPath().pathString

    # USDRT attach — Newton fabric live transform 읽기용
    try:
        stage_id = omni.usd.get_context().get_stage_id()
        _rt_stage = usdrt.Usd.Stage.Attach(stage_id)
    except Exception as exc:
        print(f"[poc_head_pc] ERROR usdrt stage attach failed: {exc}")
        return

    # fabric attribute 진단 — 없으면 Newton update_fabric 비활성 등 의심
    rt_prim = _rt_stage.GetPrimAtPath(usdrt.Sdf.Path(_link_prim_path))
    if rt_prim:
        attr = rt_prim.GetAttribute("omni:fabric:worldMatrix")
        has_attr = bool(attr and attr.IsValid())
        print(f"[poc_head_pc] tracking {_link_prim_path} (fabric worldMatrix={'OK' if has_attr else 'MISSING'})")
    else:
        print(f"[poc_head_pc] WARN usdrt prim missing at {_link_prim_path}")

    app = omni.kit.app.get_app()
    _subscription = app.get_update_event_stream().create_subscription_to_pop(
        _on_update, name="allex_poc_pc_head_frame"
    )
    print(f"[poc_head_pc] enabled — {_xyz_cam.shape[0]} pts in cam frame, "
          f"size={_point_size}")


def disable():
    global _subscription, _xyz_cam, _rgb, _link_prim_path, _rt_stage
    if _subscription is not None:
        _subscription.unsubscribe()
        _subscription = None
    _draw.clear_points()
    _xyz_cam = None
    _rgb = None
    _link_prim_path = None
    _rt_stage = None
    print("[poc_head_pc] disabled")


def find_fabric_links(filter_substr: str = ""):
    """name match + omni:fabric:worldMatrix 보유 여부 함께 dump.

    fabric=YES 인 prim 만 live transform 가능. Newton update_fabric 대상.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[poc_head_pc] no stage")
        return
    try:
        stage_id = omni.usd.get_context().get_stage_id()
        rt_stage = usdrt.Usd.Stage.Attach(stage_id)
    except Exception as exc:
        print(f"[poc_head_pc] usdrt attach failed: {exc}")
        return
    hits = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if filter_substr.lower() not in name.lower():
            continue
        path = prim.GetPath().pathString
        rt_prim = rt_stage.GetPrimAtPath(usdrt.Sdf.Path(path))
        has = False
        if rt_prim:
            attr = rt_prim.GetAttribute("omni:fabric:worldMatrix")
            has = bool(attr and attr.IsValid())
        hits.append((path, has))
    for path, has in hits[:200]:
        print(f"  {path}  fabric={'YES' if has else 'no'}")
    print(f"[poc_head_pc] {len(hits)} prim(s) match {filter_substr!r}, "
          f"{sum(1 for _, h in hits if h)} with fabric worldMatrix")


def list_links(filter_substr: str = ""):
    """stage 안 prim 이름 dump — 정확한 link_name 모를 때 확인용."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[poc_head_pc] no stage")
        return
    hits = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if filter_substr.lower() in name.lower():
            hits.append(prim.GetPath().pathString)
    for p in hits[:200]:
        print(p)
    print(f"[poc_head_pc] {len(hits)} prim(s) match {filter_substr!r}")
