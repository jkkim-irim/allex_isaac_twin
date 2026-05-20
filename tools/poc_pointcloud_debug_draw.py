"""Phase 0 PoC — Isaac Sim debug_draw 로 point cloud rendering 가능한지 확인.

사용법 (Isaac Sim Script Editor):
    import sys, importlib
    sys.path.insert(0, "/home/asher12/workspace/isaacsim/_build/linux-x86_64/release/extsUser/allex_isaac_twin/tools")
    import poc_pointcloud_debug_draw as poc
    importlib.reload(poc)

    poc.enable_once()       # 단발 draw — debug_draw persistent 인지 확인
    # 안 보이면:
    poc.enable_loop()       # 매 frame redraw (update subscription)
    # 정리:
    poc.disable()

점이 보이는지 + 어느 모드에서 보이는지가 다음 step (Phase 1) API 선택에 영향.
"""

from __future__ import annotations

import math
import random

from isaacsim.util.debug_draw import _debug_draw

try:
    import omni.kit.app
except ImportError:
    omni = None

_draw = _debug_draw.acquire_debug_draw_interface()
_subscription = None
_points: list[tuple[float, float, float]] = []
_colors: list[tuple[float, float, float, float]] = []
_sizes: list[float] = []


def _make_sphere(n, center, radius, point_size):
    cx, cy, cz = center
    pts, cols, szs = [], [], []
    rng = random.Random(0)
    for _ in range(n):
        u = rng.random()
        v = rng.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(2.0 * v - 1.0)
        x = cx + radius * math.sin(phi) * math.cos(theta)
        y = cy + radius * math.sin(phi) * math.sin(theta)
        z = cz + radius * math.cos(phi)
        t = (math.cos(phi) + 1.0) * 0.5
        pts.append((x, y, z))
        cols.append((1.0 - t, 0.2, t, 1.0))
        szs.append(float(point_size))
    return pts, cols, szs


def _draw_buffer():
    if _points:
        _draw.draw_points(_points, _colors, _sizes)


def _on_update(_event):
    _draw.clear_points()
    _draw_buffer()


def _populate(n, center, radius, point_size):
    global _points, _colors, _sizes
    _points, _colors, _sizes = _make_sphere(n, center, radius, point_size)
    print(f"[poc_pc] interface={_draw!r}")
    print(f"[poc_pc] populated {len(_points)} pts, sample={_points[0]}, "
          f"color={_colors[0]}, size={_sizes[0]}")


def enable_once(n: int = 500, center=(0.0, 0.0, 1.0), radius: float = 0.3, point_size: float = 20.0):
    """단발 draw — debug_draw persistent 인지 검증."""
    disable()
    _populate(n, center, radius, point_size)
    _draw.draw_points(_points, _colors, _sizes)
    print(f"[poc_pc] enable_once — 1-shot draw, {n} pts @ {center} r={radius} size={point_size}")


def enable_loop(n: int = 500, center=(0.0, 0.0, 1.0), radius: float = 0.3, point_size: float = 20.0):
    """매 frame redraw — persistent 안 될 때 fallback."""
    global _subscription
    disable()
    _populate(n, center, radius, point_size)
    app = omni.kit.app.get_app()
    _subscription = app.get_update_event_stream().create_subscription_to_pop(
        _on_update, name="allex_poc_pointcloud_debug_draw"
    )
    print(f"[poc_pc] enable_loop — per-frame redraw, {n} pts @ {center} r={radius} size={point_size}")


def disable():
    global _subscription, _points, _colors, _sizes
    if _subscription is not None:
        _subscription.unsubscribe()
        _subscription = None
    _draw.clear_points()
    _draw.clear_lines()
    _points, _colors, _sizes = [], [], []
    print("[poc_pc] disabled")


def show_array(xyz, rgb=None, point_size: float = 5.0, default_color=(1.0, 0.8, 0.0, 1.0)):
    """numpy Nx3 xyz (+ optional Nx3 rgb in [0,1]) → debug_draw.

    persistent 으로 한 번 그리고 끝. 갱신은 disable() + show_array() 다시 호출.
    """
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    pts = [tuple(map(float, p)) for p in xyz]
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
        rgb = np.clip(rgb, 0.0, 1.0)
        cols = [(float(r), float(g), float(b), 1.0) for r, g, b in rgb]
    else:
        cols = [tuple(default_color)] * len(pts)
    szs = [float(point_size)] * len(pts)

    disable()
    _draw.draw_points(pts, cols, szs)
    print(f"[poc_pc] show_array — {len(pts)} pts, size={point_size}, "
          f"colored={rgb is not None}")


def draw_axes_line():
    """추가 진단 — 1m 빨간선 (0,0,0)→(1,0,0). 점이 안 보일 때 line 도 안 보이면 viewport/interface 자체 문제."""
    _draw.draw_lines(
        [(0.0, 0.0, 0.0)],
        [(1.0, 0.0, 0.0)],
        [(1.0, 0.0, 0.0, 1.0)],
        [3.0],
    )
    print("[poc_pc] draw_axes_line — red line (0,0,0)->(1,0,0)")
