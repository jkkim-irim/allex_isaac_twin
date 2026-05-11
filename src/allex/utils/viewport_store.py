"""Viewport camera 슬롯 저장/복원.

사용자가 viewport 를 원하는 각도/줌으로 맞춘 뒤 이름 붙여 저장해두고, 다른 슬롯과
자유롭게 전환할 수 있게 한다. Active viewport camera (기본
``/OmniverseKit_Persp``) 의 world transform 을 (eye, target) 페어로 추출해
JSON 으로 직렬화하고, 복원 시 ``isaacsim.core.utils.viewports.set_camera_view``
로 카메라 xformOp 들을 재설정. Up vector 는 Isaac Sim 표준 +Z 로 고정.

저장 위치: ``<ext_root>/data/saved_viewports.json``
포맷: ``{"slots": {"<name>": {"eye": [x,y,z], "target": [x,y,z]}, ...}}``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pxr import Gf, Usd, UsdGeom
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view

from .constants import DEFAULT_CAMERA_PRIM_PATH


_STORE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "saved_viewports.json"
)


def _load_store() -> dict:
    if not _STORE_PATH.exists():
        return {"slots": {}}
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[viewport_store] failed to read {_STORE_PATH}: {exc}")
        return {"slots": {}}
    if not isinstance(data, dict):
        return {"slots": {}}
    slots = data.get("slots")
    if not isinstance(slots, dict):
        data["slots"] = {}
    return data


def _save_store(data: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_eye_target(camera_prim_path: str) -> Optional[tuple]:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        return None
    xformable = UsdGeom.Xformable(prim)
    m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    eye = (float(t[0]), float(t[1]), float(t[2]))
    # USD 카메라는 local -Z 방향을 바라본다. world forward = m.TransformDir(-Z).
    fwd = m.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    fl = max(fwd.GetLength(), 1e-9)
    fwd = fwd / fl
    # target 거리는 임의 (set_camera_view 는 eye/target 방향만 사용). 1m 로 고정.
    target = (eye[0] + float(fwd[0]), eye[1] + float(fwd[1]), eye[2] + float(fwd[2]))
    return eye, target


def list_viewports() -> list:
    return sorted(_load_store()["slots"].keys())


def save_viewport(name: str, camera_prim_path: str = DEFAULT_CAMERA_PRIM_PATH) -> bool:
    name = (name or "").strip()
    if not name:
        return False
    res = _get_eye_target(camera_prim_path)
    if res is None:
        return False
    eye, target = res
    store = _load_store()
    store["slots"][name] = {"eye": list(eye), "target": list(target)}
    _save_store(store)
    return True


def load_viewport(name: str, camera_prim_path: str = DEFAULT_CAMERA_PRIM_PATH) -> bool:
    slot = _load_store()["slots"].get(name)
    if not isinstance(slot, dict):
        return False
    try:
        eye = tuple(float(v) for v in slot["eye"])
        target = tuple(float(v) for v in slot["target"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[viewport_store] malformed slot '{name}': {exc}")
        return False
    set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)
    return True


def delete_viewport(name: str) -> bool:
    store = _load_store()
    if name in store["slots"]:
        del store["slots"][name]
        _save_store(store)
        return True
    return False
