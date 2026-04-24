"""Load and apply ALLEX physics configuration from config/physics.toml.

Usage:
    from .config.physics_settings import (
        get_equality_tuning,
        get_armature_overrides,
        apply_newton_config,
        apply_usd_physics_scene,
    )

The config is loaded lazily on first access and cached. Sections absent from
the TOML simply fall back to whatever Isaac Sim / Newton / our bridge already
use as defaults — this file is an *opt-in override layer*, not a hard schema.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


_EXT_ROOT = Path(__file__).resolve().parent.parent.parent  # extension root
_CONFIG_PATH = _EXT_ROOT / "config" / "physics.toml"

_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    if not _CONFIG_PATH.exists():
        print(f"[ALLEX][Cfg] {_CONFIG_PATH} not found; using built-in defaults")
        _cache = {}
        return _cache
    try:
        with open(_CONFIG_PATH, "rb") as f:
            _cache = tomllib.load(f)
        print(f"[ALLEX][Cfg] loaded {_CONFIG_PATH}")
    except Exception as exc:
        print(f"[ALLEX][Cfg] failed to parse {_CONFIG_PATH}: {exc}; using defaults")
        _cache = {}
    return _cache


def reload() -> None:
    """Drop the cache so the next access re-reads physics.toml."""
    global _cache
    _cache = None


# ---------------------------------------------------------------------------
# Section accessors
# ---------------------------------------------------------------------------

def get_equality_tuning() -> dict[str, Any]:
    """Return the [allex.equality] section (minus armature_overrides subtable)."""
    eq = _load().get("allex", {}).get("equality", {})
    return {k: v for k, v in eq.items() if k != "armature_overrides"}


def get_armature_overrides() -> dict[str, float]:
    overrides = (
        _load().get("allex", {}).get("equality", {}).get("armature_overrides", {})
    )
    try:
        return {str(k): float(v) for k, v in overrides.items()}
    except Exception as exc:
        print(f"[ALLEX][Cfg] invalid armature_overrides: {exc}")
        return {}


def get_actuator_gravcomp_cfg() -> dict[str, Any]:
    """Return [allex.actuator_gravcomp] section with safe defaults.

    Keys:
        enabled (bool): 전역 on/off. 변경 후 stage reload 필요.
        joints (list[str]): 대상 joint short-name 목록. 빈 리스트 → 모든 active joint.
    """
    cfg = _load().get("allex", {}).get("actuator_gravcomp", {}) or {}
    enabled = bool(cfg.get("enabled", False))
    joints_raw = cfg.get("joints", []) or []
    try:
        joints = [str(x) for x in joints_raw]
    except Exception:
        joints = []
    return {"enabled": enabled, "joints": joints}


# ---------------------------------------------------------------------------
# Newton cfg applier
# ---------------------------------------------------------------------------

def apply_newton_config(cfg_obj: Any) -> None:
    """Apply [newton] and [newton.solver] TOML sections onto a NewtonConfig.

    Unknown attribute names in TOML are ignored (with a log line).
    """
    cfg = _load().get("newton", {})
    if not cfg:
        return

    solver_section = cfg.get("solver", {}) or {}

    for key, val in cfg.items():
        if key == "solver":
            continue
        if hasattr(cfg_obj, key):
            old = getattr(cfg_obj, key)
            setattr(cfg_obj, key, val)
            print(f"[ALLEX][Cfg] newton.{key}: {old} -> {val}")
        else:
            print(f"[ALLEX][Cfg] newton.{key} unknown on NewtonConfig; skipped")

    solver_cfg = getattr(cfg_obj, "solver_cfg", None)
    if solver_cfg is None or not solver_section:
        return
    for key, val in solver_section.items():
        if hasattr(solver_cfg, key):
            old = getattr(solver_cfg, key)
            setattr(solver_cfg, key, val)
            print(f"[ALLEX][Cfg] newton.solver.{key}: {old} -> {val}")
        else:
            print(f"[ALLEX][Cfg] newton.solver.{key} unknown; skipped")


# ---------------------------------------------------------------------------
# USD /physicsScene applier
# ---------------------------------------------------------------------------

# Map: TOML snake_case key → UsdPhysics.Scene setter + expected value type
_USD_SCENE_KEYS = {
    "gravity_magnitude": "CreateGravityMagnitudeAttr",
    "gravity_direction": "CreateGravityDirectionAttr",
}

# Map: TOML snake_case key → PhysxSchema.PhysxSceneAPI setter
# (Only common knobs — extend as needed.)
_PHYSX_SCENE_KEYS = {
    "friction_cone_type":          "CreateFrictionTypeAttr",
    "solver_type":                 "CreateSolverTypeAttr",
    "broadphase_type":             "CreateBroadphaseTypeAttr",
    "collision_system":            "CreateCollisionSystemAttr",
    "enable_ccd":                  "CreateEnableCCDAttr",
    "enable_stabilization":        "CreateEnableStabilizationAttr",
    "enable_enhanced_determinism": "CreateEnableEnhancedDeterminismAttr",
    "min_position_iteration_count": "CreateMinPositionIterationCountAttr",
    "max_position_iteration_count": "CreateMaxPositionIterationCountAttr",
    "min_velocity_iteration_count": "CreateMinVelocityIterationCountAttr",
    "max_velocity_iteration_count": "CreateMaxVelocityIterationCountAttr",
    "bounce_threshold":            "CreateBounceThresholdAttr",
    "friction_offset_threshold":   "CreateFrictionOffsetThresholdAttr",
    "time_steps_per_second":       "CreateTimeStepsPerSecondAttr",
}


def _find_physics_scene(stage) -> "object | None":
    """Find the first UsdPhysics.Scene prim in the stage (any path)."""
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim
    return None


def apply_usd_physics_scene(stage, scene_path: str | None = None) -> int:
    """Apply [usd.physics_scene] + [usd.physics_scene.physx] to a stage prim.

    If scene_path is None, auto-locate the first UsdPhysics.Scene prim.
    Returns 0 (and prints) if no scene exists yet.
    """
    section = _load().get("usd", {}).get("physics_scene", {})
    # Skip only if both section AND its physx subsection are empty.
    has_content = bool({k: v for k, v in section.items() if k != "physx"}) or bool(section.get("physx"))
    if not has_content:
        return 0
    try:
        from pxr import UsdPhysics, PhysxSchema, Gf
    except ImportError as exc:
        print(f"[ALLEX][Cfg] pxr not importable: {exc}")
        return 0

    if scene_path is None:
        prim = _find_physics_scene(stage)
        if prim is None:
            print("[ALLEX][Cfg] no UsdPhysics.Scene prim found in stage yet; "
                  "call apply_usd_physics_scene again after World.instance() / RUN")
            return 0
        scene_path = str(prim.GetPath())
        print(f"[ALLEX][Cfg] found physics scene at {scene_path}")
    else:
        prim = stage.GetPrimAtPath(scene_path)
        if not prim or not prim.IsValid():
            print(f"[ALLEX][Cfg] {scene_path} prim not found; skipping USD scene cfg")
            return 0

    applied = 0
    usd_scene = UsdPhysics.Scene(prim)

    # Gravity (UsdPhysics.Scene attrs)
    mag = section.get("gravity_magnitude")
    direction = section.get("gravity_direction")
    if mag is not None:
        usd_scene.CreateGravityMagnitudeAttr().Set(float(mag))
        applied += 1
        print(f"[ALLEX][Cfg] /physicsScene.gravity_magnitude = {mag}")
    if direction is not None:
        vec = Gf.Vec3f(*[float(x) for x in direction])
        usd_scene.CreateGravityDirectionAttr().Set(vec)
        applied += 1
        print(f"[ALLEX][Cfg] /physicsScene.gravity_direction = {tuple(vec)}")

    # PhysxSchema.PhysxSceneAPI attrs
    physx = section.get("physx", {}) or {}
    if physx:
        if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physx_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        else:
            physx_api = PhysxSchema.PhysxSceneAPI(prim)
        for key, val in physx.items():
            setter_name = _PHYSX_SCENE_KEYS.get(key)
            if setter_name is None:
                print(f"[ALLEX][Cfg] physx.{key} unmapped; skipped")
                continue
            try:
                attr = getattr(physx_api, setter_name)()
                attr.Set(val)
                applied += 1
                print(f"[ALLEX][Cfg] /physicsScene.physx.{key} = {val}")
            except Exception as exc:
                print(f"[ALLEX][Cfg] physx.{key} set failed: {exc}")

    return applied
