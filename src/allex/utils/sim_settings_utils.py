"""Loaders + appliers for `src/allex/config/physics_config.json`.

The JSON is the single source of truth for all physics overrides — Isaac
World init kwargs, NewtonConfig fields, MuJoCo solver fields, USD
``/physicsScene`` attrs, and the per-body gravity-compensation list. This
module reads it once at import (with :func:`reload` for dev) and exposes
appliers that map the dict onto the live engine objects.

JSON sections (mirrors NewtonConfig schema)::

    world                Isaac Sim World() init kwargs
    usd.physics_scene    UsdPhysics.Scene + PhysxSchema.PhysxSceneAPI attrs
    newton               isaacsim.physics.newton.NewtonConfig fields
    newton.solver        NewtonConfig.solver_cfg (MuJoCoSolverConfig)
    newton.gravcomp      Per-body gravcomp scale + body name list

Schema notes
------------
- ``newton.pd_scale``: keep at math.pi/180 (= 0.017453292519943295) to cancel
  the Newton USD importer's deg→rad multiplication on drive gains. Setting
  1.0 inflates SI gains by ~57.3×.
- ``newton.use_cuda_graph``: True. External per-step writes to *model* arrays
  re-record the graph and cause stutter; only *data*-array contents writes
  (e.g. ``qfrc_applied``) are graph-safe.
- ``newton.gravcomp.bodies``: at least one entry must be present at builder
  finalize time so MuJoCo's ``m.ngravcomp > 0`` and the qfrc_gravcomp kernel
  runs each step (build-time gate, mujoco_warp/_src/io.py). Body names are
  resolved against ``builder.body_label`` by exact match or
  ``endswith("/" + name)``.

Usage::

    from ..utils.sim_settings_utils import (
        get_world_settings,
        apply_newton_config,
        apply_gravcomp_to_builder,
        apply_usd_physics_scene,
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "physics_config.json"
)

_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    """Return the parsed JSON, caching after first read."""
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def reload() -> None:
    """Drop the cached config; the next access re-reads from disk."""
    global _cache
    _cache = None


# ---------------------------------------------------------------------------
# Section accessors
# ---------------------------------------------------------------------------

def get_world_settings() -> dict[str, Any]:
    """Return World() init kwargs: physics_dt, rendering_dt, gravity."""
    return _load().get("world", {})


# ---------------------------------------------------------------------------
# Newton cfg applier
# ---------------------------------------------------------------------------

def apply_newton_config(cfg_obj: Any) -> None:
    """Apply newton + newton.solver sections onto a NewtonConfig.

    Unknown attribute names are ignored (with a log line).
    """
    cfg = _load().get("newton", {})
    if not cfg:
        return

    solver_section = cfg.get("solver", {}) or {}

    # Sections consumed elsewhere (not as NewtonConfig attributes).
    _skip_keys = {"solver", "gravcomp"}

    for key, val in cfg.items():
        if key in _skip_keys:
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
# Newton ModelBuilder gravcomp applier
# ---------------------------------------------------------------------------

def apply_gravcomp_to_builder(builder: Any) -> int:
    """Stamp `mujoco:gravcomp` onto the builder for bodies in `newton.gravcomp.bodies`.

    Must be called after `add_usd()` (so `builder.body_label` is populated and the
    `mujoco:gravcomp` custom attribute is registered) and before `finalize()` (so
    MuJoCo's `m.ngravcomp` is computed correctly at `put_model` time).

    Returns the number of bodies actually matched and stamped.
    """
    section = _load().get("newton", {}).get("gravcomp", {}) or {}
    body_names = list(section.get("bodies", []) or [])
    scale = float(section.get("scale", 1.0))
    if not body_names or scale <= 0.0:
        return 0

    body_labels = list(getattr(builder, "body_label", []) or [])
    if not body_labels:
        print("[ALLEX][Gravcomp] builder.body_label empty; skipping")
        return 0

    # Bodies in USD typically land with the leaf-prim name (e.g.
    # "Waist_Yaw_Link"); keep the same match-by-suffix tolerance the equality
    # injector uses so labels with prefixes still resolve.
    def _resolve(short_name: str) -> int | None:
        target = "/" + short_name
        for i, label in enumerate(body_labels):
            if label == short_name or label.endswith(target):
                return i
        return None

    attrs = getattr(builder, "custom_attributes", None)
    attr = attrs.get("mujoco:gravcomp") if attrs is not None else None
    if attr is None:
        print("[ALLEX][Gravcomp] 'mujoco:gravcomp' custom attribute not registered "
              "on builder; is the active solver MuJoCo?")
        return 0
    if attr.values is None:
        attr.values = {}

    matched = 0
    missed: list[str] = []
    for name in body_names:
        idx = _resolve(name)
        if idx is None:
            missed.append(name)
            continue
        attr.values[idx] = scale
        matched += 1

    print(
        f"[ALLEX][Gravcomp] stamped scale={scale} on {matched}/{len(body_names)} bodies"
    )
    if missed:
        print(f"[ALLEX][Gravcomp] unmatched body names: {missed}")
    return matched


# ---------------------------------------------------------------------------
# USD /physicsScene applier
# ---------------------------------------------------------------------------

# Map: snake_case key → PhysxSchema.PhysxSceneAPI setter (common knobs only).
# (UsdPhysics.Scene attrs are handled inline in `apply_usd_physics_scene` for
# gravity_magnitude / gravity_direction; no dispatch table needed yet.)
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


def _find_physics_scene(stage) -> object | None:
    """Find the first UsdPhysics.Scene prim in the stage (any path)."""
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim
    return None


def apply_usd_physics_scene(stage, scene_path: str | None = None) -> int:
    """Apply usd.physics_scene + usd.physics_scene.physx to a stage prim.

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
