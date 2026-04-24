"""ALLEX physics configuration (in-tree Python).

All physics-related overrides live in this module. Edit the ``_CONFIG`` dict
below and disable/re-enable the ALLEX extension in Isaac Sim's Extension
Manager to reload.

Usage:
    from .config.physics_settings import (
        get_equality_tuning,
        get_armature_overrides,
        apply_newton_config,
        apply_usd_physics_scene,
    )

Sections (mirrors the previous TOML schema):
    usd.physics_scene             USD /physicsScene prim (UsdPhysics.Scene)
    usd.physics_scene.physx       PhysxSceneAPI attrs (PhysX-only; Newton ignores)
    newton                        isaacsim.physics.newton.NewtonConfig
    newton.solver                 NewtonConfig.solver_cfg (MuJoCoSolverConfig)
    allex.equality                MJCF equality constraint tuning
    allex.equality.armature_overrides
                                  per-joint armature boosts

Keys absent from ``_CONFIG`` fall back to whatever Isaac Sim / Newton / our
bridge already use as defaults — this module is an *opt-in override layer*,
not a hard schema.
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Active configuration
# ---------------------------------------------------------------------------
# To override a value the engine would otherwise pick, add it under the
# matching section. Removing / commenting a key restores the engine default.

_CONFIG: dict[str, Any] = {
    # USD /physicsScene — UsdPhysics.Scene + PhysxSceneAPI
    # ---------------------------------------------------------------------
    # Leave the dicts empty (or omit keys) to keep Isaac Sim defaults.
    # Available keys are listed in _USD_SCENE_KEYS / _PHYSX_SCENE_KEYS below.
    "usd": {
        "physics_scene": {
            # "gravity_magnitude": 9.81,
            # "gravity_direction": [0.0, 0.0, -1.0],
            "physx": {
                # NOTE: PhysX-specific. Newton ignores these but they are still
                # authored on the USD stage (useful if falling back to PhysX).
                # "friction_cone_type": "patch",     # "patch" | "oneDirection"
                # "solver_type": "PGS",              # "PGS" | "TGS"
                # "broadphase_type": "SAP",          # "SAP" | "MBP"
                # "collision_system": "PCM",
                # "enable_ccd": False,
                # "enable_stabilization": True,
                # "enable_enhanced_determinism": False,
                # "min_position_iteration_count": 1,
                # "max_position_iteration_count": 255,
                # "min_velocity_iteration_count": 1,
                # "max_velocity_iteration_count": 255,
                # "bounce_threshold": 2.0,
                # "friction_offset_threshold": 0.04,
                # "time_steps_per_second": 50,
            },
        },
    },

    # Newton Physics — isaacsim.physics.newton.NewtonConfig
    # ---------------------------------------------------------------------
    # Reference: isaacsim.physics.newton.impl.newton_config.NewtonConfig
    # Note: "capture_graph_physics_step" / "auto_switch_on_startup" are carb
    # kit settings (under exts."isaacsim.physics.newton".*), NOT NewtonConfig
    # fields — they must be set in kit/settings.toml, not here.
    "newton": {
        # Cancel Newton USD importer's implicit deg→rad drive-gain multiplication.
        # USD drive stiffness/damping are SI (N·m/rad). Setting pd_scale = π/180
        # makes the importer's (1 / DegreesToRadian / pd_scale) factor equal to
        # 1.0. Leaving at default 1.0 would scale SI values by ~57.3× (broken on
        # MJCF-style USD exports).
        "pd_scale": math.pi / 180.0,

        # Common NewtonConfig fields (uncomment to override):
        # "num_substeps": 1,
        # "debug_mode": False,
        # CUDA graph capture must stay OFF while the trajectory player issues
        # set_dof_stiffnesses / set_dof_dampings / set_dof_max_forces at via
        # events — every external write breaks the captured graph and forces a
        # re-record, surfacing as visible per-event stutter.
        "use_cuda_graph": True,
        # "time_step_app": True,
        # "physics_frequency": 600.0,        # Hz (None = use USD timeStepsPerSecond)
        # "update_fabric": True,
        # "disable_physx_fabric_tracker": True,
        # "collapse_fixed_joints": False,
        # "fix_missing_xform_ops": True,
        # "contact_ke": 1.0e4,               # contact stiffness (N/m)
        # "contact_kd": 1.0e2,               # contact damping
        # "contact_kf": 1.0e1,               # contact friction stiffness
        # "contact_mu": 1.0,                 # coefficient of friction
        # "contact_ka": 0.5,
        # "restitution": 0.0,
        # "contact_margin": 0.01,
        # "soft_contact_margin": 0.01,

        # Newton Solver — MuJoCoSolverConfig (MuJoCo Warp backend)
        # Reference: isaacsim.physics.newton.impl.solver_config.MuJoCoSolverConfig
        "solver": {
            # "solver_type": "mujoco",       # "xpbd" | "mujoco" | "featherstone" | "semiImplicit"
            # "njmax": 1200,                 # max constraints / world
            # "nconmax": 200,                # max contact points / world
            # "iterations": 100,             # solver iterations
            # "ls_iterations": 15,           # line search iterations
            # "solver": "newton",            # "newton" (sparse LDL) | "cg"
            # "integrator": "implicitfast",  # "euler" | "rk4" | "implicit" | "implicitfast"
            # "cone": "elliptic",            # "pyramidal" | "elliptic"
            # "impratio": 1.0,
            # "use_mujoco_cpu": False,
            # "use_mujoco_contacts": True,
            # "disable_contacts": False,
            # "tolerance": 1.0e-8,
            # "ls_tolerance": 0.001,
            # "ls_parallel": True,
            # "update_data_interval": 1,
            # "include_sites": False,
            # "save_to_mjcf": "",            # non-empty path: dump generated MJCF
        },
    },

    # ALLEX — MJCF equality constraint tuning
    # ---------------------------------------------------------------------
    "allex": {
        "equality": {
            # Default tuning (used by finger coupling constraints).
            # solref = (timeconst [s], dampratio)
            "default_solref_timeconst": 0.02,
            "default_solref_dampratio": 1.0,
            # solimp = (dmin, dmax, width, midpoint, power)
            "default_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],

            # Softer tuning for the waist 4-bar linkage carrying chest + arms + head.
            "waist_solref_timeconst": 0.02,
            "waist_solref_dampratio": 1.0,
            "waist_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],

            # Minimum rotor inertia (armature) applied to every joint that
            # participates in an equality constraint and currently has an
            # armature below this value.
            "default_armature": 0.01,

            # Per-joint armature. Applied when the builder's current value is
            # smaller. Heavy-load waist joints need significantly more rotor
            # inertia for stability.
            "armature_overrides": {
                "Waist_Upper_Pitch_Joint": 2.0,
                "Waist_Pitch_Dummy_Joint": 2.0,
                "Waist_Lower_Pitch_Joint": 0.5,
            },
        },
    },
}


def _load() -> dict[str, Any]:
    return _CONFIG


def reload() -> None:
    """No-op kept for API compatibility (config is now a Python literal)."""
    return


# ---------------------------------------------------------------------------
# Section accessors
# ---------------------------------------------------------------------------

def get_equality_tuning() -> dict[str, Any]:
    """Return the allex.equality section (minus armature_overrides subtable)."""
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

# Map: snake_case key → UsdPhysics.Scene setter
_USD_SCENE_KEYS = {
    "gravity_magnitude": "CreateGravityMagnitudeAttr",
    "gravity_direction": "CreateGravityDirectionAttr",
}

# Map: snake_case key → PhysxSchema.PhysxSceneAPI setter
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
