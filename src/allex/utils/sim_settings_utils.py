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
    """Return World() init kwargs: physics_dt, rendering_dt, gravity.

    JSON 의 canonical key 는 ``physics_hz``. World() 가 요구하는 ``physics_dt``
    는 여기서 ``1.0 / physics_hz`` 로 변환해 함께 노출한다. 호환성을 위해 옛
    ``physics_dt`` 가 직접 있으면 그것을 우선.
    """
    raw = _load().get("world", {})
    out = dict(raw)
    if "physics_dt" not in out and "physics_hz" in out:
        hz = float(out["physics_hz"])
        if hz > 0:
            out["physics_dt"] = 1.0 / hz
    return out


def get_motor_step_per_ms() -> float:
    """Return the global motor-domain ramp step (per 1ms).

    실 제어기와 동일 — 모든 카테고리(kp/kv/tlp/tln) 단일 값. trajectory_player
    런타임에서 직접 사용하지는 않고, ``tools/gen_joint_step_sizes.py`` 가
    이 값을 source 로 ``joint_step_per_ms`` 블록을 생성한다.
    """
    section = _load().get("newton", {}).get("ramp_step_sizes", {}) or {}
    v = section.get("motor_step_per_ms", 0.0)
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


_JOINT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "joint_config.json"
)


def get_nominal_motor_gains() -> dict[str, dict[str, float]]:
    """Return per-joint motor-domain nominal PD gains and torque limit.

    Format::

        {
            "L_Shoulder_Pitch_Joint": {"kp": 3.0, "kv": 0.005, "trq_lim": 1.2},
            ...
        }

    Source: ``src/allex/config/joint_config.json::nominal_motor_gains``
    (extracted from ``trajectory/a_nominal_gain_group/*.csv``).

    Used by ``MotorStateMirror`` as the initial K_m_current for each motor —
    실 제어기가 부팅 시점에 가지는 모터-도메인 default 와 1:1 일치.
    누락 / 빈 dict → caller 가 fallback (정적 0 또는 안전 default) 결정.
    """
    try:
        with open(_JOINT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        print(f"[ALLEX][Cfg] failed to load joint_config.json: {exc}")
        return {}

    raw = cfg.get("nominal_motor_gains", {}) or {}
    out: dict[str, dict[str, float]] = {}
    for jname, vals in raw.items():
        if jname.startswith("_") or not isinstance(vals, dict):
            continue
        try:
            kp = float(vals.get("kp", 0.0))
            kv = float(vals.get("kv", 0.0))
            trq = float(vals.get("trq_lim", 0.0))
        except (TypeError, ValueError):
            continue
        out[jname] = {"kp": kp, "kv": kv, "trq_lim": trq}
    return out


def get_physics_hz() -> float:
    """Return canonical physics-loop frequency [Hz].

    JSON 의 ``world.physics_hz`` 를 우선 읽고, 옛 ``physics_dt`` 만 있을 땐
    ``1.0 / physics_dt`` 로 폴백. trajectory_generate 의 Hermite 샘플링 hz,
    UI fallback 모두 이 값을 단일 소스로 사용.
    """
    world = _load().get("world", {})
    if "physics_hz" in world:
        hz = float(world["physics_hz"])
        if hz > 0:
            return hz
    if "physics_dt" in world:
        dt = float(world["physics_dt"])
        if dt > 0:
            return 1.0 / dt
    raise RuntimeError(
        "physics_config.json::world 에 physics_hz / physics_dt 가 모두 없습니다"
    )


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
    _skip_keys = {"solver", "gravcomp", "ramp_step_sizes"}

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


def apply_actuator_gravcomp_to_builder(builder: Any) -> int:
    """Stamp `mujoco:actuatorgravcomp` on joints per `newton.gravcomp.actuator_gravcomp`.

    의미:
        false (default)  qfrc_gravcomp 가 별도 어레이로 누적 (passive). PD 와
                         보상이 분리되어 디버깅·로거 합산이 깔끔.
        true             qfrc_actuator 가 PD + 보상 합산을 모두 담당 (actuator).
                         실 모터가 보상까지 출력해야 하므로 `actuatorfrcrange`
                         한계 적용. 실 로봇 거동에 더 충실.

    config 형식:
        "actuator_gravcomp": false              # 모든 관절 default false (no stamp)
        "actuator_gravcomp": true               # 모든 관절에 true 인증
        "actuator_gravcomp": {                  # 일부 관절만 true
            "enabled": true,
            "joints": ["L_Shoulder_Pitch_Joint", ...]
        }
        "actuator_gravcomp": {                  # 전체 적용, 일부 관절만 제외
            "enabled": true,
            "exclude": ["Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint"]
        }

    Default false 는 stamping 자체를 생략 (MuJoCo default 가 false 이므로 동일).
    `apply_gravcomp_to_builder` 와 같은 finalize-pre 시점에 호출되어야 함.

    Returns: 인증된 joint 개수.
    """
    section = _load().get("newton", {}).get("gravcomp", {}) or {}
    routing = section.get("actuator_gravcomp", False)

    # Normalize bool / dict → (enabled, joint_filter, exclude_names)
    if isinstance(routing, bool):
        enabled = routing
        joint_filter: list[str] | None = None  # None = 모든 joint
        exclude_names: set[str] = set()
    elif isinstance(routing, dict):
        enabled = bool(routing.get("enabled", False))
        joints = list(routing.get("joints", []) or [])
        joint_filter = joints if joints else None
        exclude_names = set(routing.get("exclude", []) or [])
    else:
        print(f"[ALLEX][Gravcomp] invalid actuator_gravcomp type: {type(routing).__name__}; skipping")
        return 0

    if not enabled:
        return 0

    joint_labels = list(getattr(builder, "joint_label", []) or [])
    if not joint_labels:
        print("[ALLEX][Gravcomp] builder.joint_label empty; skipping actuator_gravcomp")
        return 0

    attrs = getattr(builder, "custom_attributes", None)
    attr = attrs.get("mujoco:actuatorgravcomp") if attrs is not None else None
    if attr is None:
        print("[ALLEX][Gravcomp] 'mujoco:actuatorgravcomp' custom attribute not registered "
              "on builder; is the active solver MuJoCo?")
        return 0
    if attr.values is None:
        attr.values = {}

    def _resolve(short_name: str) -> int | None:
        target = "/" + short_name
        for i, label in enumerate(joint_labels):
            if label == short_name or label.endswith(target):
                return i
        return None

    def _is_excluded(label: str) -> bool:
        return any(label == e or label.endswith("/" + e) for e in exclude_names)

    matched = 0
    missed: list[str] = []
    if joint_filter is None:
        for i, label in enumerate(joint_labels):
            if _is_excluded(label):
                continue
            attr.values[i] = True
            matched += 1
    else:
        for name in joint_filter:
            if name in exclude_names:
                continue
            idx = _resolve(name)
            if idx is None:
                missed.append(name)
                continue
            attr.values[idx] = True
            matched += 1

    scope = "all joints" if joint_filter is None else f"{matched}/{len(joint_filter)} joints"
    if exclude_names:
        scope += f" (excluding {sorted(exclude_names)})"
    print(f"[ALLEX][Gravcomp] actuator_gravcomp=true stamped on {scope}")
    if missed:
        print(f"[ALLEX][Gravcomp] unmatched joint names: {missed}")
    return matched


def apply_actuator_gravcomp_runtime() -> bool:
    """Runtime stamping for `mujoco:actuatorgravcomp` via `mjw_model.jnt_actgravcomp`.

    Newton 의 MuJoCo solver 가 USD custom_attribute 채널로 `mujoco:actuatorgravcomp`
    를 receive 하지 않아 (`builder.custom_attributes` 에 미등록), builder-time
    stamping 이 silently fail. 대신 finalize 후 활성화된 MuJoCo Warp model 의
    `jnt_actgravcomp: wp.array(dtype=int)` 어레이에 직접 1 을 써넣어 routing 을
    바꾼다 (mujoco_warp/_src/forward.py:778, passive.py:547 의 게이트가 이 어레이
    한 곳).

    호출 시점: solver/mjw_model 이 준비된 첫 physics step 시점. 그 전이면 None
    참조라 fail → False 반환 (caller 가 다음 step 에 재시도).

    Returns:
        True  — 적용 완료 (또는 disabled 라서 no-op) → caller 는 더 이상 재호출 X
        False — model 아직 미준비 → caller 가 다음 step 에 다시 시도

    NOTE: 매 step 호출하지 말 것 (CUDA graph 재기록 비용). 첫 성공 후 caller 가
    flag 로 막아야 함.
    """
    section = _load().get("newton", {}).get("gravcomp", {}) or {}
    routing = section.get("actuator_gravcomp", False)

    if isinstance(routing, bool):
        enabled = routing
        joint_filter: list[str] | None = None
        exclude_names: set[str] = set()
    elif isinstance(routing, dict):
        enabled = bool(routing.get("enabled", False))
        joints = list(routing.get("joints", []) or [])
        joint_filter = joints if joints else None
        exclude_names = set(routing.get("exclude", []) or [])
    else:
        return True  # invalid → no-op (not retry)

    if not enabled:
        return True  # disabled → no-op

    # Acquire mjw_model via Newton stage→solver
    try:
        from isaacsim.physics.newton import acquire_stage  # type: ignore[import-not-found]
        stage = acquire_stage()
        solver = getattr(stage, "solver", None) if stage is not None else None
        mjw_model = getattr(solver, "mjw_model", None) if solver is not None else None
    except Exception as exc:
        print(f"[ALLEX][Gravcomp] runtime acquire failed: {exc}")
        return False

    if mjw_model is None:
        return False  # not ready yet — retry next step

    arr = getattr(mjw_model, "jnt_actgravcomp", None)
    if arr is None:
        print("[ALLEX][Gravcomp] mjw_model.jnt_actgravcomp missing — cannot apply")
        return True  # not retryable

    import numpy as np
    arr_np = arr.numpy().copy()  # host copy, indexed by MuJoCo joint id (njnt,)

    # Newton joint_label[i] is at *Newton* joint id i; jnt_actgravcomp is indexed
    # by *MuJoCo* joint id. Map MuJoCo→Newton via solver.mjc_jnt_to_newton_jnt
    # (newton solver builds this when constructing the MuJoCo model).
    newton_model = getattr(solver, "model", None)
    joint_labels: list[str] = list(getattr(newton_model, "joint_label", []) or []) if newton_model else []
    mjc_to_newton_arr = getattr(solver, "mjc_jnt_to_newton_jnt", None)
    if mjc_to_newton_arr is None or not joint_labels:
        print("[ALLEX][Gravcomp] solver.mjc_jnt_to_newton_jnt or joint_label missing; retry next step")
        return False
    mjc_to_newton = mjc_to_newton_arr.numpy()  # (nworld, njnt)
    if mjc_to_newton.shape[0] == 0:
        return False
    mapping = mjc_to_newton[0]  # single-world ALLEX
    n_newton_joints = len(joint_labels)

    def _is_excluded(label: str) -> bool:
        return any(label == e or label.endswith("/" + e) for e in exclude_names)

    def _matches_filter(label: str) -> bool:
        if joint_filter is None:
            return True
        return any(label == n or label.endswith("/" + n) for n in joint_filter)

    matched = 0
    for mjc_jnt in range(arr_np.size):
        newton_jnt_global = int(mapping[mjc_jnt])
        if newton_jnt_global < 0:
            continue
        # multi-world: template index = global % template_size
        template_idx = newton_jnt_global % n_newton_joints
        label = joint_labels[template_idx]
        if _is_excluded(label):
            continue
        if not _matches_filter(label):
            continue
        arr_np[mjc_jnt] = 1
        matched += 1

    if joint_filter is None:
        scope = f"all joints ({matched})"
    else:
        scope = f"{matched}/{len(joint_filter)} joints"
    if exclude_names:
        scope += f" (excluding {sorted(exclude_names)})"
    missed: list[str] = []
    if joint_filter is not None:
        seen = {joint_labels[int(mapping[m]) % n_newton_joints]
                for m in range(arr_np.size) if int(mapping[m]) >= 0}
        missed = [n for n in joint_filter
                  if n not in exclude_names
                  and not any(s == n or s.endswith("/" + n) for s in seen)]

    try:
        arr.assign(arr_np)
    except Exception as exc:
        print(f"[ALLEX][Gravcomp] jnt_actgravcomp.assign failed: {exc}")
        return True

    print(f"[ALLEX][Gravcomp] runtime jnt_actgravcomp=1 applied on {scope}")
    if missed:
        print(f"[ALLEX][Gravcomp] unmatched joint names: {missed}")
    # Verification readback
    try:
        verify = arr.numpy()
        n_set = int((verify != 0).sum())
        print(f"[ALLEX][Gravcomp] readback: {n_set}/{verify.size} joints have jnt_actgravcomp=1")
    except Exception:
        pass
    return True


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
