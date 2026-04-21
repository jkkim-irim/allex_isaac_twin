"""Inject MJCF-style equality constraints into Newton ModelBuilder.

Newton's USD importer does not load `<equality>` entries. We source them from
`src/joint_config.json` (regenerated from MJCF) and inject at finalize time by
patching `newton.ModelBuilder.finalize`.

MJCF/Newton convention: joint1 = poly(joint2). We store (follower, master) and
pass them to `add_equality_constraint_joint(joint1=follower, joint2=master, ...)`.
"""
from __future__ import annotations

import json
from pathlib import Path

import carb

_orig_finalize = None
_patched = False
_equality_table: list[tuple[str, str, list[float], str]] = []

_orig_on_post_reset = None
_articulation_patched = False


def _load_equality(joint_config_path: Path) -> list[tuple[str, str, list[float], str]]:
    with open(joint_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out = []
    for e in cfg.get("equality_constraints", []):
        if not e.get("active", True):
            continue
        out.append(
            (e["follower"], e["master"], list(e["polycoef"]), e.get("label", ""))
        )
    return out


def _find_joint_idx(builder, short_name: str) -> int | None:
    target = "/" + short_name
    for i, label in enumerate(builder.joint_label):
        if label == short_name or label.endswith(target):
            return i
    for i, label in enumerate(builder.joint_label):
        if short_name in label:
            return i
    return None


def _patch_equality_armature(builder, min_armature: float = 0.01) -> None:
    """Give passive joints referenced by equality a floor armature value.

    Per-joint overrides live in _ARMATURE_OVERRIDES for joints that carry
    significantly larger body loads than the default.
    """
    joint_qd_start = list(getattr(builder, "joint_qd_start", []) or [])
    joint_armature = getattr(builder, "joint_armature", None)
    if joint_armature is None or not joint_qd_start:
        return

    try:
        from ..config import physics_settings as _ps
        overrides = _ps.get_armature_overrides()
    except Exception as exc:
        print(f"[ALLEX][Eq] could not read armature overrides: {exc}")
        overrides = {}

    # Collect unique joint indices used by equality constraints (joint1 and joint2)
    jidx_set: set[int] = set()
    j1_list = list(getattr(builder, "equality_constraint_joint1", []) or [])
    j2_list = list(getattr(builder, "equality_constraint_joint2", []) or [])
    for j in j1_list + j2_list:
        try:
            ji = int(j)
            if ji >= 0:
                jidx_set.add(ji)
        except Exception:
            pass

    total_dof = int(getattr(builder, "joint_dof_count", 0) or 0)
    bumped = []
    for ji in sorted(jidx_set):
        if ji >= len(joint_qd_start):
            continue
        label = builder.joint_label[ji]
        short_name = label.rsplit("/", 1)[-1]
        target = overrides.get(short_name, min_armature)
        qd_start = int(joint_qd_start[ji])
        end = int(joint_qd_start[ji + 1]) if ji + 1 < len(joint_qd_start) else total_dof
        for dof_idx in range(qd_start, end):
            if dof_idx >= len(joint_armature):
                continue
            cur = float(joint_armature[dof_idx])
            if cur < target:
                joint_armature[dof_idx] = target
                bumped.append((ji, dof_idx, cur, target, label))

    if bumped:
        print(f"[ALLEX][Eq] bumped armature on {len(bumped)} dof(s) touched by equality:")
        for ji, dof_idx, old, new, lbl in bumped:
            print(f"   joint[{ji}] dof[{dof_idx}] armature {old} -> {new}  ({lbl})")


def _dump_model_equality_state(model) -> None:
    """After ModelBuilder.finalize(), confirm equality constraints landed in the Model."""
    # physics dt / frequency if available
    try:
        from isaacsim.physics.newton import acquire_stage as _acq
        st = _acq()
        if st is not None:
            freq = getattr(st, "physics_frequency", None)
            dt = getattr(st, "sim_dt", None)
            print(f"[ALLEX][Phys] physics_frequency={freq} sim_dt={dt}")
    except Exception as exc:
        print(f"[ALLEX][Phys] could not read physics freq: {exc}")

    count = getattr(model, "equality_constraint_count", 0) or 0
    print(f"[ALLEX][Eq] post-finalize Model.equality_constraint_count = {count}")
    if count == 0:
        return
    try:
        types = model.equality_constraint_type.numpy().tolist()
        j1 = model.equality_constraint_joint1.numpy().tolist()
        j2 = model.equality_constraint_joint2.numpy().tolist()
        poly = model.equality_constraint_polycoef.numpy().tolist()
        enabled = model.equality_constraint_enabled.numpy().tolist()
        labels = list(getattr(model, "equality_constraint_label", []) or [])
        for i in range(count):
            lbl = labels[i] if i < len(labels) else ""
            print(
                f"   eq[{i}] type={types[i]} joint1={j1[i]} joint2={j2[i]} "
                f"enabled={enabled[i]} label='{lbl}' polycoef={poly[i]}"
            )
    except Exception as exc:
        print(f"[ALLEX][Eq] could not dump equality arrays: {exc}")

    # custom attrs (mujoco.eq_solref / eq_solimp)
    mujoco_ns = getattr(model, "mujoco", None)
    if mujoco_ns is not None:
        for attr_name in ("eq_solref", "eq_solimp"):
            val = getattr(mujoco_ns, attr_name, None)
            if val is None:
                continue
            try:
                arr = val.numpy()
                print(f"   mujoco.{attr_name} shape={arr.shape} first={arr[0] if len(arr) else '?'}")
            except Exception:
                print(f"   mujoco.{attr_name} present (opaque)")


def _joint_type_name(t) -> str:
    try:
        from newton._src.sim.enums import JointType
        try:
            return JointType(int(t)).name
        except Exception:
            return str(t)
    except Exception:
        return str(t)


def _inject(builder) -> None:
    if not _equality_table:
        return
    # dump builder joint layout once so we can cross-check indices
    n_joints = len(builder.joint_label)
    joint_type_list = list(getattr(builder, "joint_type", []) or [])
    joint_qd_start_list = list(getattr(builder, "joint_qd_start", []) or [])
    total_dof = int(getattr(builder, "joint_dof_count", 0) or 0)
    # per-joint dof count = qd_start[i+1] - qd_start[i] (last one closes at total_dof)
    per_joint_dof: list[int] = []
    for i in range(len(joint_qd_start_list)):
        start_i = int(joint_qd_start_list[i])
        end_i = int(joint_qd_start_list[i + 1]) if i + 1 < len(joint_qd_start_list) else total_dof
        per_joint_dof.append(end_i - start_i)
    print(
        f"[ALLEX][Eq] builder: joints={n_joints} bodies={len(builder.body_label)} "
        f"total_dof={total_dof}"
    )
    if n_joints < 200:
        for i, lbl in enumerate(builder.joint_label):
            t = _joint_type_name(joint_type_list[i]) if i < len(joint_type_list) else "?"
            qds = joint_qd_start_list[i] if i < len(joint_qd_start_list) else "?"
            dc = per_joint_dof[i] if i < len(per_joint_dof) else "?"
            print(f"   [{i:3d}] type={t:<12s} qd_start={str(qds):<4} dof={str(dc):<2} label={lbl}")

    # MuJoCo equality constraint softness tuned for Isaac Sim runtime physics
    # frequency (observed 50 Hz → dt = 20 ms). MuJoCo needs timeconst >= 2*dt
    # to avoid explicit-integration overshoot in the constraint correction.
    # solref=(timeconst, dampratio); solimp=(dmin, dmax, width, midpoint, power).
    stiff_solref = None
    stiff_solimp = None
    soft_solref = None
    soft_solimp = None
    try:
        import warp as wp
        from ..config import physics_settings as _ps
        tuning = _ps.get_equality_tuning()
        stiff_solref = wp.vec2(
            float(tuning.get("default_solref_timeconst", 0.05)),
            float(tuning.get("default_solref_dampratio", 1.0)),
        )
        stiff_solimp = tuple(tuning.get("default_solimp", (0.9, 0.95, 0.001, 0.5, 2.0)))
        soft_solref = wp.vec2(
            float(tuning.get("waist_solref_timeconst", 0.15)),
            float(tuning.get("waist_solref_dampratio", 1.0)),
        )
        soft_solimp = tuple(tuning.get("waist_solimp", (0.7, 0.8, 0.002, 0.5, 2.0)))
    except Exception as exc:
        print(f"[ALLEX][Eq] solref/solimp config load failed: {exc}")

    added = 0
    missed: list[tuple[str, str]] = []
    for follower, master, coef, label in _equality_table:
        f = _find_joint_idx(builder, follower)
        m = _find_joint_idx(builder, master)
        if f is None or m is None:
            missed.append((follower, master))
            continue
        is_waist = "Waist" in follower or "Waist" in master
        custom_attrs = None
        if stiff_solref is not None:
            if is_waist and soft_solref is not None:
                custom_attrs = {"mujoco:eq_solref": soft_solref, "mujoco:eq_solimp": soft_solimp}
            else:
                custom_attrs = {"mujoco:eq_solref": stiff_solref, "mujoco:eq_solimp": stiff_solimp}
        try:
            builder.add_equality_constraint_joint(
                joint1=f,
                joint2=m,
                polycoef=coef,
                label=f"allex_{label}" if label else None,
                enabled=True,
                custom_attributes=custom_attrs,
            )
        except Exception as exc:
            print(f"[ALLEX][Eq] add_equality_constraint_joint failed ({follower} <- {master}): {exc}")
            # fallback without custom_attrs
            try:
                builder.add_equality_constraint_joint(
                    joint1=f, joint2=m, polycoef=coef,
                    label=f"allex_{label}" if label else None, enabled=True,
                )
            except Exception as exc2:
                print(f"[ALLEX][Eq] fallback also failed: {exc2}")
                missed.append((follower, master))
                continue
        print(
            f"[ALLEX][Eq] + follower='{follower}'(#{f})  master='{master}'(#{m})  "
            f"polycoef={coef}"
        )
        added += 1
    if added:
        print(f"[ALLEX] injected {added}/{len(_equality_table)} equality constraints")
    if missed:
        print(f"[ALLEX] unmatched equality joints: {missed}")

    # Small rotor-inertia floor on passive joints touched by equality
    # constraints — mild stabilization without over-damping the response.
    try:
        _patch_equality_armature(builder, min_armature=0.01)
    except Exception as exc:
        print(f"[ALLEX][Eq] armature patch failed: {exc}")


def _install_articulation_reset_patch() -> None:
    """Avoid Articulation.set_gains device mismatch on Newton backend.

    Isaac Sim's Articulation._on_post_reset calls set_gains(kps=self._default_kps, ...),
    which internally does move_data(kps, device="cpu") while get_dof_stiffnesses()
    returns CUDA tensors under Newton — causing a device-mismatch RuntimeError
    during world.reset_async(). Passing kps=None, kds=None takes the safe branch
    that reads stiffnesses from the physics view on the matching device.
    """
    global _orig_on_post_reset, _articulation_patched
    if _articulation_patched:
        return
    try:
        from isaacsim.core.prims.impl.articulation import Articulation
        from isaacsim.core.prims.impl.xform_prim import XFormPrim
    except ImportError as exc:
        print(f"[ALLEX] cannot import Articulation for reset patch: {exc}")
        return

    _orig_on_post_reset = Articulation._on_post_reset

    def _patched_on_post_reset(self, event):
        XFormPrim._on_post_reset(self, event)
        Articulation.set_joint_positions(self, self._default_joints_state.positions)
        Articulation.set_joint_velocities(self, self._default_joints_state.velocities)
        Articulation.set_joint_efforts(self, self._default_joints_state.efforts)
        Articulation.set_gains(self, kps=None, kds=None)

    Articulation._on_post_reset = _patched_on_post_reset
    _articulation_patched = True
    print("[ALLEX] patched Articulation._on_post_reset for Newton device consistency")


def _uninstall_articulation_reset_patch() -> None:
    global _orig_on_post_reset, _articulation_patched
    if not _articulation_patched:
        return
    try:
        from isaacsim.core.prims.impl.articulation import Articulation
    except ImportError:
        _articulation_patched = False
        return
    if _orig_on_post_reset is not None:
        Articulation._on_post_reset = _orig_on_post_reset
    _orig_on_post_reset = None
    _articulation_patched = False


def configure_newton_from_toml() -> bool:
    """Apply [newton] and [newton.solver] sections from config/physics.toml.

    Key use: pd_scale = π/180 cancels Newton USD importer's implicit deg→rad
    drive-gain multiplication so USD-authored SI gains pass through unchanged.

    Returns True if the override was applied, False otherwise.  Call this
    multiple times if needed — it's idempotent.
    """
    print("[ALLEX][Cfg] configure_newton_from_toml() called")
    try:
        from isaacsim.physics.newton import acquire_stage
        from ..config import physics_settings as _ps
    except ImportError as exc:
        print(f"[ALLEX][Cfg] cannot import Newton stage/config: {exc}")
        return False
    stage = acquire_stage()
    if stage is None:
        print("[ALLEX][Cfg] Newton stage not yet available (will retry at LOAD)")
        return False
    cfg = getattr(stage, "cfg", None)
    if cfg is None:
        print("[ALLEX][Cfg] Newton stage has no cfg; skipping override")
        return False
    before_pd = getattr(cfg, "pd_scale", None)
    _ps.apply_newton_config(cfg)
    after_pd = getattr(cfg, "pd_scale", None)
    print(f"[ALLEX][Cfg] cfg.pd_scale verify: before={before_pd} after={after_pd}")

    # Dump solver cfg so user can verify what's actually in effect
    solver_cfg = getattr(cfg, "solver_cfg", None)
    if solver_cfg is not None:
        try:
            print(f"[ALLEX][Cfg] solver_cfg (actual runtime values):")
            keys = [
                "solver_type", "iterations", "ls_iterations", "solver",
                "integrator", "cone", "impratio", "tolerance", "ls_tolerance",
                "use_mujoco_cpu", "use_mujoco_contacts", "disable_contacts",
                "njmax", "nconmax",
            ]
            for k in keys:
                if hasattr(solver_cfg, k):
                    print(f"   solver_cfg.{k} = {getattr(solver_cfg, k)}")
        except Exception as exc:
            print(f"[ALLEX][Cfg] solver_cfg dump failed: {exc}")
    return True


def _configure_newton_from_toml() -> None:
    configure_newton_from_toml()


def install(joint_config_path: Path) -> None:
    global _orig_finalize, _patched, _equality_table
    _install_articulation_reset_patch()
    _configure_newton_from_toml()
    if _patched:
        return
    try:
        import newton
    except ImportError:
        print("[ALLEX] newton not importable; skipping equality constraint hook")
        return
    try:
        _equality_table = _load_equality(joint_config_path)
    except Exception as exc:
        print(f"[ALLEX] failed to load equality table from {joint_config_path}: {exc}")
        _equality_table = []
        return
    print(
        f"[ALLEX] loaded {len(_equality_table)} equality entries from {joint_config_path}"
    )
    _orig_finalize = newton.ModelBuilder.finalize

    def _patched_finalize(self, *args, **kwargs):
        try:
            _inject(self)
        except Exception as exc:
            print(f"[ALLEX] equality injection raised: {exc}")
        model = _orig_finalize(self, *args, **kwargs)
        try:
            _dump_model_equality_state(model)
        except Exception as exc:
            print(f"[ALLEX][Eq] post-finalize dump failed: {exc}")
        return model

    newton.ModelBuilder.finalize = _patched_finalize
    _patched = True


def uninstall() -> None:
    global _patched, _orig_finalize
    _uninstall_articulation_reset_patch()
    if not _patched:
        return
    try:
        import newton
    except ImportError:
        _patched = False
        return
    if _orig_finalize is not None:
        newton.ModelBuilder.finalize = _orig_finalize
    _orig_finalize = None
    _patched = False
