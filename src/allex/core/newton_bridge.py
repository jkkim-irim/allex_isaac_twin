"""Inject MJCF-style equality constraints into Newton ModelBuilder.

Newton's USD importer does not load `<equality>` entries. We source them from
`src/allex/config/joint_config.json` (regenerated from MJCF) and inject at
finalize time by patching `newton.ModelBuilder.finalize`.

MJCF/Newton convention: joint1 = poly(joint2). We store (follower, master) and
pass them to `add_equality_constraint_joint(joint1=follower, joint2=master, ...)`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import carb

_orig_finalize = None
_equality_table: list[tuple[str, str, list[float], str]] = []

# Sentinel attribute 이름 — 패치된 함수 자체에 박아 두고 재패치 가드로 사용.
# Module-level `_patched` flag 는 extension reload 시 module 새로 로드되어 False 로
# 리셋되지만 newton.ModelBuilder.finalize 는 이미 patched 상태라 두 번째 install 이
# patched-of-patched chain 을 만들 수 있다. 함수 객체에 직접 attribute 부착하면
# reload 와 무관하게 정확히 검출.
_FINALIZE_SENTINEL = "_allex_finalize_patched"
_ARTICULATION_RESET_SENTINEL = "_allex_on_post_reset_patched"

# Top-level MJCF softness from the active joint_config.json. Forwarded to
# Newton/MuJoCo as `custom_attributes={"mujoco:eq_solref", "mujoco:eq_solimp"}`
# on every equality constraint in this table. None → omit the attribute and
# let Newton/MuJoCo apply its own default.
_config_solref: Optional[tuple[float, float]] = None
_config_solimp: Optional[tuple[float, float, float, float, float]] = None

# Path installed at extension startup — used to restore the "master" (ALLEX full
# body) equality table when ALLEX loads after a Hysteresis session.
_master_joint_config_path: Path | None = None

# Per-DOF static/dynamic friction loaded from joint_config.json::joint_friction.
_friction_table: dict[str, float] = {}

# Per-joint actuator-gravcomp toggle config (mujoco:jnt_actgravcomp custom attr).
# Independent from the body-level mujoco:gravcomp set by sim_settings_utils.
_gravcomp_cfg: dict = {"enabled": False, "joints": []}

_orig_on_post_reset = None


def set_equality_config(path: Path | None) -> None:
    """Swap the equality table consumed by the next `ModelBuilder.finalize()`.

    `path=None` restores the master (ALLEX full-body) table that was loaded at
    `install()` time. Scenarios with a reduced joint set (e.g., Hysteresis)
    pass their own joint_config.json so the injection logs show a matched
    count (no `unmatched: [L_*, Waist_*]` noise) and the table semantically
    reflects only what the active scene needs.

    Must be called *before* Newton's add_usd finalize for the current scene.
    """
    global _equality_table, _config_solref, _config_solimp
    target = path if path is not None else _master_joint_config_path
    if target is None:
        print("[ALLEX][Eq] set_equality_config: no master path installed yet; skipping")
        return
    _equality_table, _config_solref, _config_solimp = _load_equality(target)
    kind = "master" if path is None else "scenario"
    extra = (
        f" solref={_config_solref} solimp={_config_solimp}"
        if _config_solref is not None or _config_solimp is not None
        else ""
    )
    print(
        f"[ALLEX][Eq] equality table swapped → {kind} "
        f"({target.name}): {len(_equality_table)} entries{extra}"
    )


def _load_equality(
    joint_config_path: Path,
) -> tuple[
    list[tuple[str, str, list[float], str]],
    Optional[tuple[float, float]],
    Optional[tuple[float, float, float, float, float]],
]:
    """Load the equality table plus optional file-level `solref`/`solimp`.

    MJCF semantics — Newton forwards as `custom_attributes={"mujoco:eq_solref",
    "mujoco:eq_solimp"}`:
      - solref = [timeconst, dampratio]
      - solimp = [dmin, dmax, width, midpoint, power]
    """
    with open(joint_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    entries: list[tuple[str, str, list[float], str]] = []
    for e in cfg.get("equality_constraints", []):
        if not e.get("active", True):
            continue
        entries.append(
            (e["follower"], e["master"], list(e["polycoef"]), e.get("label", ""))
        )
    solref_raw = cfg.get("solref")
    solimp_raw = cfg.get("solimp")
    solref = (
        (float(solref_raw[0]), float(solref_raw[1])) if solref_raw is not None else None
    )
    solimp = (
        tuple(float(x) for x in solimp_raw) if solimp_raw is not None else None
    )
    return entries, solref, solimp


def _load_friction(joint_config_path: Path) -> dict[str, float]:
    """joint_friction 섹션에서 {short_joint_name: dynamic_value} 반환.

    Newton Model.joint_friction 은 per-DOF single scalar → 'dynamic' 을 선호,
    없으면 'static' 폴백. 메타 키(언더스코어 시작)와 dict 아닌 값은 제외.
    """
    with open(joint_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg.get("joint_friction", {}) or {}
    out: dict[str, float] = {}
    for name, spec in raw.items():
        if name.startswith("_") or not isinstance(spec, dict):
            continue
        val = spec.get("dynamic")
        if val is None:
            val = spec.get("static")
        if val is None:
            continue
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _inject_friction(builder) -> None:
    """builder.joint_friction[qd_start] 를 joint_config.json 기반으로 덮어씀.

    Newton 이 finalize 시점에 이 리스트를 wp.array 로 할당하므로,
    finalize 호출 직전에 수정되어야 Model.joint_friction 에 반영된다.
    USD newton:friction 경로는 Newton importer 가 가끔 깨우지 못해
    이 훅으로 강제 주입이 가장 결정적.
    """
    if not _friction_table:
        return
    joint_labels = list(builder.joint_label or [])
    joint_qd_start = list(getattr(builder, "joint_qd_start", []) or [])
    joint_fric = getattr(builder, "joint_friction", None)
    if joint_fric is None or not joint_labels or not joint_qd_start:
        print("[ALLEX][Friction] builder missing joint_friction / label / qd_start; skip")
        return

    applied = 0
    for ji, label in enumerate(joint_labels):
        short = label.rsplit("/", 1)[-1]
        val = _friction_table.get(short)
        if val is None:
            continue
        if ji >= len(joint_qd_start):
            continue
        qd = int(joint_qd_start[ji])
        if 0 <= qd < len(joint_fric):
            joint_fric[qd] = val
            applied += 1
    print(f"[ALLEX][Friction] injected friction on {applied}/{len(_friction_table)} joint(s) at builder level")


def _inject_actuator_gravcomp(builder) -> None:
    """builder 의 mujoco:jnt_actgravcomp custom-attribute 값을 셋팅.

    finalize 직전에 호출되어야 builder 가 MuJoCo-Warp actuator 생성 시
    actgravcomp=True 를 읽음. 런타임 on/off 는 불가 — 값 바꿨으면 stage
    reload (확장 disable/enable) 필요.

    NOTE: 켜진 joint 는 qfrc_actuator 에 PD + 중력 토크가 **합산**되어 나옴.
    body 레벨 mujoco:gravcomp (sim_settings_utils.apply_gravcomp_to_builder) 와는
    별개의 경로다.
    """
    if not _gravcomp_cfg.get("enabled", False):
        return

    # builder 의 CustomAttribute 찾기. namespace 포함 key 우선, fallback.
    attr = None
    cust = getattr(builder, "custom_attributes", None)
    if cust is not None:
        attr = cust.get("mujoco:jnt_actgravcomp") or cust.get("jnt_actgravcomp")
    if attr is None:
        print("[ALLEX][GravComp] mujoco:jnt_actgravcomp custom attribute not found. "
              "MuJoCo solver not loaded? — skip")
        return

    joint_labels = list(builder.joint_label or [])
    joint_qd_start = list(getattr(builder, "joint_qd_start", []) or [])
    total_dof = int(getattr(builder, "joint_dof_count", 0) or 0)
    if not joint_labels or not joint_qd_start:
        print("[ALLEX][GravComp] builder missing joint_label / qd_start; skip")
        return

    target_names = set(_gravcomp_cfg.get("joints", []) or [])
    all_joints = not target_names

    if attr.values is None:
        attr.values = {}

    applied_joints = 0
    applied_dofs = 0
    for ji, label in enumerate(joint_labels):
        short = label.rsplit("/", 1)[-1]
        if not all_joints and short not in target_names:
            continue
        if ji >= len(joint_qd_start):
            continue
        qd_start = int(joint_qd_start[ji])
        qd_end = int(joint_qd_start[ji + 1]) if ji + 1 < len(joint_qd_start) else total_dof
        if qd_end <= qd_start:
            continue  # fixed/zero-DOF joint
        for qd in range(qd_start, qd_end):
            attr.values[qd] = True
            applied_dofs += 1
        applied_joints += 1
    print(f"[ALLEX][GravComp] actuator-gravcomp enabled on {applied_joints} joint(s) "
          f"/ {applied_dofs} DOF(s). NOTE: qfrc_actuator 값이 PD+중력 합산임 "
          "(torque plot 해석 주의)")


def _find_joint_idx(builder, short_name: str) -> int | None:
    target = "/" + short_name
    for i, label in enumerate(builder.joint_label):
        if label == short_name or label.endswith(target):
            return i
    for i, label in enumerate(builder.joint_label):
        if short_name in label:
            return i
    return None


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


def _joint_type_name(t) -> str:
    try:
        from newton._src.sim.enums import JointType
        try:
            return JointType(int(t)).name
        except Exception:
            return str(t)
    except Exception:
        return str(t)


def _apply_gravcomp(builder) -> None:
    """Stamp `mujoco:gravcomp` per `physics_config.json::newton.gravcomp.bodies`.

    Runs from the finalize patch — after add_usd populated body labels and the
    MuJoCo solver registered its custom attributes — so MuJoCo's `m.ngravcomp`
    counts these bodies at `put_model` time and the qfrc_gravcomp kernel runs
    every step. Failure here is non-fatal: equality injection still proceeds.
    """
    try:
        from ..utils import sim_settings_utils as _ps
        _ps.apply_gravcomp_to_builder(builder)
    except Exception as exc:
        print(f"[ALLEX][Gravcomp] apply_gravcomp_to_builder failed: {exc}")


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

    # MJCF softness from joint_config.json (top-level solref/solimp). When a key
    # is absent in the file, the corresponding custom_attribute is omitted and
    # Newton/MuJoCo applies its own default.
    custom_attrs = None
    if _config_solref is not None or _config_solimp is not None:
        try:
            import warp as wp
            attrs: dict = {}
            if _config_solref is not None:
                attrs["mujoco:eq_solref"] = wp.vec2(_config_solref[0], _config_solref[1])
            if _config_solimp is not None:
                attrs["mujoco:eq_solimp"] = tuple(_config_solimp)
            custom_attrs = attrs or None
        except Exception as exc:
            print(f"[ALLEX][Eq] could not build custom_attrs (warp import?): {exc}")

    added = 0
    missed: list[tuple[str, str]] = []
    for follower, master, coef, label in _equality_table:
        f = _find_joint_idx(builder, follower)
        m = _find_joint_idx(builder, master)
        if f is None or m is None:
            missed.append((follower, master))
            continue

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
            missed.append((follower, master))
            continue
        print(
            f"[ALLEX][Eq] + follower='{follower}'(#{f})  master='{master}'(#{m})  polycoef={coef}"
        )
        added += 1
    if added:
        print(
            f"[ALLEX] injected {added}/{len(_equality_table)} equality constraints "
            f"(solref={_config_solref}, solimp={_config_solimp})"
        )
    if missed:
        print(f"[ALLEX] unmatched equality joints: {missed}")



def _install_articulation_reset_patch() -> None:
    """Avoid Articulation.set_gains device mismatch on Newton backend.

    Isaac Sim's Articulation._on_post_reset calls set_gains(kps=self._default_kps, ...),
    which internally does move_data(kps, device="cpu") while get_dof_stiffnesses()
    returns CUDA tensors under Newton — causing a device-mismatch RuntimeError
    during world.reset_async(). Passing kps=None, kds=None takes the safe branch
    that reads stiffnesses from the physics view on the matching device.

    Re-install 가드는 함수 객체 자체에 sentinel attribute 부착으로 처리 — extension
    reload 로 module 이 새로 로드돼도 이미 패치된 상태를 정확히 감지해 patched-of-
    patched chain 을 막는다.
    """
    global _orig_on_post_reset
    try:
        from isaacsim.core.prims.impl.articulation import Articulation
        from isaacsim.core.prims.impl.xform_prim import XFormPrim
    except ImportError as exc:
        print(f"[ALLEX] cannot import Articulation for reset patch: {exc}")
        return

    cur = Articulation._on_post_reset
    if getattr(cur, _ARTICULATION_RESET_SENTINEL, False):
        # 이미 patched 인 함수 — uninstall/restore 를 위한 _orig 만 채워두고 종료.
        if _orig_on_post_reset is None:
            _orig_on_post_reset = getattr(cur, "_allex_orig", None)
        return

    _orig_on_post_reset = cur

    def _patched_on_post_reset(self, event):
        XFormPrim._on_post_reset(self, event)
        Articulation.set_joint_positions(self, self._default_joints_state.positions)
        Articulation.set_joint_velocities(self, self._default_joints_state.velocities)
        Articulation.set_joint_efforts(self, self._default_joints_state.efforts)
        Articulation.set_gains(self, kps=None, kds=None)

    setattr(_patched_on_post_reset, _ARTICULATION_RESET_SENTINEL, True)
    setattr(_patched_on_post_reset, "_allex_orig", _orig_on_post_reset)
    Articulation._on_post_reset = _patched_on_post_reset
    print("[ALLEX] patched Articulation._on_post_reset for Newton device consistency")


def _uninstall_articulation_reset_patch() -> None:
    global _orig_on_post_reset
    try:
        from isaacsim.core.prims.impl.articulation import Articulation
    except ImportError:
        _orig_on_post_reset = None
        return
    cur = Articulation._on_post_reset
    if not getattr(cur, _ARTICULATION_RESET_SENTINEL, False):
        # 패치되지 않은 상태 — 할 일 없음.
        _orig_on_post_reset = None
        return
    orig = getattr(cur, "_allex_orig", None) or _orig_on_post_reset
    if orig is not None:
        Articulation._on_post_reset = orig
    _orig_on_post_reset = None


def configure_newton_from_toml() -> bool:
    """Apply newton + newton.solver sections from src/allex/config/physics_config.json.

    Key use: pd_scale = π/180 cancels Newton USD importer's implicit deg→rad
    drive-gain multiplication so USD-authored SI gains pass through unchanged.

    Returns True if the override was applied, False otherwise.  Call this
    multiple times if needed — it's idempotent.
    """
    print("[ALLEX][Cfg] configure_newton_from_toml() called")
    try:
        from isaacsim.physics.newton import acquire_stage
        from ..utils import sim_settings_utils as _ps
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
    global _orig_finalize, _equality_table
    global _config_solref, _config_solimp, _master_joint_config_path
    global _friction_table, _gravcomp_cfg
    _install_articulation_reset_patch()
    _configure_newton_from_toml()
    _master_joint_config_path = Path(joint_config_path)

    try:
        import newton
    except ImportError:
        print("[ALLEX] newton not importable; skipping equality constraint hook")
        return

    # Re-install 가드는 함수 자체에 박힌 sentinel 로 검사 (module reload 후에도 정확).
    cur_finalize = newton.ModelBuilder.finalize
    already_patched = getattr(cur_finalize, _FINALIZE_SENTINEL, False)

    try:
        _equality_table, _config_solref, _config_solimp = _load_equality(joint_config_path)
    except Exception as exc:
        print(f"[ALLEX] failed to load equality table from {joint_config_path}: {exc}")
        _equality_table = []
        _config_solref = None
        _config_solimp = None
        return
    print(
        f"[ALLEX] loaded {len(_equality_table)} equality entries from {joint_config_path}"
    )
    try:
        _friction_table = _load_friction(joint_config_path)
    except Exception as exc:
        print(f"[ALLEX] failed to load friction table: {exc}")
        _friction_table = {}
    print(f"[ALLEX] loaded {len(_friction_table)} friction entries from {joint_config_path}")

    try:
        from ..utils import sim_settings_utils as _ps
        _gravcomp_cfg = _ps.get_actuator_gravcomp_cfg()
    except Exception as exc:
        print(f"[ALLEX] failed to load actuator_gravcomp cfg: {exc}")
        _gravcomp_cfg = {"enabled": False, "joints": []}
    print(
        f"[ALLEX] actuator_gravcomp: enabled={_gravcomp_cfg.get('enabled')}, "
        f"joints={len(_gravcomp_cfg.get('joints', []))} target(s)"
    )

    if already_patched:
        # 이미 patched 인 finalize — 같은 sentinel 발견 시 _orig_finalize 만 복원해서
        # uninstall 이 정상 동작하게 두고 재설치는 skip. equality table 등은 위에서
        # reload 됐으므로 다음 finalize 호출이 새 값을 쓴다.
        if _orig_finalize is None:
            _orig_finalize = getattr(cur_finalize, "_allex_orig", None)
        return

    _orig_finalize = cur_finalize

    def _patched_finalize(self, *args, **kwargs):
        try:
            _apply_gravcomp(self)
        except Exception as exc:
            print(f"[ALLEX] gravcomp apply raised: {exc}")
        try:
            _inject(self)
        except Exception as exc:
            print(f"[ALLEX] equality injection raised: {exc}")
        try:
            _inject_friction(self)
        except Exception as exc:
            print(f"[ALLEX] friction injection raised: {exc}")
        try:
            _inject_actuator_gravcomp(self)
        except Exception as exc:
            print(f"[ALLEX] actuator gravcomp injection raised: {exc}")
        model = _orig_finalize(self, *args, **kwargs)
        try:
            _dump_model_equality_state(model)
        except Exception as exc:
            print(f"[ALLEX][Eq] post-finalize dump failed: {exc}")
        return model

    setattr(_patched_finalize, _FINALIZE_SENTINEL, True)
    setattr(_patched_finalize, "_allex_orig", _orig_finalize)
    newton.ModelBuilder.finalize = _patched_finalize


def uninstall() -> None:
    global _orig_finalize
    _uninstall_articulation_reset_patch()
    try:
        import newton
    except ImportError:
        _orig_finalize = None
        return
    cur = newton.ModelBuilder.finalize
    if not getattr(cur, _FINALIZE_SENTINEL, False):
        _orig_finalize = None
        return
    orig = getattr(cur, "_allex_orig", None) or _orig_finalize
    if orig is not None:
        newton.ModelBuilder.finalize = orig
    _orig_finalize = None
