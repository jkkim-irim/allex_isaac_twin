"""Regenerate src/allex/config/joint_config.json from asset/ALLEX/mjcf/ALLEX.xml.

Run:
    python tools/regen_joint_config.py

The hand-managed ``ui`` and ``drive_gains`` sections of the existing JSON are
preserved across regeneration; only MJCF-derived keys are overwritten.

Active master classification rule
---------------------------------
A joint is considered an *active master* iff it has motor/actuator presence in
the MJCF — detected via either:
  - ``actuatorfrcrange`` attribute on the ``<joint>`` element, or
  - referenced by an ``<actuator>/<position|motor|general joint=...>`` entry.

A joint is a *follower* iff it appears in any active ``<equality>`` and lacks
the actuator markers above. The MJCF ``<equality joint1=... joint2=...>`` order
is independent of which side carries the actuator (joint1 is just the LHS of
the constraint equation `joint1 = poly(joint2)`); ALLEX waist has the actuator
on joint1 (Lower_Pitch) while finger DIP/IP equalities have it on joint2 (PIP).
This tool detects both cases and, when the actuator is on joint1, swaps the
output so ``master`` always points to the actuated side. For linear polycoef
``[c0, c1, 0, 0, 0]`` the swap inverts to ``[-c0/c1, 1/c1, 0, 0, 0]`` so the
constraint stays mathematically identical. Non-linear polycoef is left
unswapped with a warning (no such case in current ALLEX MJCF).
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


EXT_ROOT = Path(__file__).resolve().parent.parent
MJCF_PATH = EXT_ROOT / "asset" / "ALLEX" / "mjcf" / "ALLEX.xml"
OUT_PATH = EXT_ROOT / "src" / "allex" / "config" / "joint_config.json"


def _parse_polycoef(s: str) -> list[float]:
    vals = [float(x) for x in s.split()]
    while len(vals) < 5:
        vals.append(0.0)
    return vals[:5]


def _is_linear_polycoef(coef: list[float], eps: float = 1e-12) -> bool:
    """True if polycoef is linear (only c0, c1 may be non-zero, c1 != 0)."""
    if any(abs(coef[k]) > eps for k in (2, 3, 4)):
        return False
    return abs(coef[1]) > eps


def _invert_linear_polycoef(coef: list[float]) -> list[float]:
    """Given c0 + c1*x form, return inverse: y = -c0/c1 + (1/c1)*x."""
    c0, c1 = coef[0], coef[1]
    return [-c0 / c1, 1.0 / c1, 0.0, 0.0, 0.0]


def _collect_actuated(root: ET.Element) -> set[str]:
    """Joints with motor/PD presence per MJCF semantics.

    Two indicators (union):
      1. ``<joint ... actuatorfrcrange="...">`` — joint-level marker.
      2. ``<actuator><position|motor|general|... joint="X" />`` — actuator section.
    """
    actuated: set[str] = set()

    for j in root.iter("joint"):
        name = j.get("name")
        if name and j.get("actuatorfrcrange"):
            actuated.add(name)

    for actuator_section in root.iter("actuator"):
        for child in actuator_section:
            jn = child.get("joint")
            if jn:
                actuated.add(jn)

    return actuated


def regenerate(mjcf_path: Path, out_path: Path) -> dict:
    root = ET.parse(mjcf_path).getroot()

    joints: list[str] = []
    for j in root.iter("joint"):
        name = j.get("name")
        jtype = j.get("type")
        if not name:
            continue
        if jtype in (None, "hinge", "slide"):
            joints.append(name)

    actuated = _collect_actuated(root)

    equality: list[dict] = []
    for eq in root.iter("equality"):
        for j in eq.findall("joint"):
            j1 = j.get("joint1")
            j2 = j.get("joint2")
            coef = _parse_polycoef(j.get("polycoef", "0 1 0 0 0"))
            active = j.get("active", "true").lower() == "true"
            if not j1 or not j2:
                continue

            # Decide which side is master (actuated) and which is follower.
            # MJCF order: joint1 = poly(joint2), but actuator presence is
            # independent. Swap when needed so 'master' ≡ actuated.
            j1_act = j1 in actuated
            j2_act = j2 in actuated
            swapped = False
            if j1_act and not j2_act:
                # Actuator on joint1 → swap so non-actuated side is follower.
                if _is_linear_polycoef(coef):
                    coef = _invert_linear_polycoef(coef)
                    j1, j2 = j2, j1
                    swapped = True
                else:
                    print(
                        f"WARN: actuator on joint1='{j1}' but polycoef is non-linear "
                        f"({coef}); leaving MJCF order — downstream may need manual fix",
                        file=sys.stderr,
                    )
            # If both actuated or neither, fall back to MJCF order (joint1=follower).

            equality.append({
                "follower": j1,
                "master": j2,
                "polycoef": coef,
                "active": active,
                "label": f"{j1}__{j2}",
                "swapped_from_mjcf": swapped,
            })

    follower_set = {e["follower"] for e in equality if e["active"]}
    follower_list = sorted(follower_set)

    joint_names = {str(i): n for i, n in enumerate(joints)}
    active_indices = [str(i) for i, n in enumerate(joints) if n not in follower_set]

    cfg = {
        "notes": (
            "Auto-generated from asset/ALLEX/mjcf/ALLEX.xml by tools/regen_joint_config.py. "
            "Do not edit by hand (except the 'ui' and 'drive_gains' sections). "
            "active_joints / follower_joints / equality_constraints classify by MJCF actuator "
            "presence (actuatorfrcrange or <actuator joint=...>); 'master' always points to the "
            "actuated side, 'follower' to the equality-constrained passive side. polycoef is in "
            "follower=poly(master) form (Newton add_equality_constraint_joint expects joint1=follower)."
        ),
        "source_mjcf": str(mjcf_path.relative_to(EXT_ROOT)),
        "total_joints": len(joints),
        "joint_names": joint_names,
        "active_joints": active_indices,
        "follower_joints": follower_list,
        "equality_constraints": equality,
        "coupled_joints": {},
    }

    # Preserve hand-managed sections across regeneration.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: could not read existing {out_path}: {exc}", file=sys.stderr)
            existing = {}
        for key in ("drive_gains", "ui"):
            if key in existing:
                cfg[key] = existing[key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg


def main() -> int:
    if not MJCF_PATH.exists():
        print(f"ERROR: MJCF not found at {MJCF_PATH}", file=sys.stderr)
        return 1
    cfg = regenerate(MJCF_PATH, OUT_PATH)
    print(f"wrote {OUT_PATH}")
    print(f"  total_joints: {cfg['total_joints']}")
    print(f"  active: {len(cfg['active_joints'])}")
    print(f"  followers: {len(cfg['follower_joints'])}")
    print(f"  equality_constraints: {len(cfg['equality_constraints'])}")
    swaps = [e for e in cfg["equality_constraints"] if e.get("swapped_from_mjcf")]
    if swaps:
        print(f"  swapped (actuator on MJCF joint1): {len(swaps)}")
        for e in swaps:
            print(f"    follower='{e['follower']}'  master='{e['master']}'  polycoef={e['polycoef']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
