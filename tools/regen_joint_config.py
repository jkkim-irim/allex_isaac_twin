"""Regenerate src/config/joint_config.json from asset/ALLEX/mjcf/ALLEX.xml.

Run:
    python tools/regen_joint_config.py

The hand-managed ``ui`` and ``drive_gains`` sections of the existing JSON are
preserved across regeneration; only MJCF-derived keys are overwritten.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


EXT_ROOT = Path(__file__).resolve().parent.parent
MJCF_PATH = EXT_ROOT / "asset" / "ALLEX" / "mjcf" / "ALLEX.xml"
OUT_PATH = EXT_ROOT / "src" / "config" / "joint_config.json"


def _parse_polycoef(s: str) -> list[float]:
    vals = [float(x) for x in s.split()]
    while len(vals) < 5:
        vals.append(0.0)
    return vals[:5]


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

    equality: list[dict] = []
    for eq in root.iter("equality"):
        for j in eq.findall("joint"):
            j1 = j.get("joint1")
            j2 = j.get("joint2")
            coef = j.get("polycoef", "0 1 0 0 0")
            active = j.get("active", "true").lower() == "true"
            if not j1 or not j2:
                continue
            equality.append({
                "follower": j1,
                "master": j2,
                "polycoef": _parse_polycoef(coef),
                "active": active,
                "label": f"{j1}__{j2}",
            })

    follower_set = {e["follower"] for e in equality if e["active"]}
    follower_list = sorted(follower_set)

    joint_names = {str(i): n for i, n in enumerate(joints)}
    active_indices = [str(i) for i, n in enumerate(joints) if n not in follower_set]

    cfg = {
        "notes": "Auto-generated from asset/ALLEX/mjcf/ALLEX.xml by tools/regen_joint_config.py. Do not edit by hand (except the 'ui' and 'drive_gains' sections).",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
