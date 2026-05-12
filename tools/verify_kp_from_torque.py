"""Verify motor-side K_m by back-computing from torque + position error.

For scalar PD (slow phase, qd ≈ 0, qdt ≈ 0):
    τ_m_pd ≈ K_m · err_m              (motor-side)
    τ_j   = ratio · τ_m_clip ≈ ratio · K_m · (ratio · (qt - q))
          = ratio² · K_m · (qt - q)
  ⇒ K_m_inferred = τ_j_no_grav / (ratio² · (qt - q))

For mech-comp joints (Waist_Lower_Pitch, Neck_Pitch), τ_j_no_grav = qfrc_applied
(kernel doesn't include gravcomp in motor clip). For others, τ_j_no_grav =
qfrc_applied (since kernel writes joint_f = tau_j - g_j to qfrc_applied, and
qfrc_applied alone = motor's PD contribution without gravcomp).

Outputs per-joint plot: K_m_inferred (filtered slow points only) vs K_m_host
over time. Overlap = PD pipeline math correct. Discrepancy = bug or saturation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO = Path(__file__).resolve().parent.parent
JOINT_CONFIG = _REPO / "src" / "allex" / "config" / "joint_config.json"

# Scalar-group ratios (motor = ratio · joint, same as Isaac Sim kernel).
SHOULDER_RATIO = 40.538461540
SHOULDER_JOINTS = {f"{h}_Shoulder_{ax}_Joint"
                   for h in ("L", "R") for ax in ("Pitch", "Roll", "Yaw")}
WAIST_RATIOS = {"Waist_Yaw_Joint": 50.0, "Waist_Lower_Pitch_Joint": 25.0}
NECK_RATIOS  = {"Neck_Pitch_Joint": 7.75, "Neck_Yaw_Joint": 10.0}

MECH_COMP = {"Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint"}


def joint_ratio(name: str) -> float:
    if name in WAIST_RATIOS:
        return WAIST_RATIOS[name]
    if name in NECK_RATIOS:
        return NECK_RATIOS[name]
    if name in SHOULDER_JOINTS:
        return SHOULDER_RATIO
    return float("nan")  # coupled groups (elbow/wrist/finger/thumb): not handled here


def load_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with open(path) as f:
        r = csv.reader(f); h = next(r)
        rows = [[float(v) for v in row] for row in r]
    a = np.asarray(rows)
    return a[:, 0], h[1:], a[:, 1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, type=Path,
                    help="showcase_sim_data_<ts>/ directory")
    ap.add_argument("--out", default="data/diagnose/kp_verify", type=str)
    ap.add_argument("--qd-thresh", type=float, default=0.05,
                    help="|qd| threshold [rad/s] to declare slow phase (default 0.05)")
    ap.add_argument("--err-thresh", type=float, default=0.01,
                    help="|qt-q| threshold [rad] to declare meaningful err (default 0.01)")
    args = ap.parse_args()

    sim = args.sim
    out_dir = (_REPO / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load required CSVs
    t, pos_cols, pos = load_csv(sim / "joint_position.csv")          # actual (rad)
    _, tgt_cols, tgt = load_csv(sim / "joint_position_target.csv")   # target (rad)
    _, qfa_cols, qfa = load_csv(sim / "qfrc_applied.csv")            # joint_f
    _, gj_cols, gj   = load_csv(sim / "qfrc_gravcomp.csv")           # gravcomp
    _, km_cols, km   = load_csv(sim / "K_m_host.csv")                # motor-side kp

    # qd via numerical diff of pos
    dt = float(t[1] - t[0])
    qd = np.zeros_like(pos)
    qd[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    qd[0]  = (pos[1] - pos[0]) / dt
    qd[-1] = (pos[-1] - pos[-2]) / dt

    # Strip "pos_"/"tgt_" prefixes
    pos_names = [c[4:] if c.startswith("pos_") else c for c in pos_cols]
    tgt_names = [c[4:] if c.startswith("tgt_") else c for c in tgt_cols]

    # Diagnose scalar joints only.
    targets = [
        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
        "L_Shoulder_Pitch_Joint", "Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint",
        "Neck_Pitch_Joint", "Neck_Yaw_Joint",
    ]
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n), sharex=True)

    print(f"{'joint':30s}  {'ratio':>7s}  {'#slow':>7s}  {'K_m median':>10s}  "
          f"{'K_m_host avg':>12s}")
    print("-" * 80)
    for i, jname in enumerate(targets):
        ax = axes[i]
        r = joint_ratio(jname)
        if not np.isfinite(r):
            ax.text(0.5, 0.5, f"{jname} unsupported", transform=ax.transAxes, ha="center")
            continue
        try:
            j_pos = pos_names.index(jname)
            j_tgt = tgt_names.index(jname)
            j_qfa = qfa_cols.index(jname)
            j_gj  = gj_cols.index(jname)
            j_km  = km_cols.index(jname)
        except ValueError:
            ax.text(0.5, 0.5, f"{jname} not in CSV", transform=ax.transAxes, ha="center")
            continue

        q     = pos[:, j_pos]
        qt    = tgt[:, j_tgt]
        qfa_v = qfa[:, j_qfa]
        gj_v  = gj[:, j_gj]
        km_v  = km[:, j_km]
        qd_v  = qd[:, j_pos]
        err   = qt - q

        # For mech-comp: qfrc_applied = motor PD only (no gravcomp) → no need to subtract
        # For non-mech: qfrc_applied = motor PD output (kernel subtracted gravcomp; final
        #   joint torque = qfrc_applied + qfrc_gravcomp). PD-only = qfrc_applied
        # In both cases, qfrc_applied is the motor's contribution to joint torque.
        tau_j_motor = qfa_v.copy()
        # For motor PD inference: τ_j_motor ≈ ratio² · K_m · err (when slow)
        # But Isaac Sim kernel writes qfrc_applied = tau_j - g_j (= motor torque - gravcomp
        # for non-mech). So qfrc_applied alone represents motor's PD output (gravcomp split
        # to passive route). For mech, kernel sets g_m=0 in motor clip, then writes
        # qfrc_applied = tau_j (no subtract). Both interpretations: qfrc_applied ≈ motor PD.

        # Slow-phase mask
        mask_slow = (np.abs(qd_v) < args.qd_thresh) & (np.abs(err) > args.err_thresh)
        denom = (r * r) * err
        with np.errstate(divide="ignore", invalid="ignore"):
            km_inferred = np.where(mask_slow & (np.abs(denom) > 1e-9),
                                   tau_j_motor / denom, np.nan)

        finite_inf = km_inferred[np.isfinite(km_inferred)]
        if finite_inf.size > 0:
            med = float(np.median(finite_inf))
            print(f"{jname:30s}  {r:7.3f}  {finite_inf.size:7d}  {med:10.4f}  "
                  f"{km_v.mean():12.4f}")
        else:
            print(f"{jname:30s}  {r:7.3f}  {0:7d}      -      {km_v.mean():12.4f}")

        ax.plot(t, km_v, color="C0", lw=1.2, label="K_m_host")
        ax.plot(t, km_inferred, color="C1", lw=0.6, alpha=0.6,
                label="K_m_inferred (slow phase)")
        ax.set_ylabel(jname.replace("_Joint", "") + "\nK_m")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time [s]")

    fig.suptitle(f"K_m verification: inferred vs host  ({sim.name})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = out_dir / f"kp_verify_{sim.name}.png"
    fig.savefig(out_path, dpi=110)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
