"""Plot real (rosbag) vs sim (SimStateLogger) joint torque for the same trajectory.

real 출처: rosbag2 mcap 의 /robot_outbound_data/<group>/joint_torque 토픽
            (Float64MultiArray, joint-side torque time series)
sim 출처:  data/showcase/showcase_sim_data_*/qfrc_applied.csv
            (= MotorStateMirror 의 joint_f, real joint_torque 와 직접 비교 가능)

real / sim 의 timestamp 시작이 다르므로 0 으로 정렬 (각각 첫 sample 을 t=0 으로).
trajectory ramp-in 길이 (1.5s) 만큼 sim 쪽을 shift 할지는 옵션.

Usage:
    python tools/plot_real_vs_sim_torque.py \\
        --rosbag data/demo1_rosbag_dynamic_motion/rosbag2_2026_04_29-12_41_34 \\
        --sim    data/showcase/showcase_sim_data_<ts>

옵션:
    --sim-shift SECONDS  sim 데이터 시작 시점을 SECONDS 만큼 미루기 (ramp-in 정렬용)
    --groups LIST        쉼표로 구분된 그룹 (default: all)
    --out OUT_DIR        plot 저장 (default: data/diagnose/torque_compare)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent


# 그룹별: (group_name, CSV joint 라벨, rosbag topic suffix)
GROUPS = [
    ("theOne_waist",     ["Waist_Yaw", "Waist_Lower_Pitch"]),
    ("theOne_neck",      ["Neck_Pitch", "Neck_Yaw"]),
    ("Arm_L_theOne",     ["L_Sh_Pitch", "L_Sh_Roll", "L_Sh_Yaw", "L_Elbow",
                          "L_W_Yaw", "L_W_Roll", "L_W_Pitch"]),
    ("Arm_R_theOne",     ["R_Sh_Pitch", "R_Sh_Roll", "R_Sh_Yaw", "R_Elbow",
                          "R_W_Yaw", "R_W_Roll", "R_W_Pitch"]),
    ("Hand_L_thumb_wir", ["L_Thumb_Yaw", "L_Thumb_CMC", "L_Thumb_MCP"]),
    ("Hand_L_index_wir", ["L_Index_ABAD", "L_Index_MCP", "L_Index_PIP"]),
    ("Hand_L_middle_wir",["L_Middle_ABAD", "L_Middle_MCP", "L_Middle_PIP"]),
    ("Hand_L_ring_wir",  ["L_Ring_ABAD", "L_Ring_MCP", "L_Ring_PIP"]),
    ("Hand_L_little_wir",["L_Little_ABAD", "L_Little_MCP", "L_Little_PIP"]),
    ("Hand_R_thumb_wir", ["R_Thumb_Yaw", "R_Thumb_CMC", "R_Thumb_MCP"]),
    ("Hand_R_index_wir", ["R_Index_ABAD", "R_Index_MCP", "R_Index_PIP"]),
    ("Hand_R_middle_wir",["R_Middle_ABAD", "R_Middle_MCP", "R_Middle_PIP"]),
    ("Hand_R_ring_wir",  ["R_Ring_ABAD", "R_Ring_MCP", "R_Ring_PIP"]),
    ("Hand_R_little_wir",["R_Little_ABAD", "R_Little_MCP", "R_Little_PIP"]),
]

# Group 의 joint 라벨 → sim qfrc_applied.csv 컬럼 매핑 (sim 컬럼은 full joint name)
SIM_COL_MAP = {
    "Waist_Yaw":        "Waist_Yaw_Joint",
    "Waist_Lower_Pitch":"Waist_Lower_Pitch_Joint",
    "Neck_Pitch":       "Neck_Pitch_Joint",
    "Neck_Yaw":         "Neck_Yaw_Joint",
    "L_Sh_Pitch":       "L_Shoulder_Pitch_Joint",
    "L_Sh_Roll":        "L_Shoulder_Roll_Joint",
    "L_Sh_Yaw":         "L_Shoulder_Yaw_Joint",
    "L_Elbow":          "L_Elbow_Joint",
    "L_W_Yaw":          "L_Wrist_Yaw_Joint",
    "L_W_Roll":         "L_Wrist_Roll_Joint",
    "L_W_Pitch":        "L_Wrist_Pitch_Joint",
    "R_Sh_Pitch":       "R_Shoulder_Pitch_Joint",
    "R_Sh_Roll":        "R_Shoulder_Roll_Joint",
    "R_Sh_Yaw":         "R_Shoulder_Yaw_Joint",
    "R_Elbow":          "R_Elbow_Joint",
    "R_W_Yaw":          "R_Wrist_Yaw_Joint",
    "R_W_Roll":         "R_Wrist_Roll_Joint",
    "R_W_Pitch":        "R_Wrist_Pitch_Joint",
}
for hand in ("L", "R"):
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
        if finger == "Thumb":
            parts = ("Yaw", "CMC", "MCP")
        else:
            parts = ("ABAD", "MCP", "PIP")
        for p in parts:
            short = f"{hand}_{finger}_{p}"
            SIM_COL_MAP[short] = f"{hand}_{finger}_{p}_Joint"


def read_rosbag_topic(rosbag_dir: Path, group: str, topic_suffix: str,
                      t0_ns_ref: int | None = None
                      ) -> "tuple[np.ndarray, np.ndarray, int] | None":
    """topic /robot_outbound_data/<group>/<topic_suffix> 의
    (t_rel[s], data[n_t, n_joints], t0_ns) 반환.
    t0_ns_ref 가 주어지면 그 값을 기준으로 t_rel 계산 (그룹 간 시간 동기화)."""
    from mcap_ros2.reader import read_ros2_messages
    mcaps = list(rosbag_dir.glob("*.mcap"))
    if not mcaps:
        print(f"  no .mcap in {rosbag_dir}")
        return None
    topic = f"/robot_outbound_data/{group}/{topic_suffix}"
    times_ns: list[int] = []
    rows: list[list[float]] = []
    try:
        for msg in read_ros2_messages(str(mcaps[0]), topics=[topic]):
            times_ns.append(msg.log_time_ns)
            rows.append(list(msg.ros_msg.data))
    except Exception as exc:
        print(f"  ⚠ rosbag read failed for {topic}: {exc}")
        return None
    if not rows:
        print(f"  no messages on {topic}")
        return None
    t0 = t0_ns_ref if t0_ns_ref is not None else times_ns[0]
    t = (np.asarray(times_ns, dtype=np.int64) - t0) / 1e9
    data = np.asarray(rows, dtype=np.float64)
    return t, data, times_ns[0]


def read_rosbag_torques(rosbag_dir: Path, group: str):
    res = read_rosbag_topic(rosbag_dir, group, "joint_torque")
    if res is None:
        return None
    t, data, _t0 = res
    return t, data


def read_sim_csv(sim_dir: Path, fname: str = "qfrc_applied.csv"
                 ) -> "tuple[np.ndarray, dict[str, np.ndarray]] | None":
    """sim_dir/<fname> 읽어 (time, {col_name: data}) 반환."""
    path = sim_dir / fname
    if not path.is_file():
        print(f"  ⚠ {path} not found")
        return None
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader]
    arr = np.asarray(rows, dtype=np.float64)
    t = arr[:, 0]
    cols = {h: arr[:, i] for i, h in enumerate(header[1:], start=1)}
    return t, cols


# Joints with mechanical (spring) gravity compensation — for these the motor
# does NOT supply gravity torque, so sim's "motor joint-side output" equals
# qfrc_applied alone (NOT qfrc_applied + qfrc_gravcomp).
MECH_COMP_SIM_COLS = {"Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint"}


def _diff_deg(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Numerical derivative dx/dt (deg/s) via centered finite diff. Same shape as x."""
    out = np.zeros_like(x)
    if len(t) < 2:
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    out[0]  = (x[1]  - x[0])  / (t[1]  - t[0])
    out[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    return out


def plot_group(
    group: str, joint_labels: list[str],
    t_real: np.ndarray, trq_real: np.ndarray,
    t_sim: np.ndarray, sim_trq_cols: dict[str, np.ndarray],
    sim_shift: float, out_dir: Path, sim_source_label: str = "qfrc_applied",
    t_pos: np.ndarray | None = None,
    pos_actual_deg: np.ndarray | None = None,
    t_tgt: np.ndarray | None = None,
    pos_target_deg: np.ndarray | None = None,
    sim_pos_cols: dict[str, np.ndarray] | None = None,
    sim_tgt_cols: dict[str, np.ndarray] | None = None,
    sim_trq_cols_no_grav: dict[str, np.ndarray] | None = None,
    t_vel: np.ndarray | None = None,
    vel_actual_deg: np.ndarray | None = None,
    t_vt: np.ndarray | None = None,
    vel_target_deg: np.ndarray | None = None,
):
    n_joints = trq_real.shape[1]
    if n_joints != len(joint_labels):
        print(f"  ⚠ {group}: real has {n_joints} joints, label list has {len(joint_labels)} — using min")
        n_joints = min(n_joints, len(joint_labels))

    has_pos = (
        t_pos is not None and pos_actual_deg is not None
        and t_tgt is not None and pos_target_deg is not None
    )
    n_cols = 3 if has_pos else 1

    fig, axes = plt.subplots(n_joints, n_cols,
                             figsize=(11 * n_cols, 1.8 * n_joints + 1),
                             sharex="col", squeeze=False)

    t_sim_shifted = t_sim - sim_shift

    for j in range(n_joints):
        label = joint_labels[j]
        sim_col = SIM_COL_MAP.get(label)

        # --- torque panel (col 0) ---
        # For mech-comp joints, sim torque excludes gravcomp (motor doesn't provide
        # gravity — mechanical spring does); use qfrc_applied directly.
        ax_t = axes[j, 0]
        ax_t.plot(t_real, trq_real[:, j], color="C0", lw=1.0, label="real")
        is_mech = sim_col in MECH_COMP_SIM_COLS
        trq_dict = sim_trq_cols_no_grav if is_mech and sim_trq_cols_no_grav is not None else sim_trq_cols
        src_lbl = "qfrc_applied (mech grav)" if is_mech else sim_source_label
        if sim_col and sim_col in trq_dict:
            ax_t.plot(t_sim_shifted, trq_dict[sim_col], color="C1", lw=1.0,
                      label=f"sim {src_lbl}", alpha=0.8)
        else:
            ax_t.text(0.02, 0.5, f"sim col '{sim_col}' missing",
                      transform=ax_t.transAxes, color="red", fontsize=8)
        ax_t.set_ylabel(f"{label}\n[N·m]", fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(loc="upper right", fontsize=7)
        if j == 0:
            ax_t.set_title(f"{group} — joint torque (sim shift {sim_shift:.2f}s)")

        # --- position panel (col 1): actual + target overlay ---
        if has_pos and j < pos_actual_deg.shape[1] and j < pos_target_deg.shape[1]:
            ax_p = axes[j, 1]
            ax_p.plot(t_pos, pos_actual_deg[:, j], color="C0", lw=1.0, label="real actual")
            ax_p.plot(t_tgt, pos_target_deg[:, j], color="C0", lw=1.0, ls="--",
                      label="real target", alpha=0.7)
            if sim_col and sim_pos_cols is not None and sim_col in sim_pos_cols:
                ax_p.plot(t_sim_shifted, sim_pos_cols[sim_col] * (180.0/np.pi),
                          color="C1", lw=1.0, label="sim actual", alpha=0.9)
            if sim_col and sim_tgt_cols is not None and sim_col in sim_tgt_cols:
                ax_p.plot(t_sim_shifted, sim_tgt_cols[sim_col] * (180.0/np.pi),
                          color="C1", lw=1.0, ls="--", label="sim target", alpha=0.7)
            ax_p.set_ylabel("pos [deg]", fontsize=8)
            ax_p.grid(True, alpha=0.3)
            ax_p.legend(loc="upper right", fontsize=6, ncol=2)
            if j == 0:
                ax_p.set_title(f"{group} — position: actual vs target")

        # --- velocity panel (col 2): actual + target overlay ---
        if has_pos:
            ax_v = axes[j, 2]
            if (t_vel is not None and vel_actual_deg is not None
                    and j < vel_actual_deg.shape[1]):
                ax_v.plot(t_vel, vel_actual_deg[:, j], color="C0", lw=1.0, label="real actual")
            if (t_vt is not None and vel_target_deg is not None
                    and j < vel_target_deg.shape[1]):
                ax_v.plot(t_vt, vel_target_deg[:, j], color="C0", lw=1.0, ls="--",
                          label="real target", alpha=0.7)
            # Sim velocity from position derivative (sim_vel not logged yet)
            if sim_col and sim_pos_cols is not None and sim_col in sim_pos_cols:
                ax_v.plot(t_sim_shifted, _diff_deg(t_sim, sim_pos_cols[sim_col] * (180.0/np.pi)),
                          color="C1", lw=1.0, label="sim actual", alpha=0.9)
            if sim_col and sim_tgt_cols is not None and sim_col in sim_tgt_cols:
                ax_v.plot(t_sim_shifted, _diff_deg(t_sim, sim_tgt_cols[sim_col] * (180.0/np.pi)),
                          color="C1", lw=1.0, ls="--", label="sim target", alpha=0.7)
            ax_v.set_ylabel("vel [deg/s]", fontsize=8)
            ax_v.grid(True, alpha=0.3)
            ax_v.legend(loc="upper right", fontsize=6, ncol=2)
            if j == 0:
                ax_v.set_title(f"{group} — velocity: actual vs target")

    axes[-1, 0].set_xlabel("time [s]")
    if has_pos:
        axes[-1, 1].set_xlabel("time [s]")
        axes[-1, 2].set_xlabel("time [s]")

    fig.tight_layout()
    out_path = out_dir / f"compare_{group}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(_REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rosbag", required=True,
                    help="rosbag2 directory (contains .mcap file)")
    ap.add_argument("--sim", required=True,
                    help="sim showcase data dir (data/showcase/showcase_sim_data_*)")
    ap.add_argument("--sim-shift", type=float, default=0.0,
                    help="sim 시간축을 N 초 만큼 shift (ramp-in 정렬용; default 0)")
    ap.add_argument("--sim-source", default="tau_j",
                    choices=["tau_j", "qfrc_applied", "qfrc_gravcomp", "joint_torque"],
                    help="sim 의 어떤 양을 real joint_torque 와 비교할지. "
                         "'tau_j' = qfrc_applied + qfrc_gravcomp (= motor joint-side output, "
                         "real joint_torque 와 동일 의미). 'qfrc_applied' = 우리 kernel 의 "
                         "tau_j-g_j (real 과 비교용으로는 부적절). default tau_j.")
    ap.add_argument("--groups", default=None,
                    help="comma-sep 그룹 이름 (default: all)")
    ap.add_argument("--out", default="data/diagnose/torque_compare",
                    help="output dir (default: data/diagnose/torque_compare)")
    args = ap.parse_args()

    rosbag_dir = (_REPO / args.rosbag).resolve() if not Path(args.rosbag).is_absolute() else Path(args.rosbag)
    sim_dir    = (_REPO / args.sim).resolve()    if not Path(args.sim).is_absolute()    else Path(args.sim)

    if not rosbag_dir.is_dir():
        print(f"[ERROR] rosbag dir not found: {rosbag_dir}")
        sys.exit(1)
    if not sim_dir.is_dir():
        print(f"[ERROR] sim dir not found: {sim_dir}")
        sys.exit(1)

    out_dir = (_REPO / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"rosbag: {rosbag_dir.relative_to(_REPO) if rosbag_dir.is_relative_to(_REPO) else rosbag_dir}")
    print(f"sim:    {sim_dir.relative_to(_REPO) if sim_dir.is_relative_to(_REPO) else sim_dir}")
    print(f"output: {out_dir.relative_to(_REPO) if out_dir.is_relative_to(_REPO) else out_dir}")
    print(f"sim_shift: {args.sim_shift:.3f}s")
    print()

    # Always load qfrc_applied alone — used for mech-comp joints (spring gravity
    # compensation, motor does NOT supply gravity, so we don't add gravcomp).
    applied = read_sim_csv(sim_dir, "qfrc_applied.csv")
    sim_trq_cols_no_grav: dict[str, np.ndarray] | None = None
    if applied is not None:
        _, sim_trq_cols_no_grav = applied

    if args.sim_source == "tau_j":
        # tau_j (motor joint-side output) = qfrc_applied + qfrc_gravcomp.
        # qfrc_applied = our kernel-written tau_j - g_j; passive gravcomp adds g_j back.
        # This matches real `joint_torque` (= motor_states.torque · ratio) in semantics.
        grav = read_sim_csv(sim_dir, "qfrc_gravcomp.csv")
        if applied is None or grav is None:
            print("[ERROR] tau_j needs qfrc_applied + qfrc_gravcomp")
            sys.exit(1)
        t_sim, applied_cols = applied
        _, grav_cols = grav
        sim_trq_cols = {k: applied_cols[k] + grav_cols.get(k, np.zeros_like(applied_cols[k]))
                        for k in applied_cols}
        sim_csv = "qfrc_applied + qfrc_gravcomp"
    else:
        sim_csv = f"{args.sim_source}.csv"
        sim_data = read_sim_csv(sim_dir, sim_csv)
        if sim_data is None:
            sys.exit(1)
        t_sim, sim_trq_cols = sim_data
    print(f"sim trq: {len(t_sim)} samples from {sim_csv}, columns={len(sim_trq_cols)}, "
          f"time=[{t_sim[0]:.2f}, {t_sim[-1]:.2f}]s")

    sim_pos_data = read_sim_csv(sim_dir, "joint_position.csv")
    if sim_pos_data is not None:
        _t_sim_pos, sim_pos_cols_raw = sim_pos_data
        # joint_position.csv columns are prefixed with 'pos_<JointName>' — strip prefix
        sim_pos_cols = {k[4:] if k.startswith("pos_") else k: v
                        for k, v in sim_pos_cols_raw.items()}
        print(f"sim pos: {len(_t_sim_pos)} samples, columns={len(sim_pos_cols)}")
    else:
        sim_pos_cols = None

    # sim's own target (if logged) — preferred over rosbag target for sim_err.
    sim_tgt_data = read_sim_csv(sim_dir, "joint_position_target.csv")
    if sim_tgt_data is not None:
        _t_sim_tgt, sim_tgt_cols_raw = sim_tgt_data
        sim_tgt_cols = {k[4:] if k.startswith("tgt_") else k: v
                        for k, v in sim_tgt_cols_raw.items()}
        print(f"sim tgt: {len(_t_sim_tgt)} samples (sim's own spline target)")
    else:
        sim_tgt_cols = None
        print("sim tgt: not available (falling back to rosbag target for sim err)")
    print()

    selected: set[str] | None = (
        set(args.groups.split(",")) if args.groups else None
    )

    for group, labels in GROUPS:
        if selected is not None and group not in selected:
            continue
        print(f"=== {group} ===")
        real = read_rosbag_topic(rosbag_dir, group, "joint_torque")
        if real is None:
            continue
        t_real, trq_real, t0_ns = real
        print(f"  real trq: {len(t_real)} samples, time=[{t_real[0]:.2f}, {t_real[-1]:.2f}]s, "
              f"shape={trq_real.shape}")

        # Position (actual) and target — same t0_ns so axes align
        pos = read_rosbag_topic(rosbag_dir, group, "joint_positions_deg", t0_ns_ref=t0_ns)
        tgt = read_rosbag_topic(rosbag_dir, group, "joint_ang_target_deg", t0_ns_ref=t0_ns)
        if pos is not None and tgt is not None:
            t_pos, pos_actual_deg, _ = pos
            t_tgt, pos_target_deg, _ = tgt
            print(f"  real pos: {len(t_pos)} samples, shape={pos_actual_deg.shape}; "
                  f"tgt: {len(t_tgt)} samples, shape={pos_target_deg.shape}")
        else:
            t_pos = pos_actual_deg = t_tgt = pos_target_deg = None

        # Velocity (actual) and target
        vel = read_rosbag_topic(rosbag_dir, group, "joint_velocities_deg_s", t0_ns_ref=t0_ns)
        vt  = read_rosbag_topic(rosbag_dir, group, "joint_vel_target_deg_s", t0_ns_ref=t0_ns)
        if vel is not None and vt is not None:
            t_vel, vel_actual_deg, _ = vel
            t_vt,  vel_target_deg, _ = vt
        else:
            t_vel = vel_actual_deg = t_vt = vel_target_deg = None

        plot_group(group, labels, t_real, trq_real, t_sim, sim_trq_cols,
                   args.sim_shift, out_dir, sim_source_label=args.sim_source,
                   t_pos=t_pos, pos_actual_deg=pos_actual_deg,
                   t_tgt=t_tgt, pos_target_deg=pos_target_deg,
                   sim_pos_cols=sim_pos_cols, sim_tgt_cols=sim_tgt_cols,
                   sim_trq_cols_no_grav=sim_trq_cols_no_grav,
                   t_vel=t_vel, vel_actual_deg=vel_actual_deg,
                   t_vt=t_vt, vel_target_deg=vel_target_deg)

    print()
    print(f"Done. Plots in: {out_dir.relative_to(_REPO) if out_dir.is_relative_to(_REPO) else out_dir}")


if __name__ == "__main__":
    main()
