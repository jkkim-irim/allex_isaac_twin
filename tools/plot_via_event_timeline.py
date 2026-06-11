"""Plot trajectory via event timeline — K_pos / K_vel / trq_lim over time per joint.

trajectory CSV 의 K_pos_* / K_vel_* / trq_lim_* 컬럼이 via point 단위로 모터-단 값을
설정. 이 값들이 시간축으로 어떻게 변하는지 step plot (zero-order hold) 으로 시각화.

사용 목적:
    - 각 trajectory 시점에 모터 게인 / 한계 가 어떻게 변하는지 확인
    - real / sim torque 스파이크 시점과 게인 transition 시점 정렬
    - 동적 구간에서 게인이 떨어지는지 (compliance), 한계 가 작아지는지 확인

NaN cell 은 "변경 없음" 으로 해석 (직전 값 유지). 첫 sample 의 NaN 은 nominal (joint_config) 값 사용.

Usage:
    python tools/plot_via_event_timeline.py [trajectory_dir]
    default trajectory_dir = trajectory/demo1_dynamic_group
저장: data/diagnose/via_event_timeline/<group>.png
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
_TG   = _REPO / "src" / "allex" / "trajectory_generate"


def _load_module(name: str, path: Path):
    spec = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


hermite = _load_module("hermite_spline", _TG / "hermite_spline.py")
parse_via_csv = hermite.parse_via_csv


# CSV 별: (csv_name, joint_labels for plot legend, motor_names for nominal lookup)
GROUPS = [
    ("theOne_waist", ["Waist_Yaw", "Waist_Lower_Pitch"],
     ["Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint"]),
    ("theOne_neck", ["Neck_Pitch", "Neck_Yaw"],
     ["Neck_Pitch_Joint", "Neck_Yaw_Joint"]),
    ("Arm_L_theOne",
     ["L_Sh_Pitch","L_Sh_Roll","L_Sh_Yaw","L_Elbow","L_W_Yaw","L_W_Roll","L_W_Pitch"],
     ["L_Shoulder_Pitch_Joint","L_Shoulder_Roll_Joint","L_Shoulder_Yaw_Joint",
      "L_Elbow_Joint","L_Wrist_Yaw_Joint","L_Wrist_Roll_Joint","L_Wrist_Pitch_Joint"]),
    ("Arm_R_theOne",
     ["R_Sh_Pitch","R_Sh_Roll","R_Sh_Yaw","R_Elbow","R_W_Yaw","R_W_Roll","R_W_Pitch"],
     ["R_Shoulder_Pitch_Joint","R_Shoulder_Roll_Joint","R_Shoulder_Yaw_Joint",
      "R_Elbow_Joint","R_Wrist_Yaw_Joint","R_Wrist_Roll_Joint","R_Wrist_Pitch_Joint"]),
    ("Hand_L_thumb_wir", ["L_Thumb_Yaw","L_Thumb_CMC","L_Thumb_MCP"],
     ["L_Thumb_Yaw_Joint","L_Thumb_CMC_Joint","L_Thumb_MCP_Joint"]),
    ("Hand_L_index_wir", ["L_Index_ABAD","L_Index_MCP","L_Index_PIP"],
     ["L_Index_ABAD_Joint","L_Index_MCP_Joint","L_Index_PIP_Joint"]),
    ("Hand_L_middle_wir",["L_Middle_ABAD","L_Middle_MCP","L_Middle_PIP"],
     ["L_Middle_ABAD_Joint","L_Middle_MCP_Joint","L_Middle_PIP_Joint"]),
    ("Hand_L_ring_wir",  ["L_Ring_ABAD","L_Ring_MCP","L_Ring_PIP"],
     ["L_Ring_ABAD_Joint","L_Ring_MCP_Joint","L_Ring_PIP_Joint"]),
    ("Hand_L_little_wir",["L_Little_ABAD","L_Little_MCP","L_Little_PIP"],
     ["L_Little_ABAD_Joint","L_Little_MCP_Joint","L_Little_PIP_Joint"]),
    ("Hand_R_thumb_wir", ["R_Thumb_Yaw","R_Thumb_CMC","R_Thumb_MCP"],
     ["R_Thumb_Yaw_Joint","R_Thumb_CMC_Joint","R_Thumb_MCP_Joint"]),
    ("Hand_R_index_wir", ["R_Index_ABAD","R_Index_MCP","R_Index_PIP"],
     ["R_Index_ABAD_Joint","R_Index_MCP_Joint","R_Index_PIP_Joint"]),
    ("Hand_R_middle_wir",["R_Middle_ABAD","R_Middle_MCP","R_Middle_PIP"],
     ["R_Middle_ABAD_Joint","R_Middle_MCP_Joint","R_Middle_PIP_Joint"]),
    ("Hand_R_ring_wir",  ["R_Ring_ABAD","R_Ring_MCP","R_Ring_PIP"],
     ["R_Ring_ABAD_Joint","R_Ring_MCP_Joint","R_Ring_PIP_Joint"]),
    ("Hand_R_little_wir",["R_Little_ABAD","R_Little_MCP","R_Little_PIP"],
     ["R_Little_ABAD_Joint","R_Little_MCP_Joint","R_Little_PIP_Joint"]),
]


def zoh_fill(t_via: np.ndarray, values: np.ndarray, initial_value: float):
    """NaN cell 을 직전 non-NaN 값으로 채움 (zero-order hold).

    첫 sample NaN 이면 initial_value 로 시작.
    """
    out = values.astype(np.float64).copy()
    cur = initial_value
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = cur
        else:
            cur = out[i]
    return out


def plot_group(out_dir: Path, csv_name: str, joint_labels: list[str],
               motor_names: list[str], traj_dir: Path, nominal: dict):
    csv = traj_dir / f"{csv_name}.csv"
    if not csv.is_file():
        print(f"  skip {csv_name}: not found")
        return
    data = parse_via_csv(csv)
    t = data.t_via                       # (N,)
    n_joints = data.pos_via.shape[1]
    if n_joints != len(joint_labels):
        print(f"  ⚠ {csv_name}: shape {n_joints} != labels {len(joint_labels)}")
        n_joints = min(n_joints, len(joint_labels))

    # Build series per joint
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    cmap = plt.get_cmap("tab10")

    for j in range(n_joints):
        nom = nominal.get(motor_names[j], {})
        k_pos_nom = float(nom.get("kp", 0.0))
        k_vel_nom = float(nom.get("kv", 0.0))
        tlim_nom  = float(nom.get("trq_lim", 0.0))

        if data.kps_via is not None:
            series = zoh_fill(t, data.kps_via[:, j], k_pos_nom)
            axes[0].step(t, series, where="post", color=cmap(j),
                         label=joint_labels[j], lw=1.0)
        if data.kds_via is not None:
            series = zoh_fill(t, data.kds_via[:, j], k_vel_nom)
            axes[1].step(t, series, where="post", color=cmap(j),
                         label=joint_labels[j], lw=1.0)
        if data.trq_via is not None:
            series = zoh_fill(t, data.trq_via[:, j], tlim_nom)
            axes[2].step(t, series, where="post", color=cmap(j),
                         label=joint_labels[j], lw=1.0)

    axes[0].set_ylabel("K_pos (motor)")
    axes[1].set_ylabel("K_vel (motor)")
    axes[2].set_ylabel("τ_m_max (motor)")
    axes[2].set_xlabel("trajectory time [s]")
    axes[0].set_title(f"{csv_name} — via event timeline (motor-domain values, ZOH)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    out_path = out_dir / f"events_{csv_name}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(_REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory_dir", nargs="?",
                    default="trajectory/demo1_dynamic_group")
    ap.add_argument("--out", default="data/diagnose/via_event_timeline")
    args = ap.parse_args()

    traj_dir = (_REPO / args.trajectory_dir).resolve()
    out_dir  = (_REPO / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal = json.load(open(_REPO / "src/allex/config/joint_config.json"))["nominal_motor_gains"]
    print(f"trajectory: {traj_dir.relative_to(_REPO)}")
    print(f"output:     {out_dir.relative_to(_REPO)}")
    print()

    for csv_name, joint_labels, motor_names in GROUPS:
        print(f"=== {csv_name} ===")
        plot_group(out_dir, csv_name, joint_labels, motor_names, traj_dir, nominal)

    print(f"\nDone. Plots in: {out_dir.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
