"""Plot K_j_diag(q(t)) and τ_j_max(q(t)) over a trajectory, with USD nominal reference.

q-의존 J 를 가진 coupled group (wrist / finger / thumb) 의 K_j_diag = Σ J(q)²·K_m
및 τ_j_max = |J(q)|^T · τ_m_max 가 trajectory 동안 자세에 따라 얼마나 변하는지
관찰. USD 의 K_j (= q=0 nominal 자세에서 계산된 값) 와 비교하면 motor clip 이
"평균 nominal" 보다 빡빡해지는 구간 (= 토크 부족 의심 구간) 을 시각적으로 확인 가능.

scalar / elbow group 은 J 가 q 무관이라 K_j_diag / τ_j_max 가 상수 → 별도 plot 안 함.

Usage:
    python tools/diagnose_jacobian_gains.py [trajectory_dir]

trajectory_dir 미지정 시 'trajectory/demo1_dynamic_group' 사용.
저장 위치: data/diagnose/jacobian_gains_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# allex/__init__.py 가 isaacsim 모듈을 끌어들이므로 standalone CLI 에선 모듈을 직접 로드.
_REPO = Path(__file__).resolve().parent.parent
_TRAJ_GEN = _REPO / "src" / "allex" / "trajectory_generate"

import importlib.util as _ilu


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mjt = _load_module("motor_joint_transform", _TRAJ_GEN / "motor_joint_transform.py")
hermite = _load_module("hermite_spline", _TRAJ_GEN / "hermite_spline.py")
# hermite._resolve_hz() 가 _load() 를 통해 physics_config 를 찾는데, 우리는 CLI hz 직접 전달이라 안 문제됨.
generate_trajectory_from_csv = hermite.generate_trajectory_from_csv


# ============================================================
# 그룹 정의 — coupled q-dependent J 만 (scalar/elbow 제외)
# ============================================================
# 각 entry: (group_label, csv_filename, joint_col_indices_in_csv, joint_labels, motor_names_for_K_m_lookup)

WRIST_GROUPS = [
    ("L_Wrist", "Arm_L_theOne.csv", [5, 6], ["Roll", "Pitch"],
     ["L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint"]),
    ("R_Wrist", "Arm_R_theOne.csv", [5, 6], ["Roll", "Pitch"],
     ["R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint"]),
]

FINGER_GROUPS = []
for hand in ("L", "R"):
    for finger in ("Index", "Middle", "Ring", "Little"):
        csv = f"Hand_{hand}_{finger.lower()}_wir.csv"
        FINGER_GROUPS.append((
            f"{hand}_{finger}", csv, [0, 1, 2], ["ABAD", "MCP", "PIP"],
            [f"{hand}_{finger}_ABAD_Joint", f"{hand}_{finger}_MCP_Joint",
             f"{hand}_{finger}_PIP_Joint"],
        ))

THUMB_GROUPS = [
    ("L_Thumb", "Hand_L_thumb_wir.csv", [0, 1, 2], ["Yaw", "CMC", "MCP"],
     ["L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint"]),
    ("R_Thumb", "Hand_R_thumb_wir.csv", [0, 1, 2], ["Yaw", "CMC", "MCP"],
     ["R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint"]),
]


# ============================================================
# Helpers
# ============================================================
def load_nominal_motor_gains() -> dict:
    """joint_config.json::nominal_motor_gains."""
    path = _REPO / "src/allex/config/joint_config.json"
    cfg = json.load(open(path))
    return cfg.get("nominal_motor_gains", {})


def load_trajectory_q(csv_path: Path, hz: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate trajectory q(t) [rad] from CSV. Returns (t, q) shape (T,), (T, n_joints)."""
    t, pos, _vel = generate_trajectory_from_csv(csv_path, hz=hz)
    return np.asarray(t), np.asarray(pos)


def compute_wrist_metrics(q_traj: np.ndarray, K_m: np.ndarray, tm: np.ndarray):
    """For each t, compute K_j_diag and τ_j_max from wrist J(qr, qp)."""
    T = q_traj.shape[0]
    K_j = np.zeros((T, 2))
    T_j = np.zeros((T, 2))
    for i in range(T):
        qr, qp = float(q_traj[i, 0]), float(q_traj[i, 1])
        J = mjt.wrist_jacobian(qr, qp)
        K_j[i] = mjt.gain_transform_diag(J, K_m)
        T_j[i] = mjt.torque_limit_transform(J, tm)
    return K_j, T_j


def compute_finger_metrics(q_traj: np.ndarray, K_m: np.ndarray, tm: np.ndarray):
    """Finger J(q1, q2, q3)."""
    T = q_traj.shape[0]
    K_j = np.zeros((T, 3))
    T_j = np.zeros((T, 3))
    for i in range(T):
        q1, q2, q3 = (float(q_traj[i, j]) for j in range(3))
        J = mjt.finger_jacobian(q1, q2, q3)
        K_j[i] = mjt.gain_transform_diag(J, K_m)
        T_j[i] = mjt.torque_limit_transform(J, tm)
    return K_j, T_j


def compute_thumb_metrics(q_traj: np.ndarray, K_m: np.ndarray, tm: np.ndarray):
    """Thumb J(q1, q2, q3) — q1=yaw, q2=cmc, q3=mcp."""
    T = q_traj.shape[0]
    K_j = np.zeros((T, 3))
    T_j = np.zeros((T, 3))
    for i in range(T):
        q1, q2, q3 = (float(q_traj[i, j]) for j in range(3))
        J = mjt.thumb_jacobian(q1, q2, q3)
        K_j[i] = mjt.gain_transform_diag(J, K_m)
        T_j[i] = mjt.torque_limit_transform(J, tm)
    return K_j, T_j


# ============================================================
# Plot helpers
# ============================================================
def plot_group(
    out_dir: Path, group_label: str, joint_labels: list[str],
    t: np.ndarray, K_j: np.ndarray, T_j: np.ndarray,
    K_j_nom: np.ndarray, T_j_nom: np.ndarray, q_traj: np.ndarray,
):
    """3-row figure: q(t), K_j(t), τ_j_max(t). Each panel shows all joints in group."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    n_joints = K_j.shape[1]
    cmap = plt.get_cmap("tab10")

    # Row 0: q(t) in degrees
    ax = axes[0]
    for j in range(n_joints):
        ax.plot(t, np.degrees(q_traj[:, j]), color=cmap(j), label=joint_labels[j])
    ax.set_ylabel("q [deg]")
    ax.set_title(f"{group_label} — joint angles & gain / torque-limit landscape")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Row 1: K_j_diag(t) — runtime vs nominal
    ax = axes[1]
    for j in range(n_joints):
        ax.plot(t, K_j[:, j], color=cmap(j), label=f"{joint_labels[j]} runtime")
        ax.axhline(K_j_nom[j], color=cmap(j), linestyle=":", alpha=0.6,
                   label=f"{joint_labels[j]} nominal (q=0)")
    ax.set_ylabel("K_j_diag [N·m/rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)

    # Row 2: τ_j_max(t) — runtime vs nominal
    ax = axes[2]
    for j in range(n_joints):
        ax.plot(t, T_j[:, j], color=cmap(j), label=f"{joint_labels[j]} runtime")
        ax.axhline(T_j_nom[j], color=cmap(j), linestyle=":", alpha=0.6,
                   label=f"{joint_labels[j]} nominal (q=0)")
    ax.set_ylabel("τ_j_max [N·m]")
    ax.set_xlabel("trajectory time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    out_path = out_dir / f"jacobian_gains_{group_label}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(_REPO)}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory_dir", nargs="?",
                    default="trajectory/demo1_dynamic_group",
                    help="trajectory CSV 디렉토리")
    ap.add_argument("--hz", type=float, default=100.0,
                    help="trajectory 샘플링 주파수 (낮을수록 빠름; default=100)")
    args = ap.parse_args()

    traj_dir = (_REPO / args.trajectory_dir).resolve()
    if not traj_dir.is_dir():
        print(f"[ERROR] trajectory dir not found: {traj_dir}")
        sys.exit(1)

    out_dir = _REPO / "data/diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal = load_nominal_motor_gains()
    print(f"trajectory: {traj_dir.relative_to(_REPO)}")
    print(f"output:     {out_dir.relative_to(_REPO)}")
    print()

    # ─── Wrist ───
    print("=== Wrist groups (J 2×2, q-dependent polynomial) ===")
    for label, csv_name, cols, joint_labels, motor_names in WRIST_GROUPS:
        csv = traj_dir / csv_name
        if not csv.exists():
            print(f"  skip {label}: {csv_name} not found")
            continue
        K_m = np.array([float(nominal[m]["kp"]) for m in motor_names])
        tm  = np.array([float(nominal[m]["trq_lim"]) for m in motor_names])
        t, pos = load_trajectory_q(csv, hz=args.hz)
        q_traj = pos[:, cols]            # (T, 2) for wrist
        K_j, T_j = compute_wrist_metrics(q_traj, K_m, tm)
        J0 = mjt.wrist_jacobian(0.0, 0.0)
        K_j_nom = mjt.gain_transform_diag(J0, K_m)
        T_j_nom = mjt.torque_limit_transform(J0, tm)
        plot_group(out_dir, label, joint_labels, t, K_j, T_j, K_j_nom, T_j_nom, q_traj)

    # ─── Finger ───
    print()
    print("=== Finger groups (J 3×3 lower-triangular, q-dependent polynomial) ===")
    for label, csv_name, cols, joint_labels, motor_names in FINGER_GROUPS:
        csv = traj_dir / csv_name
        if not csv.exists():
            print(f"  skip {label}: {csv_name} not found")
            continue
        K_m = np.array([float(nominal[m]["kp"]) for m in motor_names])
        tm  = np.array([float(nominal[m]["trq_lim"]) for m in motor_names])
        t, pos = load_trajectory_q(csv, hz=args.hz)
        q_traj = pos[:, cols]
        K_j, T_j = compute_finger_metrics(q_traj, K_m, tm)
        J0 = mjt.finger_jacobian(0.0, 0.0, 0.0)
        K_j_nom = mjt.gain_transform_diag(J0, K_m)
        T_j_nom = mjt.torque_limit_transform(J0, tm)
        plot_group(out_dir, label, joint_labels, t, K_j, T_j, K_j_nom, T_j_nom, q_traj)

    # ─── Thumb ───
    print()
    print("=== Thumb groups (J 3×3 sparse, q-dependent polynomial in q2,q3) ===")
    for label, csv_name, cols, joint_labels, motor_names in THUMB_GROUPS:
        csv = traj_dir / csv_name
        if not csv.exists():
            print(f"  skip {label}: {csv_name} not found")
            continue
        K_m = np.array([float(nominal[m]["kp"]) for m in motor_names])
        tm  = np.array([float(nominal[m]["trq_lim"]) for m in motor_names])
        t, pos = load_trajectory_q(csv, hz=args.hz)
        q_traj = pos[:, cols]
        K_j, T_j = compute_thumb_metrics(q_traj, K_m, tm)
        J0 = mjt.thumb_jacobian(0.0, 0.0, 0.0)
        K_j_nom = mjt.gain_transform_diag(J0, K_m)
        T_j_nom = mjt.torque_limit_transform(J0, tm)
        plot_group(out_dir, label, joint_labels, t, K_j, T_j, K_j_nom, T_j_nom, q_traj)

    print()
    print(f"Done. Plots in: {out_dir.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
