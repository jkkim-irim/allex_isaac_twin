"""Unit tests for jk_kernel.py — verify warp kernel outputs match numpy
transform_group reference (motor_joint_transform.py).

Run:
    python tools/test_jk_kernel.py

Each test launches the warp kernel on a representative input, then
compares against the numpy reference computed via transform_group from
motor_joint_transform.py. Tolerance: rtol=1e-5 (warp uses float32, so
~6 decimal digits expected).

테스트는 standalone 으로 동작 — Isaac Sim 런타임 불필요.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp


EXT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Bypass package __init__ (which pulls isaacsim.core). Load standalone.
_mt = _load_module("_mt", EXT_ROOT / "src/allex/trajectory_generate/motor_joint_transform.py")
_jk = _load_module("_jk", EXT_ROOT / "src/allex/trajectory_generate/jk_kernel.py")


def _run_scalar_groups(
    n_motors: int,
    ratio_np: np.ndarray,
    K_m_np: np.ndarray,
    Kv_m_np: np.ndarray,
    trq_m_np: np.ndarray,
    *,
    n_dof: int,
    dof_idx_np: np.ndarray,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Launch apply_scalar_groups; return (out_ke, out_kd, out_eff) numpy arrays
    of size n_dof (only entries at dof_idx are populated)."""
    dof_idx = wp.array(dof_idx_np.astype(np.int32), dtype=int, device=device)
    ratio = wp.array(ratio_np.astype(np.float32), dtype=float, device=device)
    K_m = wp.array(K_m_np.astype(np.float32), dtype=float, device=device)
    Kv_m = wp.array(Kv_m_np.astype(np.float32), dtype=float, device=device)
    trq_m = wp.array(trq_m_np.astype(np.float32), dtype=float, device=device)

    out_ke = wp.zeros(n_dof, dtype=float, device=device)
    out_kd = wp.zeros(n_dof, dtype=float, device=device)
    out_eff = wp.zeros(n_dof, dtype=float, device=device)

    wp.launch(
        kernel=_jk.apply_scalar_groups,
        dim=n_motors,
        inputs=[dof_idx, ratio, K_m, Kv_m, trq_m, out_ke, out_kd, out_eff],
        device=device,
    )
    wp.synchronize()

    return out_ke.numpy(), out_kd.numpy(), out_eff.numpy()


def test_scalar_shoulder_waist_neck() -> None:
    """Verify apply_scalar_groups produces the same K_j as numpy transform_group
    for shoulder (ratio=40.54), waist (ratios=[50,25]), neck (ratios=[7.75,10]).

    Layout (for the test):
        motor 0..2  → shoulder L Pitch/Roll/Yaw  (ratio=40.54)
        motor 3..5  → shoulder R Pitch/Roll/Yaw  (ratio=40.54)
        motor 6..7  → waist Yaw, Lower_Pitch     (ratios=[50, 25])
        motor 8..9  → neck Pitch, Yaw            (ratios=[7.75, 10])

    DOF mapping (test fixture, n_dof=10): identity (motor i → dof i).
    """
    n_motors = 10
    n_dof = 10
    dof_idx_np = np.arange(10, dtype=np.int64)

    # Ratios per motor — must match motor_joint_transform.py constants.
    ratios = np.array([
        _mt.SHOULDER_RATIO, _mt.SHOULDER_RATIO, _mt.SHOULDER_RATIO,   # L
        _mt.SHOULDER_RATIO, _mt.SHOULDER_RATIO, _mt.SHOULDER_RATIO,   # R
        _mt.WAIST_RATIOS[0], _mt.WAIST_RATIOS[1],                     # waist
        _mt.NECK_RATIOS[0], _mt.NECK_RATIOS[1],                       # neck
    ], dtype=np.float64)

    # Sample motor values.
    K_m = np.array([3.0, 3.0, 3.0,  3.0, 3.0, 3.0,  3.0, 1.0,  4.0, 4.0])
    Kv_m = np.array([0.005]*6 + [0.01, 0.10] + [0.005, 0.005])
    trq_m = np.array([1.2]*6 + [0.5, 1.0] + [0.3, 0.23])

    # Run kernel.
    ke, kd, eff = _run_scalar_groups(
        n_motors, ratios, K_m, Kv_m, trq_m,
        n_dof=n_dof, dof_idx_np=dof_idx_np,
    )

    # Reference via motor_joint_transform.transform_group on each scalar group.
    # Shoulder (3 DOF, single ratio).
    ref_ke = np.zeros(n_dof, dtype=np.float64)
    ref_kd = np.zeros(n_dof)
    ref_eff = np.zeros(n_dof)

    # L shoulder.
    out_l = _mt.transform_group("shoulder", np.zeros(3),
                                k_pos=K_m[0:3], k_vel=Kv_m[0:3], trq_lim=trq_m[0:3])
    ref_ke[0:3] = out_l["k_pos"]; ref_kd[0:3] = out_l["k_vel"]; ref_eff[0:3] = out_l["trq_lim"]
    # R shoulder.
    out_r = _mt.transform_group("shoulder", np.zeros(3),
                                k_pos=K_m[3:6], k_vel=Kv_m[3:6], trq_lim=trq_m[3:6])
    ref_ke[3:6] = out_r["k_pos"]; ref_kd[3:6] = out_r["k_vel"]; ref_eff[3:6] = out_r["trq_lim"]
    # Waist.
    out_w = _mt.transform_group("waist", np.zeros(2),
                                k_pos=K_m[6:8], k_vel=Kv_m[6:8], trq_lim=trq_m[6:8])
    ref_ke[6:8] = out_w["k_pos"]; ref_kd[6:8] = out_w["k_vel"]; ref_eff[6:8] = out_w["trq_lim"]
    # Neck.
    out_n = _mt.transform_group("neck", np.zeros(2),
                                k_pos=K_m[8:10], k_vel=Kv_m[8:10], trq_lim=trq_m[8:10])
    ref_ke[8:10] = out_n["k_pos"]; ref_kd[8:10] = out_n["k_vel"]; ref_eff[8:10] = out_n["trq_lim"]

    # Compare.
    rtol, atol = 1e-5, 1e-7
    assert np.allclose(ke, ref_ke, rtol=rtol, atol=atol), \
        f"K_j mismatch:\n  kernel = {ke}\n  ref    = {ref_ke}\n  diff   = {ke - ref_ke}"
    assert np.allclose(kd, ref_kd, rtol=rtol, atol=atol), \
        f"D_j mismatch:\n  kernel = {kd}\n  ref    = {ref_kd}\n  diff   = {kd - ref_kd}"
    assert np.allclose(eff, ref_eff, rtol=rtol, atol=atol), \
        f"τ_j mismatch:\n  kernel = {eff}\n  ref    = {ref_eff}\n  diff   = {eff - ref_eff}"

    print(f"[OK] apply_scalar_groups: 10 motors × 3 categories matched ref (rtol={rtol})")
    print(f"     shoulder K_j sample = {ke[0]:.4f} (expected ratio²·K_m = {40.53846154**2 * 3.0:.4f})")
    print(f"     waist K_j           = {ke[6]:.4f}, {ke[7]:.4f}")
    print(f"     neck K_j            = {ke[8]:.4f}, {ke[9]:.4f}")


def _run_elbow_K_j(
    K_m_np: np.ndarray,
    Kv_m_np: np.ndarray,
    trq_m_np: np.ndarray,
    *,
    n_dof: int,
    dof_idx_np: np.ndarray,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Launch compute_elbow_K_j; expects 4 motors total (2 hands × 2)."""
    n_hands = K_m_np.size // 2
    assert n_hands * 2 == K_m_np.size, "elbow K_m size must be multiple of 2"

    dof_idx = wp.array(dof_idx_np.astype(np.int32), dtype=int, device=device)
    K_m = wp.array(K_m_np.astype(np.float32), dtype=float, device=device)
    Kv_m = wp.array(Kv_m_np.astype(np.float32), dtype=float, device=device)
    trq_m = wp.array(trq_m_np.astype(np.float32), dtype=float, device=device)

    out_ke = wp.zeros(n_dof, dtype=float, device=device)
    out_kd = wp.zeros(n_dof, dtype=float, device=device)
    out_eff = wp.zeros(n_dof, dtype=float, device=device)

    wp.launch(
        kernel=_jk.compute_elbow_K_j,
        dim=n_hands,
        inputs=[dof_idx, K_m, Kv_m, trq_m, out_ke, out_kd, out_eff],
        device=device,
    )
    wp.synchronize()
    return out_ke.numpy(), out_kd.numpy(), out_eff.numpy()


def test_elbow_constant_J() -> None:
    """Verify compute_elbow_K_j produces same K_j as numpy transform_group("elbow", ...).

    Layout:
        2 hands × (Elbow_Joint, Wrist_Yaw_Joint) = 4 DOFs
        dof_idx_np maps motor to articulation DOF index
        Test fixture: identity (motor i → dof i)
    """
    n_dof = 4
    dof_idx_np = np.array([0, 1, 2, 3], dtype=np.int64)

    # Test 1: same K_m for both motors per hand → same answer for both hands.
    K_m = np.array([3.0, 3.0,  3.0, 3.0])
    Kv_m = np.array([0.005, 0.005,  0.005, 0.005])
    trq_m = np.array([1.0, 1.0,  1.0, 1.0])

    ke, kd, eff = _run_elbow_K_j(K_m, Kv_m, trq_m, n_dof=n_dof, dof_idx_np=dof_idx_np)

    # Reference via transform_group.
    ref_ke = np.zeros(n_dof)
    ref_kd = np.zeros(n_dof)
    ref_eff = np.zeros(n_dof)
    for h in range(2):
        out = _mt.transform_group("elbow", np.zeros(2),
                                  k_pos=K_m[h*2:h*2+2], k_vel=Kv_m[h*2:h*2+2], trq_lim=trq_m[h*2:h*2+2])
        ref_ke[h*2:h*2+2] = out["k_pos"]
        ref_kd[h*2:h*2+2] = out["k_vel"]
        ref_eff[h*2:h*2+2] = out["trq_lim"]

    rtol, atol = 1e-5, 1e-7
    assert np.allclose(ke, ref_ke, rtol=rtol, atol=atol), \
        f"K_j mismatch:\n  kernel={ke}\n  ref   ={ref_ke}\n  diff  ={ke-ref_ke}"
    assert np.allclose(kd, ref_kd, rtol=rtol, atol=atol), \
        f"D_j mismatch:\n  kernel={kd}\n  ref   ={ref_kd}"
    assert np.allclose(eff, ref_eff, rtol=rtol, atol=atol), \
        f"τ_j mismatch:\n  kernel={eff}\n  ref   ={ref_eff}"

    print(f"[OK] compute_elbow_K_j (uniform K_m): matched ref")
    print(f"     L_Elbow K_j = {ke[0]:.4f}, L_Wrist_Yaw K_j = {ke[1]:.4f}")
    print(f"     L_Elbow τ_j = {eff[0]:.4f}, L_Wrist_Yaw τ_j = {eff[1]:.4f}")

    # Test 2: cross-coupling — change motor[0] only, both joints should respond.
    K_m_v2 = np.array([5.0, 1.0,  3.0, 3.0])
    ke2, _, _ = _run_elbow_K_j(K_m_v2, Kv_m, trq_m, n_dof=n_dof, dof_idx_np=dof_idx_np)
    out_l = _mt.transform_group("elbow", np.zeros(2), k_pos=K_m_v2[0:2])
    assert np.allclose(ke2[0:2], out_l["k_pos"], rtol=rtol, atol=atol), \
        f"cross-coupling check failed: kernel={ke2[0:2]}, ref={out_l['k_pos']}"
    print(f"[OK] compute_elbow_K_j (asymmetric K_m=[5,1]): cross-coupling verified")
    print(f"     L_Elbow K_j = {ke2[0]:.4f} (depends on BOTH motors)")
    print(f"     L_Wrist_Yaw K_j = {ke2[1]:.4f} (depends on BOTH motors)")


def _run_wrist_K_j(
    K_m_np: np.ndarray,
    Kv_m_np: np.ndarray,
    trq_m_np: np.ndarray,
    joint_q_full: np.ndarray,
    dof_idx_np: np.ndarray,
    *,
    n_dof: int,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Launch compute_wrist_K_j; expects 4 motors total (2 hands × 2)."""
    n_hands = K_m_np.size // 2
    dof_idx = wp.array(dof_idx_np.astype(np.int32), dtype=int, device=device)
    K_m = wp.array(K_m_np.astype(np.float32), dtype=float, device=device)
    Kv_m = wp.array(Kv_m_np.astype(np.float32), dtype=float, device=device)
    trq_m = wp.array(trq_m_np.astype(np.float32), dtype=float, device=device)
    joint_q = wp.array(joint_q_full.astype(np.float32), dtype=float, device=device)

    out_ke = wp.zeros(n_dof, dtype=float, device=device)
    out_kd = wp.zeros(n_dof, dtype=float, device=device)
    out_eff = wp.zeros(n_dof, dtype=float, device=device)

    wp.launch(
        kernel=_jk.compute_wrist_K_j,
        dim=n_hands,
        inputs=[dof_idx, K_m, Kv_m, trq_m, joint_q, out_ke, out_kd, out_eff],
        device=device,
    )
    wp.synchronize()
    return out_ke.numpy(), out_kd.numpy(), out_eff.numpy()


def test_wrist_q_dependent() -> None:
    """Verify compute_wrist_K_j vs numpy transform_group at multiple q points."""
    DEG = math.pi / 180
    samples = [
        ("q=0",         np.array([0.0, 0.0])),
        ("q=10°,20°",   np.array([10*DEG, 20*DEG])),
        ("q=-30°,45°",  np.array([-30*DEG, 45*DEG])),
        ("q=60°,-15°",  np.array([60*DEG, -15*DEG])),
    ]

    K_m = np.array([0.3, 0.3,  0.3, 0.3])  # L_roll, L_pitch, R_roll, R_pitch (per L_Wrist_Roll/Pitch joint dof)
    Kv_m = np.array([0.001, 0.001,  0.001, 0.001])
    trq_m = np.array([0.12, 0.12,  0.12, 0.12])

    n_dof = 4
    dof_idx_np = np.array([0, 1, 2, 3], dtype=np.int64)  # identity mapping

    for label, q_per_hand in samples:
        # Build full joint_q array (size n_dof). Same q for both hands in this test.
        joint_q = np.zeros(n_dof, dtype=np.float64)
        joint_q[0] = q_per_hand[0]  # L roll
        joint_q[1] = q_per_hand[1]  # L pitch
        joint_q[2] = q_per_hand[0]  # R roll
        joint_q[3] = q_per_hand[1]  # R pitch

        ke, kd, eff = _run_wrist_K_j(K_m, Kv_m, trq_m, joint_q, dof_idx_np, n_dof=n_dof)

        out = _mt.transform_group("wrist", q_per_hand,
                                  k_pos=K_m[0:2], k_vel=Kv_m[0:2], trq_lim=trq_m[0:2])
        ref_ke = out["k_pos"]
        ref_kd = out["k_vel"]
        ref_eff = out["trq_lim"]

        rtol, atol = 5e-4, 1e-5  # float32 ~6-digit, polynomial accumulation slack
        for h in range(2):
            assert np.allclose(ke[h*2:h*2+2], ref_ke, rtol=rtol, atol=atol), \
                f"{label} hand={h} K_j mismatch:\n  kernel={ke[h*2:h*2+2]}\n  ref   ={ref_ke}"
            assert np.allclose(kd[h*2:h*2+2], ref_kd, rtol=rtol, atol=atol), \
                f"{label} hand={h} D_j mismatch"
            assert np.allclose(eff[h*2:h*2+2], ref_eff, rtol=rtol, atol=atol), \
                f"{label} hand={h} τ_j mismatch:\n  kernel={eff[h*2:h*2+2]}\n  ref   ={ref_eff}"
        print(f"[OK] compute_wrist_K_j {label}: K_j={ref_ke[0]:.4f}, {ref_ke[1]:.4f}")


def _run_finger_K_j(
    K_m_np: np.ndarray,
    Kv_m_np: np.ndarray,
    trq_m_np: np.ndarray,
    joint_q_full: np.ndarray,
    dof_idx_np: np.ndarray,
    *,
    n_dof: int,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_inst = K_m_np.size // 3
    dof_idx = wp.array(dof_idx_np.astype(np.int32), dtype=int, device=device)
    K_m = wp.array(K_m_np.astype(np.float32), dtype=float, device=device)
    Kv_m = wp.array(Kv_m_np.astype(np.float32), dtype=float, device=device)
    trq_m = wp.array(trq_m_np.astype(np.float32), dtype=float, device=device)
    joint_q = wp.array(joint_q_full.astype(np.float32), dtype=float, device=device)

    out_ke = wp.zeros(n_dof, dtype=float, device=device)
    out_kd = wp.zeros(n_dof, dtype=float, device=device)
    out_eff = wp.zeros(n_dof, dtype=float, device=device)

    wp.launch(
        kernel=_jk.compute_finger_K_j,
        dim=n_inst,
        inputs=[dof_idx, K_m, Kv_m, trq_m, joint_q, out_ke, out_kd, out_eff],
        device=device,
    )
    wp.synchronize()
    return out_ke.numpy(), out_kd.numpy(), out_eff.numpy()


def test_finger_q_dependent() -> None:
    """Verify compute_finger_K_j vs numpy transform_group at multiple q points."""
    DEG = math.pi / 180
    samples = [
        ("q=0",            np.array([0.0, 0.0, 0.0])),
        ("q=0,30°,45°",    np.array([0.0, 30*DEG, 45*DEG])),
        ("q=10°,60°,70°",  np.array([10*DEG, 60*DEG, 70*DEG])),
        ("q=-15°,20°,30°", np.array([-15*DEG, 20*DEG, 30*DEG])),
    ]

    K_m = np.array([0.0012, 0.0012, 0.005])  # ABAD/MCP/PIP nominal motor kp
    Kv_m = np.array([0.00002, 0.00002, 0.00002])
    trq_m = np.array([0.00386, 0.00386, 0.01])

    n_dof = 3
    dof_idx_np = np.array([0, 1, 2], dtype=np.int64)  # identity

    for label, q in samples:
        ke, kd, eff = _run_finger_K_j(K_m, Kv_m, trq_m, q, dof_idx_np, n_dof=n_dof)

        out = _mt.transform_group("finger", q, k_pos=K_m, k_vel=Kv_m, trq_lim=trq_m)
        ref_ke = out["k_pos"]
        ref_kd = out["k_vel"]
        ref_eff = out["trq_lim"]

        rtol, atol = 1e-3, 1e-6
        assert np.allclose(ke, ref_ke, rtol=rtol, atol=atol), \
            f"{label} K_j mismatch:\n  kernel={ke}\n  ref   ={ref_ke}\n  diff  ={ke-ref_ke}"
        assert np.allclose(kd, ref_kd, rtol=rtol, atol=atol), \
            f"{label} D_j mismatch"
        assert np.allclose(eff, ref_eff, rtol=rtol, atol=atol), \
            f"{label} τ_j mismatch:\n  kernel={eff}\n  ref={ref_eff}"
        print(f"[OK] compute_finger_K_j {label}: K_j={ref_ke[0]:.5f}, {ref_ke[1]:.5f}, {ref_ke[2]:.5f}")


def _run_thumb_K_j(
    K_m_np: np.ndarray,
    Kv_m_np: np.ndarray,
    trq_m_np: np.ndarray,
    joint_q_full: np.ndarray,
    dof_idx_np: np.ndarray,
    *,
    n_dof: int,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_inst = K_m_np.size // 3
    dof_idx = wp.array(dof_idx_np.astype(np.int32), dtype=int, device=device)
    K_m = wp.array(K_m_np.astype(np.float32), dtype=float, device=device)
    Kv_m = wp.array(Kv_m_np.astype(np.float32), dtype=float, device=device)
    trq_m = wp.array(trq_m_np.astype(np.float32), dtype=float, device=device)
    joint_q = wp.array(joint_q_full.astype(np.float32), dtype=float, device=device)

    out_ke = wp.zeros(n_dof, dtype=float, device=device)
    out_kd = wp.zeros(n_dof, dtype=float, device=device)
    out_eff = wp.zeros(n_dof, dtype=float, device=device)

    wp.launch(
        kernel=_jk.compute_thumb_K_j,
        dim=n_inst,
        inputs=[dof_idx, K_m, Kv_m, trq_m, joint_q, out_ke, out_kd, out_eff],
        device=device,
    )
    wp.synchronize()
    return out_ke.numpy(), out_kd.numpy(), out_eff.numpy()


def test_thumb_q_dependent() -> None:
    """Verify compute_thumb_K_j vs numpy transform_group at multiple q points.
    Note: q1 (Yaw) doesn't affect J — only q2, q3 used."""
    DEG = math.pi / 180
    samples = [
        ("q=0",            np.array([0.0, 0.0, 0.0])),
        ("q=0,20°,30°",    np.array([0.0, 20*DEG, 30*DEG])),
        ("q=10°,40°,50°",  np.array([10*DEG, 40*DEG, 50*DEG])),
        ("q=-15°,60°,70°", np.array([-15*DEG, 60*DEG, 70*DEG])),
    ]

    K_m = np.array([0.00193, 0.01, 0.01])      # nominal Yaw/CMC/MCP motor kp
    Kv_m = np.array([0.00001, 0.00001, 0.00001])
    trq_m = np.array([0.00193, 0.003, 0.003])

    n_dof = 3
    dof_idx_np = np.array([0, 1, 2], dtype=np.int64)  # identity

    for label, q in samples:
        ke, kd, eff = _run_thumb_K_j(K_m, Kv_m, trq_m, q, dof_idx_np, n_dof=n_dof)

        out = _mt.transform_group("thumb", q, k_pos=K_m, k_vel=Kv_m, trq_lim=trq_m)
        ref_ke = out["k_pos"]
        ref_kd = out["k_vel"]
        ref_eff = out["trq_lim"]

        # 15차 다항식의 float32 누적 → 더 큰 tolerance
        rtol, atol = 5e-3, 1e-5
        assert np.allclose(ke, ref_ke, rtol=rtol, atol=atol), \
            f"{label} K_j mismatch:\n  kernel={ke}\n  ref   ={ref_ke}\n  diff  ={ke-ref_ke}"
        assert np.allclose(kd, ref_kd, rtol=rtol, atol=atol), \
            f"{label} D_j mismatch"
        assert np.allclose(eff, ref_eff, rtol=rtol, atol=atol), \
            f"{label} τ_j mismatch"
        print(f"[OK] compute_thumb_K_j {label}: K_j={ref_ke[0]:.5f}, {ref_ke[1]:.5f}, {ref_ke[2]:.5f}")


def _load_msm_with_fake_pkg():
    """Load motor_state_mirror.py under a fake package so relative imports work.

    motor_state_mirror does:
        from . import jk_kernel
        from .motor_joint_transform import ...
        from ..utils.sim_settings_utils import get_nominal_motor_gains
    We materialize a fake `_fake_allex` parent + `_fake_allex.trajectory_generate`
    + `_fake_allex.utils` and register all needed submodules in sys.modules.
    """
    import types

    # Fake parent pkg
    pkg = types.ModuleType("_fake_allex"); pkg.__path__ = []
    sys.modules["_fake_allex"] = pkg
    traj_pkg = types.ModuleType("_fake_allex.trajectory_generate"); traj_pkg.__path__ = []
    sys.modules["_fake_allex.trajectory_generate"] = traj_pkg
    utils_pkg = types.ModuleType("_fake_allex.utils"); utils_pkg.__path__ = []
    sys.modules["_fake_allex.utils"] = utils_pkg

    # Load motor_joint_transform under fake namespace
    spec = importlib.util.spec_from_file_location(
        "_fake_allex.trajectory_generate.motor_joint_transform",
        EXT_ROOT / "src/allex/trajectory_generate/motor_joint_transform.py",
    )
    mt2 = importlib.util.module_from_spec(spec)
    sys.modules["_fake_allex.trajectory_generate.motor_joint_transform"] = mt2
    spec.loader.exec_module(mt2)

    # Load jk_kernel under fake namespace
    spec = importlib.util.spec_from_file_location(
        "_fake_allex.trajectory_generate.jk_kernel",
        EXT_ROOT / "src/allex/trajectory_generate/jk_kernel.py",
    )
    jk2 = importlib.util.module_from_spec(spec)
    sys.modules["_fake_allex.trajectory_generate.jk_kernel"] = jk2
    spec.loader.exec_module(jk2)

    # Load sim_settings_utils under fake namespace
    spec = importlib.util.spec_from_file_location(
        "_fake_allex.utils.sim_settings_utils",
        EXT_ROOT / "src/allex/utils/sim_settings_utils.py",
    )
    ssu2 = importlib.util.module_from_spec(spec)
    sys.modules["_fake_allex.utils.sim_settings_utils"] = ssu2
    spec.loader.exec_module(ssu2)

    # Now load motor_state_mirror — its relative imports will resolve.
    spec = importlib.util.spec_from_file_location(
        "_fake_allex.trajectory_generate.motor_state_mirror",
        EXT_ROOT / "src/allex/trajectory_generate/motor_state_mirror.py",
    )
    msm = importlib.util.module_from_spec(spec)
    sys.modules["_fake_allex.trajectory_generate.motor_state_mirror"] = msm
    spec.loader.exec_module(msm)
    return msm, ssu2


def test_motor_state_mirror_end_to_end() -> None:
    """Build MotorStateMirror with fake articulation/model/solver, run update(),
    verify joint_target_ke contents match per-group transform_group references.

    Validates: layout consistency, kernel orchestration, view writes.
    """
    _msm, _ssu = _load_msm_with_fake_pkg()
    _jmap = _load_module(
        "_jmap", EXT_ROOT / "src/allex/trajectory_generate/joint_name_map.py"
    )
    dof_names: list[str] = []
    seen: set[str] = set()
    for stem, names in _jmap.ALLEX_CSV_JOINT_NAMES.items():
        for n in names:
            if n not in seen:
                seen.add(n)
                dof_names.append(n)
    n_dof = len(dof_names)
    assert n_dof == 48, f"expected 48 active joints, got {n_dof}"

    # Fake articulation.
    class _FakeArticulation:
        pass
    art = _FakeArticulation()
    art.dof_names = dof_names

    # Fake Newton model with warp arrays.
    device = "cuda:0"
    class _FakeModel:
        pass
    model = _FakeModel()
    model.joint_q = wp.zeros(n_dof, dtype=float, device=device)
    model.joint_target_ke = wp.zeros(n_dof, dtype=float, device=device)
    model.joint_target_kd = wp.zeros(n_dof, dtype=float, device=device)
    model.joint_effort_limit = wp.zeros(n_dof, dtype=float, device=device)

    # Fake solver — no-op sync.
    sync_calls = [0]
    class _FakeSolver:
        def _update_joint_dof_properties(self):
            sync_calls[0] += 1
    solver = _FakeSolver()

    # Build mirror.
    mirror = _msm.MotorStateMirror(art, model, solver, device=device)
    print(f"[OK] MotorStateMirror built (managed motors={mirror.num_motors()})")

    # Set some q values: bend the L Index finger, neutral elsewhere.
    DEG = math.pi / 180
    q_host = np.zeros(n_dof, dtype=np.float32)
    abad_dof = dof_names.index("L_Index_ABAD_Joint")
    mcp_dof  = dof_names.index("L_Index_MCP_Joint")
    pip_dof  = dof_names.index("L_Index_PIP_Joint")
    q_host[abad_dof] = 0.0
    q_host[mcp_dof]  = 30 * DEG
    q_host[pip_dof]  = 45 * DEG
    # Bend R Thumb too.
    thumb_cmc_dof = dof_names.index("R_Thumb_CMC_Joint")
    thumb_mcp_dof = dof_names.index("R_Thumb_MCP_Joint")
    q_host[thumb_cmc_dof] = 20 * DEG
    q_host[thumb_mcp_dof] = 30 * DEG
    model.joint_q.assign(q_host)

    # Run update().
    mirror.update()
    wp.synchronize()
    assert sync_calls[0] == 1, f"expected 1 sync call, got {sync_calls[0]}"

    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    eff = model.joint_effort_limit.numpy()

    # Verify L_Index_PIP_Joint K_j (q=[0, 30°, 45°], nominal motors).
    nominal = _ssu.get_nominal_motor_gains()
    Kp_motor = np.array([
        nominal["L_Index_ABAD_Joint"]["kp"],
        nominal["L_Index_MCP_Joint"]["kp"],
        nominal["L_Index_PIP_Joint"]["kp"],
    ])
    out_ref = _mt.transform_group("finger", np.array([0.0, 30*DEG, 45*DEG]), k_pos=Kp_motor)
    ref_pip_K_j = float(out_ref["k_pos"][2])
    actual_pip_K_j = float(ke[pip_dof])
    rtol, atol = 1e-3, 1e-6
    assert abs(actual_pip_K_j - ref_pip_K_j) < rtol*max(1, abs(ref_pip_K_j)) + atol, \
        f"L_Index_PIP K_j mismatch: actual={actual_pip_K_j} ref={ref_pip_K_j}"
    print(f"[OK] L_Index_PIP K_j (q=[0,30°,45°]) = {actual_pip_K_j:.5f} (ref={ref_pip_K_j:.5f})")

    # Verify L Shoulder_Pitch (scalar, q-invariant).
    sp_dof = dof_names.index("L_Shoulder_Pitch_Joint")
    expected_sp = _mt.SHOULDER_RATIO**2 * nominal["L_Shoulder_Pitch_Joint"]["kp"]
    assert abs(ke[sp_dof] - expected_sp) < rtol*expected_sp, \
        f"L_Shoulder_Pitch K_j mismatch: {ke[sp_dof]} vs {expected_sp}"
    print(f"[OK] L_Shoulder_Pitch K_j = {ke[sp_dof]:.4f} (ref ratio²·K_m = {expected_sp:.4f})")

    # Verify L_Elbow + L_Wrist_Yaw (cross-coupling).
    elb_dof = dof_names.index("L_Elbow_Joint")
    wyaw_dof = dof_names.index("L_Wrist_Yaw_Joint")
    Kp_elb_motor = np.array([nominal["L_Elbow_Joint"]["kp"], nominal["L_Wrist_Yaw_Joint"]["kp"]])
    out_elb = _mt.transform_group("elbow", np.zeros(2), k_pos=Kp_elb_motor)
    assert abs(ke[elb_dof] - out_elb["k_pos"][0]) < rtol*out_elb["k_pos"][0], \
        f"L_Elbow K_j mismatch: {ke[elb_dof]} vs {out_elb['k_pos'][0]}"
    assert abs(ke[wyaw_dof] - out_elb["k_pos"][1]) < rtol*out_elb["k_pos"][1], \
        f"L_Wrist_Yaw K_j mismatch: {ke[wyaw_dof]} vs {out_elb['k_pos'][1]}"
    print(f"[OK] L_Elbow K_j={ke[elb_dof]:.4f}, L_Wrist_Yaw K_j={ke[wyaw_dof]:.4f}")

    # Verify R_Thumb (q-dependent).
    r_yaw = dof_names.index("R_Thumb_Yaw_Joint")
    r_cmc = dof_names.index("R_Thumb_CMC_Joint")
    r_mcp = dof_names.index("R_Thumb_MCP_Joint")
    Kp_thumb_motor = np.array([
        nominal["R_Thumb_Yaw_Joint"]["kp"],
        nominal["R_Thumb_CMC_Joint"]["kp"],
        nominal["R_Thumb_MCP_Joint"]["kp"],
    ])
    q_thumb = np.array([0.0, 20*DEG, 30*DEG])  # q1 ignored by J
    out_thumb = _mt.transform_group("thumb", q_thumb, k_pos=Kp_thumb_motor)
    rtol_t = 5e-3
    for jdof, ref, label in [(r_yaw, out_thumb["k_pos"][0], "Yaw"),
                              (r_cmc, out_thumb["k_pos"][1], "CMC"),
                              (r_mcp, out_thumb["k_pos"][2], "MCP")]:
        assert abs(ke[jdof] - ref) < rtol_t*max(1, abs(ref)), \
            f"R_Thumb_{label} K_j mismatch: {ke[jdof]} vs {ref}"
    print(f"[OK] R_Thumb K_j (q=[0,20°,30°]): "
          f"Yaw={ke[r_yaw]:.4f}, CMC={ke[r_cmc]:.4f}, MCP={ke[r_mcp]:.4f}")


def main() -> int:
    print("=== jk_kernel unit tests ===\n")
    wp.init()

    if wp.get_cuda_device_count() < 1:
        print("[SKIP] no CUDA device available")
        return 0

    test_scalar_shoulder_waist_neck()
    test_elbow_constant_J()
    test_wrist_q_dependent()
    test_finger_q_dependent()
    test_thumb_q_dependent()
    test_motor_state_mirror_end_to_end()
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
