"""Custom warp kernels for motor→joint PD gain transform (real-time GPU mirror).

매 physics step (1ms) 마다 warp kernel 로 K_m_motor → K_j_joint 변환하여
Newton MuJoCo 의 ``model.joint_target_ke / kd / effort_limit`` view 에 직접 write.
다항식 계수와 변환 수식의 ground truth 는 ``motor_joint_transform.py``.

그룹 분류 (수학적 구조):
  1. Decoupled scalar — shoulder/waist/neck. J = diag(ratio).
     `apply_scalar_groups` (q 무관, K_m 변경 시에만 launch).
  2. Coupled constant-J — elbow + wrist_yaw. 차동풀리 상수 2x2 행렬.
     `compute_elbow_K_j` (q 무관, K_m 변경 시에만 launch).
  3. Coupled q-dependent — wrist_roll/pitch, finger, thumb. J(q) 다항식.
     `compute_wrist_K_j` / `compute_finger_K_j` / `compute_thumb_K_j`
     (매 step launch — q 가 매 step 변하므로).

수학:
    K_j_diag[i] = Σ_k J[k,i]² · K_m[k]
    D_j_diag[i] = Σ_k J[k,i]² · Kv_m[k]
    τ_j_max[i] = Σ_k |J[k,i]| · τ_m_max[k]   (보수적 절대값 합산)

Per-element clamp 패턴: caller 가 ``out_ke[dof_idx[base+i]] = ...`` 로
articulation DOF 인덱스에 직접 write. zero-copy view + index write 라
CUDA graph 호환.
"""
from __future__ import annotations

import warp as wp


# ============================================================
# Constants — Elbow differential pulley (q 무관, 좌우 동일)
# 출처: motor_joint_transform.py:29-31
# ============================================================
ELBOW_N1: float = 18.0 / 85.0
ELBOW_N2: float = (20.0 + 1.8) / (80.0 + 1.8)
ELBOW_N3: float = (2.38 + 80.0) / (2.38 + 46.0)
ELBOW_A: float = 1.0 / (ELBOW_N1 * ELBOW_N2)
ELBOW_B: float = 1.0 / (ELBOW_N1 * ELBOW_N2 * ELBOW_N3)


# ============================================================
# [1] apply_scalar_groups — shoulder / waist / neck
#
# Decoupled scalar mapping: m_i = ratio_i · q_i
#   K_j[i] = ratio_i² · K_m[i]
#   D_j[i] = ratio_i² · Kv_m[i]
#   τ_j[i] = ratio_i  · τ_m_max[i]
#
# 단일 kernel 로 6 shoulder + 2 waist + 2 neck = 10 motors 일괄 처리.
# caller 가 motor_idx 순서대로 ratio[] 배열을 미리 빌드해서 넘김.
# ============================================================
@wp.kernel
def apply_scalar_groups(
    # size = N_scalar (shoulder/waist/neck motors 합산)
    dof_idx: wp.array(dtype=int),       # articulation DOF index per motor
    ratio: wp.array(dtype=float),       # per-motor reduction ratio
    K_m: wp.array(dtype=float),         # motor stiffness
    Kv_m: wp.array(dtype=float),        # motor damping
    trq_m: wp.array(dtype=float),       # motor torque limit
    out_ke: wp.array(dtype=float),      # writes to model.joint_target_ke (zero-copy)
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()
    r = ratio[tid]
    r2 = r * r
    idx = dof_idx[tid]
    out_ke[idx] = r2 * K_m[tid]
    out_kd[idx] = r2 * Kv_m[tid]
    out_eff[idx] = r * trq_m[tid]


# ============================================================
# [2] compute_elbow_K_j — Elbow + Wrist_Yaw (차동풀리)
#
# Coupled constant-J: 두 모터 ↔ 두 관절 cross-coupling. J 는 q 무관 상수.
#
#   J = [[ a, b],
#        [-a, b]]
#
# K_j_diag[i] = Σ_k J[k,i]² · K_m[k]:
#   K_j[Elbow_Joint]      = a²·(K_m[motor0] + K_m[motor1])
#   K_j[Wrist_Yaw_Joint]  = b²·(K_m[motor0] + K_m[motor1])
# 한 모터 K_m 변경 시 두 관절 K_j 모두 영향 — 교차결합 명시.
#
# τ_j_max = |J|^T · τ_m_max:
#   τ_j[Elbow_Joint]      = a·(τ_m[motor0] + τ_m[motor1])
#   τ_j[Wrist_Yaw_Joint]  = b·(τ_m[motor0] + τ_m[motor1])
#
# Launch: dim=2 (양손).
#   instance 0 = L_Elbow + L_Wrist_Yaw, instance 1 = R_Elbow + R_Wrist_Yaw
#   dof_idx[base+0] = elbow joint dof, dof_idx[base+1] = wrist_yaw joint dof
#   K_m[base+0..1] = motor 0..1 of that hand's elbow group
# ============================================================
@wp.kernel
def compute_elbow_K_j(
    # size = 2 hands × 2 motors = 4
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()           # 0..1 (2 hands)
    base = tid * 2

    a = wp.float32(ELBOW_A)
    b = wp.float32(ELBOW_B)
    a2 = a * a
    b2 = b * b

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    sum_km = km0 + km1
    K_j_elb   = a2 * sum_km
    K_j_wyaw  = b2 * sum_km

    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    sum_kvm = kvm0 + kvm1
    D_j_elb   = a2 * sum_kvm
    D_j_wyaw  = b2 * sum_kvm

    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]
    sum_tm = tm0 + tm1
    T_j_elb   = a * sum_tm   # a > 0
    T_j_wyaw  = b * sum_tm   # b > 0

    elb_idx  = dof_idx[base + 0]
    wyaw_idx = dof_idx[base + 1]

    out_ke[elb_idx]   = K_j_elb
    out_ke[wyaw_idx]  = K_j_wyaw
    out_kd[elb_idx]   = D_j_elb
    out_kd[wyaw_idx]  = D_j_wyaw
    out_eff[elb_idx]  = T_j_elb
    out_eff[wyaw_idx] = T_j_wyaw


# ============================================================
# [3] compute_wrist_K_j — Wrist_Roll + Wrist_Pitch (볼스크류, 비선형)
#
# Coupled q-dependent: J(qr, qp) 가 10차 다항식. 매 step 재계산.
# 좌우 동일 다항식 계수 (motor_joint_transform.py:60-123 의 식 그대로).
#
#   row 0 = right ballscrew motor, row 1 = left ballscrew motor
#   col 0 = wrist_roll, col 1 = wrist_pitch
#
# K_j_diag[i] = J[0,i]²·K_m[0] + J[1,i]²·K_m[1]
# τ_j_max[i]  = |J[0,i]|·τ_m[0] + |J[1,i]|·τ_m[1]
#
# Launch: dim=2 (양손).
#   per instance: 2 motors, 2 joint positions, 2 K_j outputs
#   dof_idx[base+0..1] = Wrist_Roll/Pitch articulation DOF index
#   K_m[base+0..1]     = right/left ballscrew motor stiffness
# ============================================================
@wp.func
def _wrist_jacobian(qr: wp.float32, qp: wp.float32) -> wp.mat22:
    """볼스크류 자코비안 (motor_joint_transform.py wrist_jacobian 1:1)."""
    qp2 = qp * qp
    qp3 = qp2 * qp
    qp4 = qp3 * qp
    qp5 = qp4 * qp
    qp6 = qp5 * qp
    qp7 = qp6 * qp
    qp8 = qp7 * qp
    qp9 = qp8 * qp
    qr2 = qr * qr
    qr3 = qr2 * qr
    qr4 = qr3 * qr
    qr5 = qr4 * qr
    qr6 = qr5 * qr
    qr7 = qr6 * qr
    qr8 = qr7 * qr
    qr9 = qr8 * qr

    j00 = (
        wp.float32(-3.7354676001e+00) * qr9 + wp.float32(-8.3501722583e+00) * qr8 * qp + wp.float32(-1.7390634802e+00) * qr8
        + wp.float32(-1.0665641890e+01) * qr7 * qp2 + wp.float32(-4.5424388008e+00) * qr7 * qp + wp.float32(5.5373061340e+00) * qr7
        + wp.float32(-6.5498578999e+00) * qr6 * qp3 + wp.float32(-2.7320425149e+00) * qr6 * qp2 + wp.float32(9.7003201536e+00) * qr6 * qp + wp.float32(2.1484494033e+00) * qr6
        + wp.float32(-1.0233019925e+00) * qr5 * qp4 + wp.float32(4.2330644296e-02) * qr5 * qp3 + wp.float32(6.9093543779e+00) * qr5 * qp2 + wp.float32(7.1574195924e+00) * qr5 * qp + wp.float32(-3.8813657587e+00) * qr5
        + wp.float32(2.6704783080e-01) * qr4 * qp5 + wp.float32(-1.0494854274e+00) * qr4 * qp4 + wp.float32(2.3422755464e+00) * qr4 * qp3 + wp.float32(7.6100656758e+00) * qr4 * qp2 + wp.float32(-4.3526577013e+00) * qr4 * qp + wp.float32(-4.0902032486e+00) * qr4
        + wp.float32(2.1643547508e-01) * qr3 * qp6 + wp.float32(1.3141976365e-01) * qr3 * qp5 + wp.float32(1.4812996506e+00) * qr3 * qp4 + wp.float32(3.7305298229e+00) * qr3 * qp3 + wp.float32(-3.2758078842e+00) * qr3 * qp2 + wp.float32(-5.8157906227e+00) * qr3 * qp + wp.float32(-2.2586652008e+00) * qr3
        + wp.float32(9.4371114321e-02) * qr2 * qp7 + wp.float32(1.7417167534e-01) * qr2 * qp6 + wp.float32(-8.4955920006e-02) * qr2 * qp5 + wp.float32(1.5448458463e-01) * qr2 * qp4 + wp.float32(1.4449105188e+00) * qr2 * qp3 + wp.float32(-1.4484506406e+00) * qr2 * qp2 + wp.float32(-1.5411090792e+01) * qr2 * qp + wp.float32(5.2661293879e+00) * qr2
        + wp.float32(-1.6854855413e-02) * qr * qp8 + wp.float32(-7.2506342270e-02) * qr * qp7 + wp.float32(1.5313192544e-03) * qr * qp6 + wp.float32(8.1660407302e-01) * qr * qp5 + wp.float32(1.7118923925e+00) * qr * qp4 + wp.float32(-2.3861339391e+00) * qr * qp3 + wp.float32(-1.0285533606e+01) * qr * qp2 + wp.float32(-1.7054911470e+00) * qr * qp + wp.float32(1.8507127234e+01) * qr
        + wp.float32(9.3277119958e-04) * qp9 + wp.float32(5.0831047124e-03) * qp8 + wp.float32(1.6716295795e-03) * qp7 + wp.float32(-2.7187103564e-02) * qp6 + wp.float32(-1.6158213922e-02) * qp5 + wp.float32(9.2901404362e-02) * qp4 + wp.float32(2.1566552480e-01) * qp3 + wp.float32(-3.0857597302e-01) * qp2 + wp.float32(-1.9322304644e+00) * qp + wp.float32(3.1969447084e+01)
    )

    j01 = (
        wp.float32(-9.2779691759e-01) * qr9 + wp.float32(-2.6664104726e+00) * qr8 * qp + wp.float32(-5.6780485010e-01) * qr8
        + wp.float32(-2.8070819571e+00) * qr7 * qp2 + wp.float32(-7.8058357567e-01) * qr7 * qp + wp.float32(1.3857600219e+00) * qr7
        + wp.float32(-6.8220132836e-01) * qr6 * qp3 + wp.float32(2.1165322148e-02) * qr6 * qp2 + wp.float32(2.3031181260e+00) * qr6 * qp + wp.float32(1.1929032654e+00) * qr6
        + wp.float32(2.6704783080e-01) * qr5 * qp4 + wp.float32(-8.3958834193e-01) * qr5 * qp3 + wp.float32(1.4053653278e+00) * qr5 * qp2 + wp.float32(3.0440262703e+00) * qr5 * qp + wp.float32(-8.7053154025e-01) * qr5
        + wp.float32(3.2465321262e-01) * qr4 * qp5 + wp.float32(1.6427470456e-01) * qr4 * qp4 + wp.float32(1.4812996506e+00) * qr4 * qp3 + wp.float32(2.7978973672e+00) * qr4 * qp2 + wp.float32(-1.6379039421e+00) * qr4 * qp + wp.float32(-1.4539476557e+00) * qr4
        + wp.float32(2.2019926675e-01) * qr3 * qp6 + wp.float32(3.4834335068e-01) * qr3 * qp5 + wp.float32(-1.4159320001e-01) * qr3 * qp4 + wp.float32(2.0597944617e-01) * qr3 * qp3 + wp.float32(1.4449105188e+00) * qr3 * qp2 + wp.float32(-9.6563376043e-01) * qr3 * qp + wp.float32(-5.1370302639e+00) * qr3
        + wp.float32(-6.7419421650e-02) * qr2 * qp7 + wp.float32(-2.5377219794e-01) * qr2 * qp6 + wp.float32(4.5939577631e-03) * qr2 * qp5 + wp.float32(2.0415101825e+00) * qr2 * qp4 + wp.float32(3.4237847850e+00) * qr2 * qp3 + wp.float32(-3.5792009086e+00) * qr2 * qp2 + wp.float32(-1.0285533606e+01) * qr2 * qp + wp.float32(-8.5274557352e-01) * qr2
        + wp.float32(8.3949407963e-03) * qr * qp8 + wp.float32(4.0664837699e-02) * qr * qp7 + wp.float32(1.1701407056e-02) * qr * qp6 + wp.float32(-1.6312262139e-01) * qr * qp5 + wp.float32(-8.0791069611e-02) * qr * qp4 + wp.float32(3.7160561745e-01) * qr * qp3 + wp.float32(6.4699657441e-01) * qr * qp2 + wp.float32(-6.1715194604e-01) * qr * qp + wp.float32(-1.9322304644e+00) * qr
        + wp.float32(-2.9297953530e-03) * qp9 + wp.float32(-3.5303544263e-02) * qp8 + wp.float32(-2.1021604979e-02) * qp7 + wp.float32(2.8714417481e-01) * qp6 + wp.float32(7.3346559585e-01) * qp5 + wp.float32(-9.4488847584e-01) * qp4 + wp.float32(-5.0494287812e+00) * qp3 + wp.float32(-8.2334873779e+00) * qp2 + wp.float32(7.3590613687e+00) * qp + wp.float32(2.8516757797e+01)
    )

    j10 = (
        wp.float32(-3.7354676000e+00) * qr9 + wp.float32(8.3501722589e+00) * qr8 * qp + wp.float32(1.7390634805e+00) * qr8
        + wp.float32(-1.0665641890e+01) * qr7 * qp2 + wp.float32(-4.5424388006e+00) * qr7 * qp + wp.float32(5.5373061339e+00) * qr7
        + wp.float32(6.5498579006e+00) * qr6 * qp3 + wp.float32(2.7320425161e+00) * qr6 * qp2 + wp.float32(-9.7003201538e+00) * qr6 * qp + wp.float32(-2.1484494037e+00) * qr6
        + wp.float32(-1.0233019925e+00) * qr5 * qp4 + wp.float32(4.2330644443e-02) * qr5 * qp3 + wp.float32(6.9093543778e+00) * qr5 * qp2 + wp.float32(7.1574195922e+00) * qr5 * qp + wp.float32(-3.8813657587e+00) * qr5
        + wp.float32(-2.6704783033e-01) * qr4 * qp5 + wp.float32(1.0494854286e+00) * qr4 * qp4 + wp.float32(-2.3422755462e+00) * qr4 * qp3 + wp.float32(-7.6100656767e+00) * qr4 * qp2 + wp.float32(4.3526577011e+00) * qr4 * qp + wp.float32(4.0902032487e+00) * qr4
        + wp.float32(2.1643547508e-01) * qr3 * qp6 + wp.float32(1.3141976364e-01) * qr3 * qp5 + wp.float32(1.4812996505e+00) * qr3 * qp4 + wp.float32(3.7305298228e+00) * qr3 * qp3 + wp.float32(-3.2758078841e+00) * qr3 * qp2 + wp.float32(-5.8157906226e+00) * qr3 * qp + wp.float32(-2.2586652008e+00) * qr3
        + wp.float32(-9.4371114236e-02) * qr2 * qp7 + wp.float32(-1.7417167507e-01) * qr2 * qp6 + wp.float32(8.4955920089e-02) * qr2 * qp5 + wp.float32(-1.5448458501e-01) * qr2 * qp4 + wp.float32(-1.4449105190e+00) * qr2 * qp3 + wp.float32(1.4484506408e+00) * qr2 * qp2 + wp.float32(1.5411090792e+01) * qr2 * qp + wp.float32(-5.2661293879e+00) * qr2
        + wp.float32(-1.6854855406e-02) * qr * qp8 + wp.float32(-7.2506342258e-02) * qr * qp7 + wp.float32(1.5313192440e-03) * qr * qp6 + wp.float32(8.1660407299e-01) * qr * qp5 + wp.float32(1.7118923925e+00) * qr * qp4 + wp.float32(-2.3861339390e+00) * qr * qp3 + wp.float32(-1.0285533606e+01) * qr * qp2 + wp.float32(-1.7054911470e+00) * qr * qp + wp.float32(1.8507127234e+01) * qr
        + wp.float32(-9.3277119715e-04) * qp9 + wp.float32(-5.0831047059e-03) * qp8 + wp.float32(-1.6716295815e-03) * qp7 + wp.float32(2.7187103546e-02) * qp6 + wp.float32(1.6158213915e-02) * qp5 + wp.float32(-9.2901404347e-02) * qp4 + wp.float32(-2.1566552479e-01) * qp3 + wp.float32(3.0857597302e-01) * qp2 + wp.float32(1.9322304644e+00) * qp + wp.float32(-3.1969447084e+01)
    )

    j11 = (
        wp.float32(9.2779691765e-01) * qr9 + wp.float32(-2.6664104725e+00) * qr8 * qp + wp.float32(-5.6780485007e-01) * qr8
        + wp.float32(2.8070819574e+00) * qr7 * qp2 + wp.float32(7.8058357604e-01) * qr7 * qp + wp.float32(-1.3857600220e+00) * qr7
        + wp.float32(-6.8220132832e-01) * qr6 * qp3 + wp.float32(2.1165322222e-02) * qr6 * qp2 + wp.float32(2.3031181259e+00) * qr6 * qp + wp.float32(1.1929032654e+00) * qr6
        + wp.float32(-2.6704783033e-01) * qr5 * qp4 + wp.float32(8.3958834286e-01) * qr5 * qp3 + wp.float32(-1.4053653277e+00) * qr5 * qp2 + wp.float32(-3.0440262707e+00) * qr5 * qp + wp.float32(8.7053154022e-01) * qr5
        + wp.float32(3.2465321262e-01) * qr4 * qp5 + wp.float32(1.6427470455e-01) * qr4 * qp4 + wp.float32(1.4812996505e+00) * qr4 * qp3 + wp.float32(2.7978973671e+00) * qr4 * qp2 + wp.float32(-1.6379039421e+00) * qr4 * qp + wp.float32(-1.4539476557e+00) * qr4
        + wp.float32(-2.2019926655e-01) * qr3 * qp6 + wp.float32(-3.4834335014e-01) * qr3 * qp5 + wp.float32(1.4159320015e-01) * qr3 * qp4 + wp.float32(-2.0597944667e-01) * qr3 * qp3 + wp.float32(-1.4449105190e+00) * qr3 * qp2 + wp.float32(9.6563376054e-01) * qr3 * qp + wp.float32(5.1370302639e+00) * qr3
        + wp.float32(-6.7419421626e-02) * qr2 * qp7 + wp.float32(-2.5377219790e-01) * qr2 * qp6 + wp.float32(4.5939577319e-03) * qr2 * qp5 + wp.float32(2.0415101825e+00) * qr2 * qp4 + wp.float32(3.4237847850e+00) * qr2 * qp3 + wp.float32(-3.5792009086e+00) * qr2 * qp2 + wp.float32(-1.0285533606e+01) * qr2 * qp + wp.float32(-8.5274557352e-01) * qr2
        + wp.float32(-8.3949407743e-03) * qr * qp8 + wp.float32(-4.0664837647e-02) * qr * qp7 + wp.float32(-1.1701407070e-02) * qr * qp6 + wp.float32(1.6312262128e-01) * qr * qp5 + wp.float32(8.0791069575e-02) * qr * qp4 + wp.float32(-3.7160561739e-01) * qr * qp3 + wp.float32(-6.4699657438e-01) * qr * qp2 + wp.float32(6.1715194603e-01) * qr * qp + wp.float32(1.9322304644e+00) * qr
        + wp.float32(-2.9297953546e-03) * qp9 + wp.float32(-3.5303544260e-02) * qp8 + wp.float32(-2.1021604973e-02) * qp7 + wp.float32(2.8714417481e-01) * qp6 + wp.float32(7.3346559584e-01) * qp5 + wp.float32(-9.4488847584e-01) * qp4 + wp.float32(-5.0494287812e+00) * qp3 + wp.float32(-8.2334873779e+00) * qp2 + wp.float32(7.3590613687e+00) * qp + wp.float32(2.8516757797e+01)
    )

    return wp.mat22(j00, j01, j10, j11)


@wp.kernel
def compute_wrist_K_j(
    # size = 2 hands × 2 motors = 4
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    # joint positions for q-dependent J(q)
    joint_q: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()           # 0..1 (2 hands)
    base = tid * 2

    # CSV col 0 = Wrist_Roll, col 1 = Wrist_Pitch (motor_joint_transform.py:407)
    qr_idx = dof_idx[base + 0]
    qp_idx = dof_idx[base + 1]
    qr = joint_q[qr_idx]
    qp = joint_q[qp_idx]

    J = _wrist_jacobian(qr, qp)
    j00 = J[0, 0]
    j01 = J[0, 1]
    j10 = J[1, 0]
    j11 = J[1, 1]

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    K_j_roll  = j00 * j00 * km0 + j10 * j10 * km1
    K_j_pitch = j01 * j01 * km0 + j11 * j11 * km1

    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    D_j_roll  = j00 * j00 * kvm0 + j10 * j10 * kvm1
    D_j_pitch = j01 * j01 * kvm0 + j11 * j11 * kvm1

    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]
    T_j_roll  = wp.abs(j00) * tm0 + wp.abs(j10) * tm1
    T_j_pitch = wp.abs(j01) * tm0 + wp.abs(j11) * tm1

    out_ke[qr_idx]  = K_j_roll
    out_ke[qp_idx]  = K_j_pitch
    out_kd[qr_idx]  = D_j_roll
    out_kd[qp_idx]  = D_j_pitch
    out_eff[qr_idx] = T_j_roll
    out_eff[qp_idx] = T_j_pitch


# ============================================================
# [4] compute_finger_K_j — 4-finger (index/middle/ring/little, 좌우 동일)
#
# Coupled q-dependent: J(q1, q2, q3) 다항식. 6 nonzero cells.
#   J[0,0] = f(q1)         (proximal motor depends only on q1)
#   J[1,0], J[1,1] = f(q1, q2)   (middle motor)
#   J[2,0], J[2,1], J[2,2] = f(q1, q2, q3)  (distal motor)
#   J[0,1] = J[0,2] = J[1,2] = 0
#
# K_j_diag (3 DOF, ABAD/MCP/PIP):
#   K_j[ABAD] = J[0,0]²·K_m[0] + J[1,0]²·K_m[1] + J[2,0]²·K_m[2]
#   K_j[MCP]  =                  J[1,1]²·K_m[1] + J[2,1]²·K_m[2]
#   K_j[PIP]  =                                   J[2,2]²·K_m[2]
#
# Launch: dim = N_finger_instances (양손 4 finger × 2 = 8 per full robot).
#   per instance: base = tid * 3
#   dof_idx[base+0..2] = ABAD/MCP/PIP joint dof
#   K_m[base+0..2]     = proximal/middle/distal motor stiffness
# ============================================================
@wp.func
def _finger_jacobian_cells(
    q1: wp.float32, q2: wp.float32, q3: wp.float32
) -> wp.vec(length=6, dtype=wp.float32):
    """6 nonzero cells in row-major (J00, J10, J11, J20, J21, J22)."""
    q1_2 = q1 * q1
    q2_2 = q2 * q2
    q3_2 = q3 * q3

    j00 = wp.float32(-12.34580882002154) * q1_2 + wp.float32(3.26639663102094) * q1 + wp.float32(54.3942504070892)

    j10 = (
        wp.float32(-0.4940422858107675) * q1_2
        + wp.float32(-41.63623628755128) * q1 * q2
        + wp.float32(32.69037801607999) * q1
        + wp.float32(-1.115501119537795) * q2_2
        + wp.float32(1.185508947234422) * q2
        + wp.float32(-2.506887503234288)
    )

    j11 = (
        wp.float32(-20.81811814377564) * q1_2
        + wp.float32(-2.231002239075589) * q1 * q2
        + wp.float32(1.185508947234422) * q1
        + wp.float32(-29.99827749164071) * q2_2
        + wp.float32(25.59106651177869) * q2
        + wp.float32(57.69145986124609)
    )

    j20 = (
        wp.float32(6.96805224663322) * q1_2
        + wp.float32(29.38888411889691) * q1 * q2
        + wp.float32(5.814305056019611) * q1 * q3
        + wp.float32(38.09302675413097) * q1
        + wp.float32(-0.3309972764553427) * q2_2
        + wp.float32(0.8684268765944178) * q2 * q3
        + wp.float32(-3.300663858112859) * q2
        + wp.float32(2.650070377887364) * q3_2
        + wp.float32(-6.059430988043804) * q3
        + wp.float32(-1.919430452480224)
    )

    j21 = (
        wp.float32(14.69444205944846) * q1_2
        + wp.float32(-0.6619945529106853) * q1 * q2
        + wp.float32(0.8684268765944178) * q1 * q3
        + wp.float32(-3.300663858112859) * q1
        + wp.float32(-5.107765190777431) * q2_2
        + wp.float32(-12.49631288448714) * q2 * q3
        + wp.float32(69.07092246166906) * q2
        + wp.float32(-6.336541937802106) * q3_2
        + wp.float32(30.08633445339482) * q3
        + wp.float32(30.80868964850648)
    )

    j22 = (
        wp.float32(2.907152528009806) * q1_2
        + wp.float32(0.8684268765944178) * q1 * q2
        + wp.float32(5.300140755774729) * q1 * q3
        + wp.float32(-6.059430988043804) * q1
        + wp.float32(-6.24815644224357) * q2_2
        + wp.float32(-12.67308387560421) * q2 * q3
        + wp.float32(30.08633445339482) * q2
        + wp.float32(-14.37221028018762) * q3_2
        + wp.float32(56.01176116478506) * q3
        + wp.float32(11.48681046815231)
    )

    out = wp.vec(length=6, dtype=wp.float32)
    out[0] = j00
    out[1] = j10
    out[2] = j11
    out[3] = j20
    out[4] = j21
    out[5] = j22
    return out


@wp.kernel
def compute_finger_K_j(
    # size = N_finger_instances × 3 (per finger: ABAD/MCP/PIP motors)
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()           # finger instance index (0..N-1)
    base = tid * 3

    abad_idx = dof_idx[base + 0]
    mcp_idx  = dof_idx[base + 1]
    pip_idx  = dof_idx[base + 2]

    q1 = joint_q[abad_idx]
    q2 = joint_q[mcp_idx]
    q3 = joint_q[pip_idx]

    cells = _finger_jacobian_cells(q1, q2, q3)
    j00 = cells[0]
    j10 = cells[1]
    j11 = cells[2]
    j20 = cells[3]
    j21 = cells[4]
    j22 = cells[5]

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    km2 = K_m[base + 2]
    K_j_abad = j00 * j00 * km0 + j10 * j10 * km1 + j20 * j20 * km2
    K_j_mcp  =                   j11 * j11 * km1 + j21 * j21 * km2
    K_j_pip  =                                     j22 * j22 * km2

    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    kvm2 = Kv_m[base + 2]
    D_j_abad = j00 * j00 * kvm0 + j10 * j10 * kvm1 + j20 * j20 * kvm2
    D_j_mcp  =                    j11 * j11 * kvm1 + j21 * j21 * kvm2
    D_j_pip  =                                       j22 * j22 * kvm2

    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]
    tm2 = trq_m[base + 2]
    T_j_abad = wp.abs(j00) * tm0 + wp.abs(j10) * tm1 + wp.abs(j20) * tm2
    T_j_mcp  =                     wp.abs(j11) * tm1 + wp.abs(j21) * tm2
    T_j_pip  =                                         wp.abs(j22) * tm2

    out_ke[abad_idx]  = K_j_abad
    out_ke[mcp_idx]   = K_j_mcp
    out_ke[pip_idx]   = K_j_pip
    out_kd[abad_idx]  = D_j_abad
    out_kd[mcp_idx]   = D_j_mcp
    out_kd[pip_idx]   = D_j_pip
    out_eff[abad_idx] = T_j_abad
    out_eff[mcp_idx]  = T_j_mcp
    out_eff[pip_idx]  = T_j_pip


# ============================================================
# [5] compute_thumb_K_j — Thumb (좌우 동일)
#
# Coupled q-dependent: J(q2, q3) — q1 (Yaw) 은 자코비안에 영향 없음
# (a0 모터는 선형이라 J[0,0] 상수). J[2,1] / J[2,2] 가 15차 다항식.
#
# 희소 J:
#   J[0,0] = constant, J[0,1] = J[0,2] = 0
#   J[1,1] = poly(q2)  (5 terms)
#   J[1,0] = J[1,2] = 0
#   J[2,1] = poly(q2, q3) — 136 terms
#   J[2,2] = poly(q2, q3) — 136 terms
#   J[2,0] = 0
#
# K_j_diag (3 DOF: Yaw/CMC/MCP):
#   K_j[Yaw] = J[0,0]² · K_m[0]
#   K_j[CMC] = J[1,1]² · K_m[1] + J[2,1]² · K_m[2]
#   K_j[MCP] =                    J[2,2]² · K_m[2]
#
# Launch: dim=2 (양손).
# ============================================================
@wp.func
def _thumb_jacobian_cells(
    q2: wp.float32, q3: wp.float32
) -> wp.vec(length=4, dtype=wp.float32):
    """4 nonzero cells: J[0,0] constant, J[1,1] = f(q2), J[2,1] / J[2,2] = f(q2,q3).
    Returns (J00, J11, J21, J22).

    J[0,0] 은 thumb_jacobian.py:244 의 1.2703682581E+02 (q1 무관 상수).
    """
    q2_2 = q2 * q2
    q2_3 = q2_2 * q2
    q2_4 = q2_3 * q2
    q2_5 = q2_4 * q2
    q2_6 = q2_5 * q2
    q2_7 = q2_6 * q2
    q2_8 = q2_7 * q2
    q2_9 = q2_8 * q2
    q2_10 = q2_9 * q2
    q2_11 = q2_10 * q2
    q2_12 = q2_11 * q2
    q2_13 = q2_12 * q2
    q2_14 = q2_13 * q2
    q2_15 = q2_14 * q2

    q3_2 = q3 * q3
    q3_3 = q3_2 * q3
    q3_4 = q3_3 * q3
    q3_5 = q3_4 * q3
    q3_6 = q3_5 * q3
    q3_7 = q3_6 * q3
    q3_8 = q3_7 * q3
    q3_9 = q3_8 * q3
    q3_10 = q3_9 * q3
    q3_11 = q3_10 * q3
    q3_12 = q3_11 * q3
    q3_13 = q3_12 * q3
    q3_14 = q3_13 * q3
    q3_15 = q3_14 * q3

    j00 = wp.float32(1.2703682581e+02)

    j11 = (
        wp.float32(-9.4037869428e-02) * q2_4 + wp.float32(-3.6527982855e+01) * q2_3
        + wp.float32(6.9065216603e+01) * q2_2 + wp.float32(-3.1978298063e+01) * q2
        + wp.float32(9.4037910629e+01)
    )

    j21 = (
        wp.float32(3.9156424913e-02) * q2_15 + wp.float32(-2.1262239788e-01) * q2_14 * q3 + wp.float32(-3.6622387365e-01) * q2_14 + wp.float32(-4.6143339171e-01) * q2_13 * q3_2 + wp.float32(2.6751648067e+00) * q2_13 * q3 + wp.float32(1.3649565267e+00) * q2_13
        + wp.float32(-2.5258874634e+00) * q2_12 * q3_3 + wp.float32(9.0045900729e+00) * q2_12 * q3_2 + wp.float32(-1.6969984003e+01) * q2_12 * q3 + wp.float32(-1.9623164648e+00) * q2_12
        + wp.float32(-1.3907038592e+00) * q2_11 * q3_4 + wp.float32(3.2092515814e+01) * q2_11 * q3_3 + wp.float32(-7.2024472374e+01) * q2_11 * q3_2 + wp.float32(7.2234443649e+01) * q2_11 * q3 + wp.float32(-3.2534079268e+00) * q2_11
        + wp.float32(2.4838257756e+01) * q2_10 * q3_5 + wp.float32(-4.1544566952e+01) * q2_10 * q3_4 + wp.float32(-1.3472407471e+02) * q2_10 * q3_3 + wp.float32(3.0799664329e+02) * q2_10 * q3_2 + wp.float32(-2.2248319225e+02) * q2_10 * q3 + wp.float32(2.3535341515e+01) * q2_10
        + wp.float32(8.8834572641e+01) * q2_9 * q3_6 + wp.float32(-4.8053872550e+02) * q2_9 * q3_5 + wp.float32(7.2916332996e+02) * q2_9 * q3_4 + wp.float32(1.8896950947e+01) * q2_9 * q3_3 + wp.float32(-7.4034630173e+02) * q2_9 * q3_2 + wp.float32(4.9535104625e+02) * q2_9 * q3 + wp.float32(-6.3388921400e+01) * q2_9
        + wp.float32(1.4248722872e+02) * q2_8 * q3_7 + wp.float32(-1.2016382408e+03) * q2_8 * q3_6 + wp.float32(3.6402079104e+03) * q2_8 * q3_5 + wp.float32(-4.6919884772e+03) * q2_8 * q3_4 + wp.float32(1.9150514975e+03) * q2_8 * q3_3 + wp.float32(8.0897288924e+02) * q2_8 * q3_2 + wp.float32(-7.6667276875e+02) * q2_8 * q3 + wp.float32(1.1055684270e+02) * q2_8
        + wp.float32(9.2645500720e+01) * q2_7 * q3_8 + wp.float32(-1.2974242726e+03) * q2_7 * q3_7 + wp.float32(6.2693874219e+03) * q2_7 * q3_6 + wp.float32(-1.4277648245e+04) * q2_7 * q3_5 + wp.float32(1.6256307826e+04) * q2_7 * q3_4 + wp.float32(-8.2377559187e+03) * q2_7 * q3_3 + wp.float32(6.2383636158e+02) * q2_7 * q3_2 + wp.float32(7.3853869816e+02) * q2_7 * q3 + wp.float32(-1.3574197172e+02) * q2_7
        + wp.float32(-5.7990655576e+01) * q2_6 * q3_9 + wp.float32(-6.9701927035e+01) * q2_6 * q3_8 + wp.float32(3.2828339561e+03) * q2_6 * q3_7 + wp.float32(-1.4963187214e+04) * q2_6 * q3_6 + wp.float32(3.0842513517e+04) * q2_6 * q3_5 + wp.float32(-3.3129003135e+04) * q2_6 * q3_4 + wp.float32(1.7906008634e+04) * q2_6 * q3_3 + wp.float32(-3.7116701980e+03) * q2_6 * q3_2 + wp.float32(-2.3313428878e+02) * q2_6 * q3 + wp.float32(1.1352557978e+02) * q2_6
        + wp.float32(-1.8005120299e+02) * q2_5 * q3_10 + wp.float32(1.5457024447e+03) * q2_5 * q3_9 + wp.float32(-4.7754408584e+03) * q2_5 * q3_8 + wp.float32(4.2858946490e+03) * q2_5 * q3_7 + wp.float32(1.0170154555e+04) * q2_5 * q3_6 + wp.float32(-3.2611000657e+04) * q2_5 * q3_5 + wp.float32(3.8586425371e+04) * q2_5 * q3_4 + wp.float32(-2.2720526047e+04) * q2_5 * q3_3 + wp.float32(6.2317724756e+03) * q2_5 * q3_2 + wp.float32(-4.7528054110e+02) * q2_5 * q3 + wp.float32(-4.8365274538e+01) * q2_5
        + wp.float32(-1.8503302244e+02) * q2_4 * q3_11 + wp.float32(2.0908233762e+03) * q2_4 * q3_10 + wp.float32(-1.0033890844e+04) * q2_4 * q3_9 + wp.float32(2.6196853654e+04) * q2_4 * q3_8 + wp.float32(-3.8879957514e+04) * q2_4 * q3_7 + wp.float32(2.8699555679e+04) * q2_4 * q3_6 + wp.float32(3.0737085608e+02) * q2_4 * q3_5 + wp.float32(-1.9139680618e+04) * q2_4 * q3_4 + wp.float32(1.5758267760e+04) * q2_4 * q3_3 + wp.float32(-5.6486969305e+03) * q2_4 * q3_2 + wp.float32(8.5716515409e+02) * q2_4 * q3 + wp.float32(-2.5695825213e+01) * q2_4
        + wp.float32(-1.1332242859e+02) * q2_3 * q3_12 + wp.float32(1.4797637329e+03) * q2_3 * q3_11 + wp.float32(-8.5436615041e+03) * q2_3 * q3_10 + wp.float32(2.8561662281e+04) * q2_3 * q3_9 + wp.float32(-6.0717534868e+04) * q2_3 * q3_8 + wp.float32(8.4599832226e+04) * q2_3 * q3_7 + wp.float32(-7.6178339742e+04) * q2_3 * q3_6 + wp.float32(4.0814532666e+04) * q2_3 * q3_5 + wp.float32(-8.9003044989e+03) * q2_3 * q3_4 + wp.float32(-2.9401061716e+03) * q2_3 * q3_3 + wp.float32(2.5723196688e+03) * q2_3 * q3_2 + wp.float32(-7.0658263266e+02) * q2_3 * q3 + wp.float32(7.3526527684e+01) * q2_3
        + wp.float32(-4.4962390411e+01) * q2_2 * q3_13 + wp.float32(6.4348201364e+02) * q2_2 * q3_12 + wp.float32(-4.1586538526e+03) * q2_2 * q3_11 + wp.float32(1.5997314896e+04) * q2_2 * q3_10 + wp.float32(-4.0632111172e+04) * q2_2 * q3_9 + wp.float32(7.1426702231e+04) * q2_2 * q3_8 + wp.float32(-8.8530490944e+04) * q2_2 * q3_7 + wp.float32(7.7274772969e+04) * q2_2 * q3_6 + wp.float32(-4.6508700874e+04) * q2_2 * q3_5 + wp.float32(1.8295977651e+04) * q2_2 * q3_4 + wp.float32(-4.0124177896e+03) * q2_2 * q3_3 + wp.float32(3.5090650679e+01) * q2_2 * q3_2 + wp.float32(2.8957976580e+02) * q2_2 * q3 + wp.float32(-7.6526644352e+01) * q2_2
        + wp.float32(-1.1401635926e+01) * q2 * q3_14 + wp.float32(1.7186132770e+02) * q2 * q3_13 + wp.float32(-1.1869202904e+03) * q2 * q3_12 + wp.float32(4.9666153614e+03) * q2 * q3_11 + wp.float32(-1.4030475870e+04) * q2 * q3_10 + wp.float32(2.8222450700e+04) * q2 * q3_9 + wp.float32(-4.1556031658e+04) * q2 * q3_8 + wp.float32(4.5372012801e+04) * q2 * q3_7 + wp.float32(-3.6854139509e+04) * q2 * q3_6 + wp.float32(2.2207139539e+04) * q2 * q3_5 + wp.float32(-9.8660043317e+03) * q2 * q3_4 + wp.float32(3.1865618626e+03) * q2 * q3_3 + wp.float32(-7.0786518422e+02) * q2 * q3_2 + wp.float32(1.7291647672e+01) * q2 * q3 + wp.float32(6.2220314855e+01) * q2
        + wp.float32(-1.5894756922e+00) * q3_15 + wp.float32(2.4242829723e+01) * q3_14 + wp.float32(-1.7107512972e+02) * q3_13 + wp.float32(7.4044364633e+02) * q3_12 + wp.float32(-2.1967256231e+03) * q3_11 + wp.float32(4.7293940616e+03) * q3_10 + wp.float32(-7.6328434511e+03) * q3_9 + wp.float32(9.4144246842e+03) * q3_8 + wp.float32(-8.9815883313e+03) * q3_7 + wp.float32(6.6918254738e+03) * q3_6 + wp.float32(-3.9461035365e+03) * q3_5 + wp.float32(1.8900283851e+03) * q3_4 + wp.float32(-7.7793640168e+02) * q3_3 + wp.float32(2.9439366007e+02) * q3_2 + wp.float32(-7.8465086452e+01) * q3 + wp.float32(2.7656561601e+00)
    )

    j22 = (
        wp.float32(-1.4174826525e-02) * q2_15 + wp.float32(-6.5919055959e-02) * q2_14 * q3 + wp.float32(1.9108320048e-01) * q2_14 + wp.float32(-5.8289710695e-01) * q2_13 * q3_2 + wp.float32(1.3853215497e+00) * q2_13 * q3 + wp.float32(-1.3053833848e+00) * q2_13
        + wp.float32(-4.6356795307e-01) * q2_12 * q3_3 + wp.float32(8.0231289534e+00) * q2_12 * q3_2 + wp.float32(-1.2004078729e+01) * q2_12 * q3 + wp.float32(6.0195369707e+00) * q2_12
        + wp.float32(1.1290117162e+01) * q2_11 * q3_4 + wp.float32(-1.5107115255e+01) * q2_11 * q3_3 + wp.float32(-3.6742929466e+01) * q2_11 * q3_2 + wp.float32(5.5999389690e+01) * q2_11 * q3 + wp.float32(-2.0225744750e+01) * q2_11
        + wp.float32(5.3300743585e+01) * q2_10 * q3_5 + wp.float32(-2.4026936275e+02) * q2_10 * q3_4 + wp.float32(2.9166533198e+02) * q2_10 * q3_3 + wp.float32(5.6690852841e+00) * q2_10 * q3_2 + wp.float32(-1.4806926035e+02) * q2_10 * q3 + wp.float32(4.9535104625e+01) * q2_10
        + wp.float32(1.1082340011e+02) * q2_9 * q3_6 + wp.float32(-8.0109216052e+02) * q2_9 * q3_5 + wp.float32(2.0223377280e+03) * q2_9 * q3_4 + wp.float32(-2.0853282121e+03) * q2_9 * q3_3 + wp.float32(6.3835049916e+02) * q2_9 * q3_2 + wp.float32(1.7977175316e+02) * q2_9 * q3 + wp.float32(-8.5185863194e+01) * q2_9
        + wp.float32(9.2645500720e+01) * q2_8 * q3_7 + wp.float32(-1.1352462386e+03) * q2_8 * q3_6 + wp.float32(4.7020405665e+03) * q2_8 * q3_5 + wp.float32(-8.9235301533e+03) * q2_8 * q3_4 + wp.float32(8.1281539132e+03) * q2_8 * q3_3 + wp.float32(-3.0891584695e+03) * q2_8 * q3_2 + wp.float32(1.5595909040e+02) * q2_8 * q3 + wp.float32(9.2317337270e+01) * q2_8
        + wp.float32(-7.4559414312e+01) * q2_7 * q3_8 + wp.float32(-7.9659345183e+01) * q2_7 * q3_7 + wp.float32(3.2828339561e+03) * q2_7 * q3_6 + wp.float32(-1.2825589040e+04) * q2_7 * q3_5 + wp.float32(2.2030366798e+04) * q2_7 * q3_4 + wp.float32(-1.8930858934e+04) * q2_7 * q3_3 + wp.float32(7.6740037004e+03) * q2_7 * q3_2 + wp.float32(-1.0604771994e+03) * q2_7 * q3 + wp.float32(-3.3304898397e+01) * q2_7
        + wp.float32(-3.0008533832e+02) * q2_6 * q3_9 + wp.float32(2.3185536670e+03) * q2_6 * q3_8 + wp.float32(-6.3672544779e+03) * q2_6 * q3_7 + wp.float32(5.0002104238e+03) * q2_6 * q3_6 + wp.float32(1.0170154555e+04) * q2_6 * q3_5 + wp.float32(-2.7175833881e+04) * q2_6 * q3_4 + wp.float32(2.5724283581e+04) * q2_6 * q3_3 + wp.float32(-1.1360263023e+04) * q2_6 * q3_2 + wp.float32(2.0772574919e+03) * q2_6 * q3 + wp.float32(-7.9213423517e+01) * q2_6
        + wp.float32(-4.0707264938e+02) * q2_5 * q3_10 + wp.float32(4.1816467525e+03) * q2_5 * q3_9 + wp.float32(-1.8061003519e+04) * q2_5 * q3_8 + wp.float32(4.1914965846e+04) * q2_5 * q3_7 + wp.float32(-5.4431940520e+04) * q2_5 * q3_6 + wp.float32(3.4439466814e+04) * q2_5 * q3_5 + wp.float32(3.0737085608e+02) * q2_5 * q3_4 + wp.float32(-1.5311744494e+04) * q2_5 * q3_3 + wp.float32(9.4549606561e+03) * q2_5 * q3_2 + wp.float32(-2.2594787722e+03) * q2_5 * q3 + wp.float32(1.7143303082e+02) * q2_5
        + wp.float32(-3.3996728578e+02) * q2_4 * q3_11 + wp.float32(4.0693502654e+03) * q2_4 * q3_10 + wp.float32(-2.1359153760e+04) * q2_4 * q3_9 + wp.float32(6.4263740132e+04) * q2_4 * q3_8 + wp.float32(-1.2143506974e+05) * q2_4 * q3_7 + wp.float32(1.4804970640e+05) * q2_4 * q3_6 + wp.float32(-1.1426750961e+05) * q2_4 * q3_5 + wp.float32(5.1018165832e+04) * q2_4 * q3_4 + wp.float32(-8.9003044989e+03) * q2_4 * q3_3 + wp.float32(-2.2050796287e+03) * q2_4 * q3_2 + wp.float32(1.2861598344e+03) * q2_4 * q3 + wp.float32(-1.7664565816e+02) * q2_4
        + wp.float32(-1.9483702511e+02) * q2_3 * q3_12 + wp.float32(2.5739280545e+03) * q2_3 * q3_11 + wp.float32(-1.5248397460e+04) * q2_3 * q3_10 + wp.float32(5.3324382985e+04) * q2_3 * q3_9 + wp.float32(-1.2189633352e+05) * q2_3 * q3_8 + wp.float32(1.9047120595e+05) * q2_3 * q3_7 + wp.float32(-2.0657114554e+05) * q2_3 * q3_6 + wp.float32(1.5454954594e+05) * q2_3 * q3_5 + wp.float32(-7.7514501457e+04) * q2_3 * q3_4 + wp.float32(2.4394636868e+04) * q2_3 * q3_3 + wp.float32(-4.0124177896e+03) * q2_3 * q3_2 + wp.float32(2.3393767119e+01) * q2_3 * q3 + wp.float32(9.6526588601e+01) * q2_3
        + wp.float32(-7.9811451480e+01) * q2_2 * q3_13 + wp.float32(1.1170986300e+03) * q2_2 * q3_12 + wp.float32(-7.1215217427e+03) * q2_2 * q3_11 + wp.float32(2.7316384488e+04) * q2_2 * q3_10 + wp.float32(-7.0152379349e+04) * q2_2 * q3_9 + wp.float32(1.2700102815e+05) * q2_2 * q3_8 + wp.float32(-1.6622412663e+05) * q2_2 * q3_7 + wp.float32(1.5880204481e+05) * q2_2 * q3_6 + wp.float32(-1.1056241853e+05) * q2_2 * q3_5 + wp.float32(5.5517848848e+04) * q2_2 * q3_4 + wp.float32(-1.9732008663e+04) * q2_2 * q3_3 + wp.float32(4.7798427939e+03) * q2_2 * q3_2 + wp.float32(-7.0786518422e+02) * q2_2 * q3 + wp.float32(8.6458238359e+00) * q2_2
        + wp.float32(-2.3842135382e+01) * q2 * q3_14 + wp.float32(3.3939961612e+02) * q2 * q3_13 + wp.float32(-2.2239766863e+03) * q2 * q3_12 + wp.float32(8.8853237560e+03) * q2 * q3_11 + wp.float32(-2.4163981854e+04) * q2 * q3_10 + wp.float32(4.7293940616e+04) * q2 * q3_9 + wp.float32(-6.8695591060e+04) * q2 * q3_8 + wp.float32(7.5315397473e+04) * q2 * q3_7 + wp.float32(-6.2871118319e+04) * q2 * q3_6 + wp.float32(4.0150952843e+04) * q2 * q3_5 + wp.float32(-1.9730517682e+04) * q2 * q3_4 + wp.float32(7.5601135404e+03) * q2 * q3_3 + wp.float32(-2.3338092050e+03) * q2 * q3_2 + wp.float32(5.8878732015e+02) * q2 * q3 + wp.float32(-7.8465086452e+01) * q2
        + wp.float32(-4.7187176495e+00) * q3_15 + wp.float32(6.6279960534e+01) * q3_14 + wp.float32(-4.2939820160e+02) * q3_13 + wp.float32(1.7028677979e+03) * q3_12 + wp.float32(-4.6267912567e+03) * q3_11 + wp.float32(9.1361880513e+03) * q3_10 + wp.float32(-1.3576137971e+04) * q3_9 + wp.float32(1.5520767915e+04) * q3_8 + wp.float32(-1.3859972937e+04) * q3_7 + wp.float32(9.7909541716e+03) * q3_6 + wp.float32(-5.5572720952e+03) * q3_5 + wp.float32(2.6027107055e+03) * q3_4 + wp.float32(-1.0662033003e+03) * q3_3 + wp.float32(4.1858739863e+02) * q3_2 + wp.float32(-1.6784783284e+02) * q3 + wp.float32(1.2090078755e+02)
    )

    out = wp.vec(length=4, dtype=wp.float32)
    out[0] = j00
    out[1] = j11
    out[2] = j21
    out[3] = j22
    return out


@wp.kernel
def compute_thumb_K_j(
    # size = N_thumb_instances × 3 (Yaw/CMC/MCP per thumb)
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()           # 0..N-1 (양손 thumb 2 instance)
    base = tid * 3

    yaw_idx = dof_idx[base + 0]
    cmc_idx = dof_idx[base + 1]
    mcp_idx = dof_idx[base + 2]

    # q1 (Yaw) 은 thumb J 에 영향 없음 (a0 motor 가 q1 에 선형 → J[0,0] 상수)
    q2 = joint_q[cmc_idx]
    q3 = joint_q[mcp_idx]

    cells = _thumb_jacobian_cells(q2, q3)
    j00 = cells[0]
    j11 = cells[1]
    j21 = cells[2]
    j22 = cells[3]

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    km2 = K_m[base + 2]
    K_j_yaw = j00 * j00 * km0
    K_j_cmc = j11 * j11 * km1 + j21 * j21 * km2
    K_j_mcp =                   j22 * j22 * km2

    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    kvm2 = Kv_m[base + 2]
    D_j_yaw = j00 * j00 * kvm0
    D_j_cmc = j11 * j11 * kvm1 + j21 * j21 * kvm2
    D_j_mcp =                    j22 * j22 * kvm2

    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]
    tm2 = trq_m[base + 2]
    T_j_yaw = wp.abs(j00) * tm0
    T_j_cmc = wp.abs(j11) * tm1 + wp.abs(j21) * tm2
    T_j_mcp =                     wp.abs(j22) * tm2

    out_ke[yaw_idx]  = K_j_yaw
    out_ke[cmc_idx]  = K_j_cmc
    out_ke[mcp_idx]  = K_j_mcp
    out_kd[yaw_idx]  = D_j_yaw
    out_kd[cmc_idx]  = D_j_cmc
    out_kd[mcp_idx]  = D_j_mcp
    out_eff[yaw_idx] = T_j_yaw
    out_eff[cmc_idx] = T_j_cmc
    out_eff[mcp_idx] = T_j_mcp
