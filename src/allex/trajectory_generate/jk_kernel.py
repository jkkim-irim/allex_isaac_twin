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
# Polynomial FK evaluators (joint angle → motor angle).
# Used by pd_clip_wrist / finger / thumb 의 Y(motor_space) 모드.
# Coefficient tables: fk_coeffs.py (parsed from allex_control C++).
# Layout: per part 'offsets[n_out+1]' slices a flat (exp..., coef) table.
# ============================================================
@wp.func
def eval_poly_2d(qr: float, qp: float,
                 exp_qr: wp.array(dtype=int),
                 exp_qp: wp.array(dtype=int),
                 coef: wp.array(dtype=float),
                 start: int, end: int) -> float:
    val = wp.float32(0.0)
    for i in range(start, end):
        t = coef[i]
        for _ in range(exp_qr[i]):
            t *= qr
        for _ in range(exp_qp[i]):
            t *= qp
        val += t
    return val


@wp.func
def eval_poly_3d(q1: float, q2: float, q3: float,
                 exp_q1: wp.array(dtype=int),
                 exp_q2: wp.array(dtype=int),
                 exp_q3: wp.array(dtype=int),
                 coef: wp.array(dtype=float),
                 start: int, end: int) -> float:
    val = wp.float32(0.0)
    for i in range(start, end):
        t = coef[i]
        for _ in range(exp_q1[i]):
            t *= q1
        for _ in range(exp_q2[i]):
            t *= q2
        for _ in range(exp_q3[i]):
            t *= q3
        val += t
    return val


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


# ============================================================================
# Motor-space torque clip kernels (feature/motor_torque_limit)
# ============================================================================
# 위쪽 compute_*_K_j 커널은 K_j_diag 를 model.joint_target_ke 등에 써서 MuJoCo
# 의 actuator PD 가 사용하게 했다. 아래 pd_clip_* 커널은 우리가 직접 PD 식을
# 풀어 motor 공간에서 saturation 시킨 뒤 model.qfrc_applied 에 쓰는 path
# 이며, 동시에 model.joint_target_ke / kd 를 0 으로 강제해 MuJoCo 측 PD 를
# 비활성화한다. effort_limit 에는 |J|^T·τ_m_max safety net 값을 그대로 둬
# Newton 의 내부 clip 도 이중 안전망으로 활용.
#
# 모드:
#   pd_mode = 0 (motor_space, Y) — q_m = J·q 변환 후 motor-domain PD.
#   pd_mode = 1 (joint_space_diag, X) — joint-domain diag K_j=Σ J[k,i]²·K_m[k]
#                                       로 PD 계산 후 J⁻ᵀ 로 motor 변환.
# scalar 그룹은 X≡Y 라 모드 분기 없음. coupled (elbow/wrist/finger/thumb)
# 에서만 분기 발생.
#
# 비-mech-comp motor 의 처리 흐름:
#   q_m, qt_m = J·q, J·qt   (Y 모드)
#   τ_m_pd    = K_m·(qt_m−q_m) + Kv_m·(qdt_m−qd_m)
#   g_m       = J⁻ᵀ · g_j
#   τ_m_des   = τ_m_pd + g_m
#   τ_m_clip  = clamp(τ_m_des, ±τ_m_max)
#   τ_j_clip  = Jᵀ · τ_m_clip
#   joint_f   = τ_j_clip − g_j         (MuJoCo passive 가 g_j 다시 더함)
#   safety    = |J|ᵀ · τ_m_max
#   joint_f   = clamp(joint_f, ±safety)
# mech-comp motor: gravcomp 합산 단계 skip (scalar 그룹 내부에서만).


# ----------------------------------------------------------------------------
# [6] pd_clip_scalar — shoulder/waist/neck (10 motors, 1×1 J)
#
# scalar 라 X≡Y. 모드 인자 불필요.
# is_mech_comp[k] = 1 → gravcomp 합산 skip.
# ----------------------------------------------------------------------------
@wp.kernel
def pd_clip_scalar(
    dof_idx: wp.array(dtype=int),
    ratio: wp.array(dtype=float),
    is_mech_comp: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    qt: wp.array(dtype=float),
    qdt: wp.array(dtype=float),
    gravcomp_j: wp.array(dtype=float),
    out_joint_f: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()
    i = dof_idx[tid]
    r = ratio[tid]
    km = K_m[tid]
    kvm = Kv_m[tid]
    tm = trq_m[tid]

    # Motor-space PD (1×1 J = diag(r)): τ_m = r·K_m·(qt−q) + r·Kv_m·(qdt−qd)
    err = r * (qt[i] - q[i])
    derr = r * (qdt[i] - qd[i])
    tau_m_pd = km * err + kvm * derr

    # Gravcomp (skip for mech-comp scalar motors)
    g_j = wp.where(is_mech_comp[tid] == 1, wp.float32(0.0), gravcomp_j[i])
    g_m = g_j / r

    # Motor saturation
    tau_m_total = tau_m_pd + g_m
    tau_m_clip = wp.clamp(tau_m_total, -tm, tm)

    # Safety net 은 motor 출력 (tau_j) 자체에 적용 — 정상 시 |tau_j| ≤ r·tm 이미
    # 보장되므로 no-op. NaN/pinv blowup backstop 용. gravcomp 차감 후 clamp 하면
    # 모터 saturate + 중력 반대 push 구간에서 gravcomp 분만큼 잘려 토크 부족 발생.
    tau_j = r * tau_m_clip
    safety = r * tm
    tau_j = wp.clamp(tau_j, -safety, safety)
    joint_f_val = tau_j - g_j

    out_joint_f[i] = joint_f_val
    out_ke[i] = wp.float32(0.0)
    out_kd[i] = wp.float32(0.0)
    out_eff[i] = safety


# ----------------------------------------------------------------------------
# [7] pd_clip_elbow — Elbow + Wrist_Yaw (2×2 constant J, X/Y 모드)
#
# J = [[a,  b], [-a, b]]  (a = ELBOW_A, b = ELBOW_B)
# det(J) = 2ab > 0 (절대 singular 아님; det 모니터링 생략 가능하지만 instance 별
# 값은 일관성 위해 기록).
#
# 모든 elbow 모터는 non-mech-comp → gravcomp 항상 포함.
# Launch: dim = 2 (L/R), per instance 2 motors / 2 joints.
# ----------------------------------------------------------------------------
@wp.kernel
def pd_clip_elbow(
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    qt: wp.array(dtype=float),
    qdt: wp.array(dtype=float),
    gravcomp_j: wp.array(dtype=float),
    pd_mode: int,                       # 0 = Y (motor-space), 1 = X (joint-diag)
    out_joint_f: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
    out_det: wp.array(dtype=float),     # |det(J)| per instance
):
    tid = wp.tid()
    base = tid * 2
    elb_idx  = dof_idx[base + 0]      # joint 0 (col 0 in J)
    wyaw_idx = dof_idx[base + 1]      # joint 1 (col 1 in J)

    a = wp.float32(ELBOW_A)
    b = wp.float32(ELBOW_B)
    # J = [[a, b], [-a, b]] (mat22 row-major in warp)
    J = wp.mat22(a, b, -a, b)

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]

    # Joint-space q's
    q_elb = q[elb_idx]
    q_wyaw = q[wyaw_idx]
    qd_elb = qd[elb_idx]
    qd_wyaw = qd[wyaw_idx]
    qt_elb = qt[elb_idx]
    qt_wyaw = qt[wyaw_idx]
    qdt_elb = qdt[elb_idx]
    qdt_wyaw = qdt[wyaw_idx]
    g_elb = gravcomp_j[elb_idx]
    g_wyaw = gravcomp_j[wyaw_idx]

    # Compute motor-space PD
    tau_m_pd_0 = wp.float32(0.0)
    tau_m_pd_1 = wp.float32(0.0)
    if pd_mode == 0:
        # Y: q_m = J·q etc., then per-motor PD
        q_m_0 = J[0, 0] * q_elb + J[0, 1] * q_wyaw
        q_m_1 = J[1, 0] * q_elb + J[1, 1] * q_wyaw
        qd_m_0 = J[0, 0] * qd_elb + J[0, 1] * qd_wyaw
        qd_m_1 = J[1, 0] * qd_elb + J[1, 1] * qd_wyaw
        qt_m_0 = J[0, 0] * qt_elb + J[0, 1] * qt_wyaw
        qt_m_1 = J[1, 0] * qt_elb + J[1, 1] * qt_wyaw
        qdt_m_0 = J[0, 0] * qdt_elb + J[0, 1] * qdt_wyaw
        qdt_m_1 = J[1, 0] * qdt_elb + J[1, 1] * qdt_wyaw
        tau_m_pd_0 = km0 * (qt_m_0 - q_m_0) + kvm0 * (qdt_m_0 - qd_m_0)
        tau_m_pd_1 = km1 * (qt_m_1 - q_m_1) + kvm1 * (qdt_m_1 - qd_m_1)
    else:
        # X: joint-diag K_j, then J^-T · τ_j
        K_j_elb  = J[0, 0] * J[0, 0] * km0 + J[1, 0] * J[1, 0] * km1
        K_j_wyaw = J[0, 1] * J[0, 1] * km0 + J[1, 1] * J[1, 1] * km1
        D_j_elb  = J[0, 0] * J[0, 0] * kvm0 + J[1, 0] * J[1, 0] * kvm1
        D_j_wyaw = J[0, 1] * J[0, 1] * kvm0 + J[1, 1] * J[1, 1] * kvm1
        tau_j_pd_elb  = K_j_elb  * (qt_elb  - q_elb)  + D_j_elb  * (qdt_elb  - qd_elb)
        tau_j_pd_wyaw = K_j_wyaw * (qt_wyaw - q_wyaw) + D_j_wyaw * (qdt_wyaw - qd_wyaw)
        # J^-T = inverse(J^T)
        J_inv_T = wp.inverse(wp.transpose(J))
        tau_m_pd_0 = J_inv_T[0, 0] * tau_j_pd_elb + J_inv_T[0, 1] * tau_j_pd_wyaw
        tau_m_pd_1 = J_inv_T[1, 0] * tau_j_pd_elb + J_inv_T[1, 1] * tau_j_pd_wyaw

    # Gravcomp: J^-T · g_j
    J_inv_T = wp.inverse(wp.transpose(J))
    g_m_0 = J_inv_T[0, 0] * g_elb + J_inv_T[0, 1] * g_wyaw
    g_m_1 = J_inv_T[1, 0] * g_elb + J_inv_T[1, 1] * g_wyaw

    # Motor saturation
    tau_m_clip_0 = wp.clamp(tau_m_pd_0 + g_m_0, -tm0, tm0)
    tau_m_clip_1 = wp.clamp(tau_m_pd_1 + g_m_1, -tm1, tm1)

    # Back to joint: J^T · τ_m
    tau_j_elb  = J[0, 0] * tau_m_clip_0 + J[1, 0] * tau_m_clip_1
    tau_j_wyaw = J[0, 1] * tau_m_clip_0 + J[1, 1] * tau_m_clip_1

    # Safety net 은 motor 출력 tau_j 자체에 적용 (gravcomp 차감 전).
    safety_elb  = wp.abs(J[0, 0]) * tm0 + wp.abs(J[1, 0]) * tm1
    safety_wyaw = wp.abs(J[0, 1]) * tm0 + wp.abs(J[1, 1]) * tm1
    tau_j_elb   = wp.clamp(tau_j_elb,   -safety_elb,  safety_elb)
    tau_j_wyaw  = wp.clamp(tau_j_wyaw,  -safety_wyaw, safety_wyaw)

    # Subtract gravcomp (MuJoCo passive re-adds it)
    joint_f_elb  = tau_j_elb  - g_elb
    joint_f_wyaw = tau_j_wyaw - g_wyaw

    # Outputs
    out_joint_f[elb_idx]  = joint_f_elb
    out_joint_f[wyaw_idx] = joint_f_wyaw
    out_ke[elb_idx]  = wp.float32(0.0)
    out_ke[wyaw_idx] = wp.float32(0.0)
    out_kd[elb_idx]  = wp.float32(0.0)
    out_kd[wyaw_idx] = wp.float32(0.0)
    out_eff[elb_idx]  = safety_elb
    out_eff[wyaw_idx] = safety_wyaw
    out_det[tid] = wp.abs(wp.determinant(J))


# ----------------------------------------------------------------------------
# [8] pd_clip_wrist — Wrist_Roll + Wrist_Pitch (2×2 q-dependent J)
# Launch: dim = 2 (L/R), uses _wrist_jacobian(qr, qp).
# ----------------------------------------------------------------------------
@wp.kernel
def pd_clip_wrist(
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    qt: wp.array(dtype=float),
    qdt: wp.array(dtype=float),
    gravcomp_j: wp.array(dtype=float),
    pd_mode: int,
    # FK polynomial (allex_control calcWristJoint2MotorAngle, qr/qp → motor 0/1)
    poly_offsets: wp.array(dtype=int),   # size 3: [start_a0, start_a1, end_a1]
    poly_exp_qr: wp.array(dtype=int),
    poly_exp_qp: wp.array(dtype=int),
    poly_coef:   wp.array(dtype=float),
    out_joint_f: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
    out_det: wp.array(dtype=float),
):
    tid = wp.tid()
    base = tid * 2
    qr_idx = dof_idx[base + 0]       # Wrist_Roll = col 0
    qp_idx = dof_idx[base + 1]       # Wrist_Pitch = col 1

    qr = q[qr_idx]
    qp = q[qp_idx]
    J = _wrist_jacobian(qr, qp)

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]

    qd_r = qd[qr_idx]
    qd_p = qd[qp_idx]
    qt_r = qt[qr_idx]
    qt_p = qt[qp_idx]
    qdt_r = qdt[qr_idx]
    qdt_p = qdt[qp_idx]
    g_r = gravcomp_j[qr_idx]
    g_p = gravcomp_j[qp_idx]

    tau_m_pd_0 = wp.float32(0.0)
    tau_m_pd_1 = wp.float32(0.0)
    if pd_mode == 0:
        # Y: motor-space PD — position via polynomial FK, velocity via J(q).
        a0 = poly_offsets[0]
        a1 = poly_offsets[1]
        a2 = poly_offsets[2]
        q_m_0  = eval_poly_2d(qr,    qp,    poly_exp_qr, poly_exp_qp, poly_coef, a0, a1)
        q_m_1  = eval_poly_2d(qr,    qp,    poly_exp_qr, poly_exp_qp, poly_coef, a1, a2)
        qt_m_0 = eval_poly_2d(qt_r,  qt_p,  poly_exp_qr, poly_exp_qp, poly_coef, a0, a1)
        qt_m_1 = eval_poly_2d(qt_r,  qt_p,  poly_exp_qr, poly_exp_qp, poly_coef, a1, a2)
        qd_m_0 = J[0, 0] * qd_r + J[0, 1] * qd_p
        qd_m_1 = J[1, 0] * qd_r + J[1, 1] * qd_p
        qdt_m_0 = J[0, 0] * qdt_r + J[0, 1] * qdt_p
        qdt_m_1 = J[1, 0] * qdt_r + J[1, 1] * qdt_p
        tau_m_pd_0 = km0 * (qt_m_0 - q_m_0) + kvm0 * (qdt_m_0 - qd_m_0)
        tau_m_pd_1 = km1 * (qt_m_1 - q_m_1) + kvm1 * (qdt_m_1 - qd_m_1)
    else:
        K_j_r = J[0, 0] * J[0, 0] * km0 + J[1, 0] * J[1, 0] * km1
        K_j_p = J[0, 1] * J[0, 1] * km0 + J[1, 1] * J[1, 1] * km1
        D_j_r = J[0, 0] * J[0, 0] * kvm0 + J[1, 0] * J[1, 0] * kvm1
        D_j_p = J[0, 1] * J[0, 1] * kvm0 + J[1, 1] * J[1, 1] * kvm1
        tau_j_pd_r = K_j_r * (qt_r - qr) + D_j_r * (qdt_r - qd_r)
        tau_j_pd_p = K_j_p * (qt_p - qp) + D_j_p * (qdt_p - qd_p)
        J_inv_T = wp.inverse(wp.transpose(J))
        tau_m_pd_0 = J_inv_T[0, 0] * tau_j_pd_r + J_inv_T[0, 1] * tau_j_pd_p
        tau_m_pd_1 = J_inv_T[1, 0] * tau_j_pd_r + J_inv_T[1, 1] * tau_j_pd_p

    # Gravcomp via J^-T
    J_inv_T = wp.inverse(wp.transpose(J))
    g_m_0 = J_inv_T[0, 0] * g_r + J_inv_T[0, 1] * g_p
    g_m_1 = J_inv_T[1, 0] * g_r + J_inv_T[1, 1] * g_p

    tau_m_clip_0 = wp.clamp(tau_m_pd_0 + g_m_0, -tm0, tm0)
    tau_m_clip_1 = wp.clamp(tau_m_pd_1 + g_m_1, -tm1, tm1)

    tau_j_r = J[0, 0] * tau_m_clip_0 + J[1, 0] * tau_m_clip_1
    tau_j_p = J[0, 1] * tau_m_clip_0 + J[1, 1] * tau_m_clip_1

    # Safety net 은 motor 출력 tau_j 자체에 적용 (gravcomp 차감 전).
    safety_r = wp.abs(J[0, 0]) * tm0 + wp.abs(J[1, 0]) * tm1
    safety_p = wp.abs(J[0, 1]) * tm0 + wp.abs(J[1, 1]) * tm1
    tau_j_r  = wp.clamp(tau_j_r, -safety_r, safety_r)
    tau_j_p  = wp.clamp(tau_j_p, -safety_p, safety_p)

    joint_f_r = tau_j_r - g_r
    joint_f_p = tau_j_p - g_p

    out_joint_f[qr_idx] = joint_f_r
    out_joint_f[qp_idx] = joint_f_p
    out_ke[qr_idx] = wp.float32(0.0)
    out_ke[qp_idx] = wp.float32(0.0)
    out_kd[qr_idx] = wp.float32(0.0)
    out_kd[qp_idx] = wp.float32(0.0)
    out_eff[qr_idx] = safety_r
    out_eff[qp_idx] = safety_p
    out_det[tid] = wp.abs(wp.determinant(J))


# ----------------------------------------------------------------------------
# [9] pd_clip_finger — 4-finger × 2 hands = 8 instances, 3×3 lower-triangular J
# Launch: dim = 8, per instance 3 motors / 3 joints (ABAD/MCP/PIP).
# J 의 nonzero cell 만 _finger_jacobian_cells 가 반환 (J00, J10, J11, J20, J21, J22).
# ----------------------------------------------------------------------------
@wp.kernel
def pd_clip_finger(
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    qt: wp.array(dtype=float),
    qdt: wp.array(dtype=float),
    gravcomp_j: wp.array(dtype=float),
    pd_mode: int,
    # FK polynomial (allex_control cal_motorAngles_janghwan, q1/q2/q3 → m0/m1/m2)
    poly_offsets: wp.array(dtype=int),   # size 4
    poly_exp_q1: wp.array(dtype=int),
    poly_exp_q2: wp.array(dtype=int),
    poly_exp_q3: wp.array(dtype=int),
    poly_coef:   wp.array(dtype=float),
    # Motor-side Coulomb friction (allex_control robot_model.json):
    #   τ_m_actual = τ_m_clip - friction_motor · tanh(qd_m · tanh_scale)
    friction_motor:    wp.array(dtype=float),
    friction_tanh_scl: wp.array(dtype=float),
    out_joint_f: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
    out_det: wp.array(dtype=float),
):
    tid = wp.tid()
    base = tid * 3
    abad_idx = dof_idx[base + 0]
    mcp_idx  = dof_idx[base + 1]
    pip_idx  = dof_idx[base + 2]

    q1 = q[abad_idx]
    q2 = q[mcp_idx]
    q3 = q[pip_idx]
    cells = _finger_jacobian_cells(q1, q2, q3)
    j00 = cells[0]
    j10 = cells[1]
    j11 = cells[2]
    j20 = cells[3]
    j21 = cells[4]
    j22 = cells[5]
    # Full 3×3 J (lower-triangular)
    J = wp.mat33(j00, wp.float32(0.0), wp.float32(0.0),
                 j10, j11,              wp.float32(0.0),
                 j20, j21,              j22)

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    km2 = K_m[base + 2]
    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    kvm2 = Kv_m[base + 2]
    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]
    tm2 = trq_m[base + 2]

    qd_abad = qd[abad_idx]
    qd_mcp  = qd[mcp_idx]
    qd_pip  = qd[pip_idx]
    qt_abad = qt[abad_idx]
    qt_mcp  = qt[mcp_idx]
    qt_pip  = qt[pip_idx]
    qdt_abad = qdt[abad_idx]
    qdt_mcp  = qdt[mcp_idx]
    qdt_pip  = qdt[pip_idx]
    g_abad = gravcomp_j[abad_idx]
    g_mcp  = gravcomp_j[mcp_idx]
    g_pip  = gravcomp_j[pip_idx]

    tau_m_pd_0 = wp.float32(0.0)
    tau_m_pd_1 = wp.float32(0.0)
    tau_m_pd_2 = wp.float32(0.0)
    if pd_mode == 0:
        # Y: position via polynomial FK, velocity via J(q).
        a0 = poly_offsets[0]
        a1 = poly_offsets[1]
        a2 = poly_offsets[2]
        a3 = poly_offsets[3]
        q_m_0  = eval_poly_3d(q1,       q2,      q3,      poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a0, a1)
        q_m_1  = eval_poly_3d(q1,       q2,      q3,      poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a1, a2)
        q_m_2  = eval_poly_3d(q1,       q2,      q3,      poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a2, a3)
        qt_m_0 = eval_poly_3d(qt_abad,  qt_mcp,  qt_pip,  poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a0, a1)
        qt_m_1 = eval_poly_3d(qt_abad,  qt_mcp,  qt_pip,  poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a1, a2)
        qt_m_2 = eval_poly_3d(qt_abad,  qt_mcp,  qt_pip,  poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a2, a3)
        qd_m_0 = j00 * qd_abad
        qd_m_1 = j10 * qd_abad + j11 * qd_mcp
        qd_m_2 = j20 * qd_abad + j21 * qd_mcp + j22 * qd_pip
        qdt_m_0 = j00 * qdt_abad
        qdt_m_1 = j10 * qdt_abad + j11 * qdt_mcp
        qdt_m_2 = j20 * qdt_abad + j21 * qdt_mcp + j22 * qdt_pip
        tau_m_pd_0 = km0 * (qt_m_0 - q_m_0) + kvm0 * (qdt_m_0 - qd_m_0)
        tau_m_pd_1 = km1 * (qt_m_1 - q_m_1) + kvm1 * (qdt_m_1 - qd_m_1)
        tau_m_pd_2 = km2 * (qt_m_2 - q_m_2) + kvm2 * (qdt_m_2 - qd_m_2)
    else:
        # X: diag K_j with cross-coupling ignored
        K_j_a = j00 * j00 * km0 + j10 * j10 * km1 + j20 * j20 * km2
        K_j_m = j11 * j11 * km1 + j21 * j21 * km2
        K_j_p = j22 * j22 * km2
        D_j_a = j00 * j00 * kvm0 + j10 * j10 * kvm1 + j20 * j20 * kvm2
        D_j_m = j11 * j11 * kvm1 + j21 * j21 * kvm2
        D_j_p = j22 * j22 * kvm2
        tau_j_pd_a = K_j_a * (qt_abad - q1) + D_j_a * (qdt_abad - qd_abad)
        tau_j_pd_m = K_j_m * (qt_mcp  - q2) + D_j_m * (qdt_mcp  - qd_mcp)
        tau_j_pd_p = K_j_p * (qt_pip  - q3) + D_j_p * (qdt_pip  - qd_pip)
        J_inv_T = wp.inverse(wp.transpose(J))
        tau_m_pd_0 = J_inv_T[0, 0] * tau_j_pd_a + J_inv_T[0, 1] * tau_j_pd_m + J_inv_T[0, 2] * tau_j_pd_p
        tau_m_pd_1 = J_inv_T[1, 0] * tau_j_pd_a + J_inv_T[1, 1] * tau_j_pd_m + J_inv_T[1, 2] * tau_j_pd_p
        tau_m_pd_2 = J_inv_T[2, 0] * tau_j_pd_a + J_inv_T[2, 1] * tau_j_pd_m + J_inv_T[2, 2] * tau_j_pd_p

    # Gravcomp via J^-T
    J_inv_T = wp.inverse(wp.transpose(J))
    g_m_0 = J_inv_T[0, 0] * g_abad + J_inv_T[0, 1] * g_mcp + J_inv_T[0, 2] * g_pip
    g_m_1 = J_inv_T[1, 0] * g_abad + J_inv_T[1, 1] * g_mcp + J_inv_T[1, 2] * g_pip
    g_m_2 = J_inv_T[2, 0] * g_abad + J_inv_T[2, 1] * g_mcp + J_inv_T[2, 2] * g_pip

    tau_m_clip_0 = wp.clamp(tau_m_pd_0 + g_m_0, -tm0, tm0)
    tau_m_clip_1 = wp.clamp(tau_m_pd_1 + g_m_1, -tm1, tm1)
    tau_m_clip_2 = wp.clamp(tau_m_pd_2 + g_m_2, -tm2, tm2)

    # Motor-side Coulomb friction (after motor saturation, before J^T):
    # τ_m_actual = τ_m_clip - friction · tanh(qd_m · tanh_scale)
    # Cap |friction| ≤ |τ_m_clip| so friction can stall the motor but cannot
    # reverse the motor command direction (avoids tanh model jitter when motor
    # can't overcome friction).
    fric0 = friction_motor[base + 0] * wp.tanh(qd_m_0 * friction_tanh_scl[base + 0])
    fric1 = friction_motor[base + 1] * wp.tanh(qd_m_1 * friction_tanh_scl[base + 1])
    fric2 = friction_motor[base + 2] * wp.tanh(qd_m_2 * friction_tanh_scl[base + 2])
    fric0 = wp.clamp(fric0, -wp.abs(tau_m_clip_0), wp.abs(tau_m_clip_0))
    fric1 = wp.clamp(fric1, -wp.abs(tau_m_clip_1), wp.abs(tau_m_clip_1))
    fric2 = wp.clamp(fric2, -wp.abs(tau_m_clip_2), wp.abs(tau_m_clip_2))
    tau_m_clip_0 = tau_m_clip_0 - fric0
    tau_m_clip_1 = tau_m_clip_1 - fric1
    tau_m_clip_2 = tau_m_clip_2 - fric2

    # τ_j = Jᵀ · τ_m
    tau_j_a = j00 * tau_m_clip_0 + j10 * tau_m_clip_1 + j20 * tau_m_clip_2
    tau_j_m = j11 * tau_m_clip_1 + j21 * tau_m_clip_2
    tau_j_p = j22 * tau_m_clip_2

    # Safety net 은 motor 출력 tau_j 자체에 적용 (gravcomp 차감 전).
    safety_a = wp.abs(j00) * tm0 + wp.abs(j10) * tm1 + wp.abs(j20) * tm2
    safety_m = wp.abs(j11) * tm1 + wp.abs(j21) * tm2
    safety_p = wp.abs(j22) * tm2
    tau_j_a  = wp.clamp(tau_j_a, -safety_a, safety_a)
    tau_j_m  = wp.clamp(tau_j_m, -safety_m, safety_m)
    tau_j_p  = wp.clamp(tau_j_p, -safety_p, safety_p)

    joint_f_a = tau_j_a - g_abad
    joint_f_m = tau_j_m - g_mcp
    joint_f_p = tau_j_p - g_pip

    out_joint_f[abad_idx] = joint_f_a
    out_joint_f[mcp_idx]  = joint_f_m
    out_joint_f[pip_idx]  = joint_f_p
    out_ke[abad_idx] = wp.float32(0.0)
    out_ke[mcp_idx]  = wp.float32(0.0)
    out_ke[pip_idx]  = wp.float32(0.0)
    out_kd[abad_idx] = wp.float32(0.0)
    out_kd[mcp_idx]  = wp.float32(0.0)
    out_kd[pip_idx]  = wp.float32(0.0)
    out_eff[abad_idx] = safety_a
    out_eff[mcp_idx]  = safety_m
    out_eff[pip_idx]  = safety_p
    # det(lower-triangular) = product of diagonals
    out_det[tid] = wp.abs(j00 * j11 * j22)


# ----------------------------------------------------------------------------
# [10] pd_clip_thumb — 2 hands × 3 motors, 3×3 sparse J
# Layout (motor_joint_transform.py thumb):
#   J[0,0] = constant, J[0,1] = J[0,2] = 0
#   J[1,0] = 0, J[1,1] = poly(q2), J[1,2] = 0
#   J[2,0] = 0, J[2,1] = poly(q2,q3), J[2,2] = poly(q2,q3)
# Joints in instance: Yaw=col0, CMC=col1, MCP=col2.
# ----------------------------------------------------------------------------
@wp.kernel
def pd_clip_thumb(
    dof_idx: wp.array(dtype=int),
    K_m: wp.array(dtype=float),
    Kv_m: wp.array(dtype=float),
    trq_m: wp.array(dtype=float),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    qt: wp.array(dtype=float),
    qdt: wp.array(dtype=float),
    gravcomp_j: wp.array(dtype=float),
    pd_mode: int,
    # FK polynomial (allex_control cal_motorAngles_thumb, q1/q2/q3 → m0/m1/m2)
    poly_offsets: wp.array(dtype=int),   # size 4
    poly_exp_q1: wp.array(dtype=int),
    poly_exp_q2: wp.array(dtype=int),
    poly_exp_q3: wp.array(dtype=int),
    poly_coef:   wp.array(dtype=float),
    # Motor-side Coulomb friction (allex_control robot_model.json)
    friction_motor:    wp.array(dtype=float),
    friction_tanh_scl: wp.array(dtype=float),
    out_joint_f: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
    out_det: wp.array(dtype=float),
):
    tid = wp.tid()
    base = tid * 3
    yaw_idx = dof_idx[base + 0]
    cmc_idx = dof_idx[base + 1]
    mcp_idx = dof_idx[base + 2]

    q2v = q[cmc_idx]
    q3v = q[mcp_idx]
    cells = _thumb_jacobian_cells(q2v, q3v)
    j00 = cells[0]   # constant
    j11 = cells[1]   # poly(q2)
    j21 = cells[2]   # poly(q2,q3)
    j22 = cells[3]   # poly(q2,q3)
    # Sparse J as full 3×3 with zeros where appropriate
    J = wp.mat33(j00, wp.float32(0.0), wp.float32(0.0),
                 wp.float32(0.0), j11, wp.float32(0.0),
                 wp.float32(0.0), j21, j22)

    km0 = K_m[base + 0]
    km1 = K_m[base + 1]
    km2 = K_m[base + 2]
    kvm0 = Kv_m[base + 0]
    kvm1 = Kv_m[base + 1]
    kvm2 = Kv_m[base + 2]
    tm0 = trq_m[base + 0]
    tm1 = trq_m[base + 1]
    tm2 = trq_m[base + 2]

    q1v = q[yaw_idx]
    qd_y = qd[yaw_idx]
    qd_c = qd[cmc_idx]
    qd_m = qd[mcp_idx]
    qt_y = qt[yaw_idx]
    qt_c = qt[cmc_idx]
    qt_m = qt[mcp_idx]
    qdt_y = qdt[yaw_idx]
    qdt_c = qdt[cmc_idx]
    qdt_m = qdt[mcp_idx]
    g_y = gravcomp_j[yaw_idx]
    g_c = gravcomp_j[cmc_idx]
    g_mp = gravcomp_j[mcp_idx]

    tau_m_pd_0 = wp.float32(0.0)
    tau_m_pd_1 = wp.float32(0.0)
    tau_m_pd_2 = wp.float32(0.0)
    if pd_mode == 0:
        # Y: position via polynomial FK, velocity via J(q).
        a0 = poly_offsets[0]
        a1 = poly_offsets[1]
        a2 = poly_offsets[2]
        a3 = poly_offsets[3]
        q_m_0  = eval_poly_3d(q1v,    q2v,    q3v,    poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a0, a1)
        q_m_1  = eval_poly_3d(q1v,    q2v,    q3v,    poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a1, a2)
        q_m_2  = eval_poly_3d(q1v,    q2v,    q3v,    poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a2, a3)
        qt_m_0 = eval_poly_3d(qt_y,   qt_c,   qt_m,   poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a0, a1)
        qt_m_1 = eval_poly_3d(qt_y,   qt_c,   qt_m,   poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a1, a2)
        qt_m_2 = eval_poly_3d(qt_y,   qt_c,   qt_m,   poly_exp_q1, poly_exp_q2, poly_exp_q3, poly_coef, a2, a3)
        qd_m_0 = j00 * qd_y
        qd_m_1 = j11 * qd_c
        qd_m_2 = j21 * qd_c + j22 * qd_m
        qdt_m_0 = j00 * qdt_y
        qdt_m_1 = j11 * qdt_c
        qdt_m_2 = j21 * qdt_c + j22 * qdt_m
        tau_m_pd_0 = km0 * (qt_m_0 - q_m_0) + kvm0 * (qdt_m_0 - qd_m_0)
        tau_m_pd_1 = km1 * (qt_m_1 - q_m_1) + kvm1 * (qdt_m_1 - qd_m_1)
        tau_m_pd_2 = km2 * (qt_m_2 - q_m_2) + kvm2 * (qdt_m_2 - qd_m_2)
    else:
        # X: diag K_j
        K_j_y = j00 * j00 * km0
        K_j_c = j11 * j11 * km1 + j21 * j21 * km2
        K_j_m = j22 * j22 * km2
        D_j_y = j00 * j00 * kvm0
        D_j_c = j11 * j11 * kvm1 + j21 * j21 * kvm2
        D_j_m = j22 * j22 * kvm2
        tau_j_pd_y = K_j_y * (qt_y - q1v) + D_j_y * (qdt_y - qd_y)
        tau_j_pd_c = K_j_c * (qt_c - q2v) + D_j_c * (qdt_c - qd_c)
        tau_j_pd_m = K_j_m * (qt_m - q3v) + D_j_m * (qdt_m - qd_m)
        J_inv_T = wp.inverse(wp.transpose(J))
        tau_m_pd_0 = J_inv_T[0, 0] * tau_j_pd_y + J_inv_T[0, 1] * tau_j_pd_c + J_inv_T[0, 2] * tau_j_pd_m
        tau_m_pd_1 = J_inv_T[1, 0] * tau_j_pd_y + J_inv_T[1, 1] * tau_j_pd_c + J_inv_T[1, 2] * tau_j_pd_m
        tau_m_pd_2 = J_inv_T[2, 0] * tau_j_pd_y + J_inv_T[2, 1] * tau_j_pd_c + J_inv_T[2, 2] * tau_j_pd_m

    J_inv_T = wp.inverse(wp.transpose(J))
    g_m_0 = J_inv_T[0, 0] * g_y + J_inv_T[0, 1] * g_c + J_inv_T[0, 2] * g_mp
    g_m_1 = J_inv_T[1, 0] * g_y + J_inv_T[1, 1] * g_c + J_inv_T[1, 2] * g_mp
    g_m_2 = J_inv_T[2, 0] * g_y + J_inv_T[2, 1] * g_c + J_inv_T[2, 2] * g_mp

    tau_m_clip_0 = wp.clamp(tau_m_pd_0 + g_m_0, -tm0, tm0)
    tau_m_clip_1 = wp.clamp(tau_m_pd_1 + g_m_1, -tm1, tm1)
    tau_m_clip_2 = wp.clamp(tau_m_pd_2 + g_m_2, -tm2, tm2)

    # Motor-side Coulomb friction subtraction (allex_control robot_model.json):
    # τ_m_actual = τ_m_clip - friction · tanh(qd_m · tanh_scale)
    # Cap |friction| ≤ |τ_m_clip| (no direction reversal — friction can stall
    # but not reverse motor).
    fric0 = friction_motor[base + 0] * wp.tanh(qd_m_0 * friction_tanh_scl[base + 0])
    fric1 = friction_motor[base + 1] * wp.tanh(qd_m_1 * friction_tanh_scl[base + 1])
    fric2 = friction_motor[base + 2] * wp.tanh(qd_m_2 * friction_tanh_scl[base + 2])
    fric0 = wp.clamp(fric0, -wp.abs(tau_m_clip_0), wp.abs(tau_m_clip_0))
    fric1 = wp.clamp(fric1, -wp.abs(tau_m_clip_1), wp.abs(tau_m_clip_1))
    fric2 = wp.clamp(fric2, -wp.abs(tau_m_clip_2), wp.abs(tau_m_clip_2))
    tau_m_clip_0 = tau_m_clip_0 - fric0
    tau_m_clip_1 = tau_m_clip_1 - fric1
    tau_m_clip_2 = tau_m_clip_2 - fric2

    # τ_j = J^T · τ_m
    tau_j_y = j00 * tau_m_clip_0
    tau_j_c = j11 * tau_m_clip_1 + j21 * tau_m_clip_2
    tau_j_m = j22 * tau_m_clip_2

    # Safety net 은 motor 출력 tau_j 자체에 적용 (gravcomp 차감 전).
    safety_y = wp.abs(j00) * tm0
    safety_c = wp.abs(j11) * tm1 + wp.abs(j21) * tm2
    safety_m = wp.abs(j22) * tm2
    tau_j_y  = wp.clamp(tau_j_y, -safety_y, safety_y)
    tau_j_c  = wp.clamp(tau_j_c, -safety_c, safety_c)
    tau_j_m  = wp.clamp(tau_j_m, -safety_m, safety_m)

    joint_f_y = tau_j_y - g_y
    joint_f_c = tau_j_c - g_c
    joint_f_m = tau_j_m - g_mp

    out_joint_f[yaw_idx] = joint_f_y
    out_joint_f[cmc_idx] = joint_f_c
    out_joint_f[mcp_idx] = joint_f_m
    out_ke[yaw_idx] = wp.float32(0.0)
    out_ke[cmc_idx] = wp.float32(0.0)
    out_ke[mcp_idx] = wp.float32(0.0)
    out_kd[yaw_idx] = wp.float32(0.0)
    out_kd[cmc_idx] = wp.float32(0.0)
    out_kd[mcp_idx] = wp.float32(0.0)
    out_eff[yaw_idx] = safety_y
    out_eff[cmc_idx] = safety_c
    out_eff[mcp_idx] = safety_m
    out_det[tid] = wp.abs(j00 * j11 * j22)


# ----------------------------------------------------------------------------
# [11] pd_clip_joint_nominal — pd_mode='joint_nominal' (X 모드의 nominal-pose
# 고정 변형). J(q=0) 으로 사전계산된 K_j_nom / Kv_j_nom / τ_j_max_nom 사용,
# 자세 변화 / via event 무시. 순수 joint-space PD + joint-space scalar clip.
#
# 기존 5 개 motor-space 커널 (pd_clip_*) 대신 이 단일 커널만 launch 됨.
# dim = total active motors (= 48 for ALLEX). 한 motor = 한 joint (그룹 내부
# coupling 은 K_j_nom 의 사전계산에서 이미 반영됨).
#
# mech-comp (is_mech_comp[tid] == 1) — Waist_Lower_Pitch, Neck_Pitch:
#   gravcomp 합산 skip → motor clip 식에서 PD 만 clip
# non-mech (== 0):
#   PD + gravcomp 합산 후 joint-space clip → joint_f - gravcomp 로 write
# ----------------------------------------------------------------------------
@wp.kernel
def pd_clip_joint_nominal(
    dof_idx: wp.array(dtype=int),          # size = N_active (48)
    is_mech_comp: wp.array(dtype=int),     # 1=mech, 0=actuator
    K_j_nom: wp.array(dtype=float),        # 사전계산 J(0)²·K_m diag
    Kv_j_nom: wp.array(dtype=float),       # 사전계산 J(0)²·Kv_m diag
    tau_j_max_nom: wp.array(dtype=float),  # 사전계산 |J(0)|·τ_m_max
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    qt: wp.array(dtype=float),
    qdt: wp.array(dtype=float),
    gravcomp_j: wp.array(dtype=float),
    out_joint_f: wp.array(dtype=float),
    out_ke: wp.array(dtype=float),
    out_kd: wp.array(dtype=float),
    out_eff: wp.array(dtype=float),
):
    tid = wp.tid()
    i = dof_idx[tid]
    K_j  = K_j_nom[tid]
    Kv_j = Kv_j_nom[tid]
    tau_max = tau_j_max_nom[tid]

    # Joint-space PD
    PD = K_j * (qt[i] - q[i]) + Kv_j * (qdt[i] - qd[i])

    # mech-comp 분기 — gravcomp 합산 skip
    g_j = wp.where(is_mech_comp[tid] == 1, wp.float32(0.0), gravcomp_j[i])
    tau_total = PD + g_j

    # Joint-space scalar clip
    tau_clip = wp.clamp(tau_total, -tau_max, tau_max)

    # MuJoCo passive 가 gravcomp_j 다시 더하므로 차감
    joint_f_val = tau_clip - g_j

    out_joint_f[i] = joint_f_val
    out_ke[i] = wp.float32(0.0)
    out_kd[i] = wp.float32(0.0)
    out_eff[i] = tau_max
