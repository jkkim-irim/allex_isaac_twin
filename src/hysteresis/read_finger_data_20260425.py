#!/usr/bin/env python3
"""
작성자: 문준서 and Gemini
2024-06-25: 관절 간 중첩이 적용된 히스테리시스 모델 최적화 및 피팅 스크립트
- 관절 간 중첩(Coupling)을 반영한 다변량 Prandtl-Ishlinskii 모델 구현
- scipy.optimize.least_squares를 사용하여 모델 파라미터 최적화

"""

import sys, os
import pandas as pd
import numpy as np
from src.scripts.core.curve_fitting import curvefitting_2d, curvefitting_j2m
from  src.scripts.core.polyfitn import polyfitn, polyvaln

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares

def smooth_max(a, b, epsilon=1e-3):
    """미분 가능한 부드러운 최대값 함수"""
    return 0.5 * (a + b + np.sqrt((a - b)**2 + epsilon))

def smooth_min(a, b, epsilon=1e-3):
    """미분 가능한 부드러운 최소값 함수"""
    return 0.5 * (a + b - np.sqrt((a - b)**2 + epsilon))

def hyst_model_coupled(params, c_t, H_tm1):
    """
    관절 간 중첩(Coupling)을 반영한 다변량 Prandtl-Ishlinskii 모델
    
    params: 15개의 최적화 파라미터 (1D 배열 또는 리스트)
    c_t: 현재 모터 명령값 [abad, mcp, pip]
    H_tm1: 이전 시간(t-1)의 Play 연산자 내부 상태값 [H_abad, H_mcp, H_pip]
    """
    # 1. 파라미터 언패킹 (총 15개)
    # r: 각 관절 모터의 고유 문턱값 (3개)
    r_0, r_1, r_2 = params[0:3]
    
    # w: 히스테리시스 가중치 행렬 성분 (6개) - 하삼각 구조
    w_00 = params[3]                    # Abad -> Abad
    w_10, w_11 = params[4:6]            # Abad -> MCP,  MCP -> MCP
    w_20, w_21, w_22 = params[6:9]      # Abad -> PIP,  MCP -> PIP,  PIP -> PIP
    
    # q: 선형 기구학 가중치 행렬 성분 (6개) - 하삼각 구조
    q_00 = params[9]                    # Abad -> Abad
    q_10, q_11 = params[10:12]          # Abad -> MCP,  MCP -> MCP
    q_20, q_21, q_22 = params[12:15]    # Abad -> PIP,  MCP -> PIP,  PIP -> PIP


    def Fr(c, h_prev, r):
        """Play Operator (Backlash 연산자)"""
        #return max(c - r, min(c + r, h_prev))  # 원래의 비부드러운 버전
        return smooth_max(c - r, smooth_min(c + r, h_prev)) # softmax 사용 버전

    y_tpred = [0.0, 0.0, 0.0]
    
    # 각 관절별 Play 연산자 결과 계산 (각자의 문턱값 r 적용)
    H_0 = Fr(c_t[0], H_tm1[0], r_0)
    H_1 = Fr(c_t[1], H_tm1[1], r_1)
    H_2 = Fr(c_t[2], H_tm1[2], r_2)

    # 1. Abad 조인트 (독립적)
    y_tpred[0] = (w_00 * H_0) + (q_00 * c_t[0])
    
    # 2. MCP 조인트 (Abad의 히스테리시스 + 선형 움직임 모두 중첩)
    y_tpred[1] = (w_10 * H_0 + w_11 * H_1) + \
                 (q_10 * c_t[0] + q_11 * c_t[1])
    
    # 3. PIP 조인트 (Abad, MCP의 히스테리시스 + 선형 움직임 모두 중첩)
    y_tpred[2] = (w_20 * H_0 + w_21 * H_1 + w_22 * H_2) + \
                 (q_20 * c_t[0] + q_21 * c_t[1] + q_22 * c_t[2])

    # 주의: 다음 스텝(t+1)을 위해 반드시 예측 각도가 아닌 'Play 상태값(H)'을 반환해야 함
    return y_tpred, [H_0, H_1, H_2]


def hyst_model_coupled_2(params, c_t, H_tm1):
    """
    관절 간 중첩(Coupling)을 반영한 다변량 Prandtl-Ishlinskii 모델
    
    params: 15개의 최적화 파라미터 (1D 배열 또는 리스트)
    c_t: 현재 모터 명령값 [abad, mcp, pip]
    H_tm1: 이전 시간(t-1)의 Play 연산자 내부 상태값 [H_abad, H_mcp, H_pip]
    """
    # 1. 파라미터 언패킹 (총 15개)
    # r: 각 관절 모터의 고유 문턱값 (3개)
    r_0, r_1, r_2 = params[0:3]
    
    # w: 히스테리시스 가중치 행렬 성분 (6개) - 하삼각 구조
    w_00 = params[3]                    # Abad -> Abad
    w_10, w_11 = params[4:6]            # Abad -> MCP,  MCP -> MCP
    w_20, w_21, w_22 = params[6:9]      # Abad -> PIP,  MCP -> PIP,  PIP -> PIP
    
    # q: 선형 기구학 가중치 행렬 성분 (6개) - 하삼각 구조
    q_00 = params[9]                    # Abad -> Abad
    q_10, q_11 = params[10:12]          # Abad -> MCP,  MCP -> MCP
    q_20, q_21, q_22 = params[12:15]    # Abad -> PIP,  MCP -> PIP,  PIP -> PIP


    def Fr(c, h_prev, r):
        """Play Operator (Backlash 연산자)"""
        return max(c - r, min(c + r, h_prev))  # 원래의 비부드러운 버전
        return smooth_max(c - r, smooth_min(c + r, h_prev)) # softmax 사용 버전

    y_tpred = [0.0, 0.0, 0.0]
    
    # 각 관절별 Play 연산자 결과 계산 (각자의 문턱값 r 적용)
    H_0 = Fr(c_t[0], H_tm1[0], r_0)
    H_1 = Fr(c_t[1], H_tm1[1], r_1)
    H_2 = Fr(c_t[2], H_tm1[2], r_2)

    # 1. Abad 조인트 (독립적)
    y_tpred[0] = (w_00 * H_0) + (q_00 * c_t[0])
    
    # 2. MCP 조인트 (Abad의 히스테리시스 + 선형 움직임 모두 중첩)
    y_tpred[1] = (w_10 * H_0 + w_11 * H_1) + \
                 (q_10 * c_t[0] + q_11 * c_t[1])
    
    # 3. PIP 조인트 (Abad, MCP의 히스테리시스 + 선형 움직임 모두 중첩)
    y_tpred[2] = (w_20 * H_0 + w_21 * H_1 + w_22 * H_2) + \
                 (q_20 * c_t[0] + q_21 * c_t[1] + q_22 * c_t[2])

    # 주의: 다음 스텝(t+1)을 위해 반드시 예측 각도가 아닌 'Play 상태값(H)'을 반환해야 함
    return y_tpred, [H_0, H_1, H_2]




import numpy as np
from scipy.optimize import least_squares

# (이전에 작성한 hyst_model_coupled 함수가 위에 있다고 가정합니다)

def fit_hysteresis_model(commands, actual_joints):
    """
    관절 간 중첩이 적용된 히스테리시스 모델 최적화 피팅 함수
    
    Parameters:
    commands (ndarray): 모터 명령값 데이터, shape (N, 3)
    actual_joints (ndarray): 실제 측정된 관절 각도 데이터, shape (N, 3)
    """
    commands = np.asarray(commands)
    actual_joints = np.asarray(actual_joints)
    N = len(commands)

    def calculate_residuals(params):
        """최적화 알고리즘이 반복 호출하며 오차를 평가하는 내부 함수"""
        predictions = np.zeros((N, 3))
        
        # 첫 번째 스텝의 초기 H 상태 (초기에는 유격이 없다고 가정하고 첫 명령값으로 세팅)
        H_prev = [commands[0, 0], commands[0, 1], commands[0, 2]]
        
        # 시간(t)의 흐름에 따라 순차적으로 모델을 시뮬레이션
        for t in range(N):
            y_pred, H_curr = hyst_model_coupled(params, commands[t], H_prev)
            predictions[t] = y_pred
            H_prev = H_curr  # 다음 스텝을 위해 H 상태 업데이트
            
        # 예측값과 실제값의 차이(오차) 계산 후 1차원 배열로 펼쳐서 반환 
        # (scipy.least_squares의 필수 요구사항)
        error = actual_joints - predictions
        return error.flatten()

    # 1. 초기 추정값 설정 (총 15개 파라미터)
    # 팁: 최적화가 길을 잃지 않도록 물리적으로 말이 되는 초기값을 주는 것이 중요합니다.
    initial_params = [
        # r (문턱값): 약간의 유격이 있다고 가정
        0.5, 0.5, 0.5,           
        # w (히스테리시스 가중치): 초기엔 영향을 모른다고 가정하고 0.0 또는 작은 값 부여
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        # q (선형 가중치): 모터가 1도 돌면 관절도 1도 돈다고 가정(대각성분 1.0), 크로스 커플링은 0.0
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0  
    ]

    # 2. 파라미터 경계 조건 (Bounds) 설정
    # 문턱값(r)은 '물리적인 유격 크기'이므로 음수가 될 수 없습니다. (0 이상이어야 함)
    # 나머지는 일단 무한대(-inf ~ inf)로 둡니다.
    lower_bounds = [
        0, 0, 0,                                                    # r 범위
        -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,       # w 범위
        -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf        # q 범위
    ]
    upper_bounds = np.inf

    print("[*] 최적화 진행 중... (시간이 다소 소요될 수 있습니다)")
    
    # 3. 최적화 수행 (Levenberg-Marquardt 대신 bounds를 지원하는 trf 메서드 사용)
    result = least_squares(calculate_residuals, 
                           initial_params, 
                           bounds=(lower_bounds, upper_bounds),
                           method='trf', # 경계 조건이 있을 때 추천되는 알고리즘
                           verbose=1)    # 진행 상황 출력 (원치 않으면 0으로 설정)

    print("[*] 최적화 완료!")
    
    return result.x  # 최적화된 15개의 파라미터 반환


def predict_hysteresis(fitted_params, commands):
    """
    학습된 파라미터를 사용하여 전체 시계열 모터 명령에 대한 관절 각도를 예측합니다.
    
    Parameters:
    fitted_params (list or ndarray): 최적화된 15개의 파라미터
    commands (ndarray): 모터 명령값 데이터, shape (N, 3)
    
    Returns:
    ndarray: 예측된 관절 각도 데이터, shape (N, 3)
    """
    commands = np.asarray(commands)
    N = len(commands)
    predictions = np.zeros((N, 3))
    
    # 첫 번째 스텝의 초기 H 상태 세팅 (초기 유격 0 가정)
    H_prev = [commands[0, 0], commands[0, 1], commands[0, 2]]
    
    # 시간 순서대로 시뮬레이션 진행
    for t in range(N):
        y_pred, H_curr = hyst_model_coupled(fitted_params, commands[t], H_prev)
        predictions[t] = y_pred
        H_prev = H_curr  # 상태 업데이트
        
    return predictions

def read_csv_motor_and_joint(csv1="src/data/joint_trajectory/command_joint_deg_0.csv", 
                             csv2="src/data/joint_trajectory/measured_joint_deg_0.csv"):
    """CSV 파일을 읽어와서 전처리 후 반환하는 함수"""
    if not os.path.exists(csv1) or not os.path.exists(csv2):
        raise FileNotFoundError("파일 경로를 확인해주세요.")

    T1 = pd.read_csv(csv1)  # Motor Command (명령값)
    T2 = pd.read_csv(csv2)  # Measured Joint (실제값)

    # MATLAB 스타일의 1-based indexing 시작점 처리 (필요한 경우)
    start = 1
    motor_ang = T1.iloc[start - 1:].reset_index(drop=True)
    joint_ang = T2.iloc[start - 1:].reset_index(drop=True)

    return motor_ang, joint_ang


def plot_comparison(motor_ang, joint_ang, fitted_joint_ang_poly, fitted_joint_ang_hyst):
    """3. 명령값, 실제값, 피팅값을 모두 비교하는 함수"""
    cols = ['abad_deg', 'mcp_deg', 'pip_deg']
    titles = ['Abduction/Adduction', 'MCP Joint', 'PIP Joint']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Kinematics Comparison: Command vs Actual vs Fitted', fontsize=16)

    for i, col in enumerate(cols):
        # 1. 명령값 (Target)
        axes[i].plot(motor_ang['t'], motor_ang[col], 
                     label='Commanded', linestyle='--', color='gray', alpha=0.5)
        
        # 2. 실제 측정값 (Ground Truth)
        axes[i].plot(joint_ang['t'], joint_ang[col], 
                     label='Actual', linestyle='-', color='blue', alpha=0.7)
        
        # 3. 2차 피팅값 (Model Prediction)
        axes[i].plot(fitted_joint_ang_poly['t'], fitted_joint_ang_poly[col], 
                     label='2nd-order Fitted', linestyle='-', color='red', linewidth=2)
        
        # 4. 히스테리시스 피팅값 (Model Prediction)
        axes[i].plot(fitted_joint_ang_hyst['t'], fitted_joint_ang_hyst[col], 
                     label='Hysteresis Fitted', linestyle='-.', color='green', linewidth=2)

        axes[i].set_ylabel('Angle (deg)')
        axes[i].set_title(f'Joint: {titles[i]}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle=':', alpha=0.6)

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_errors(motor_ang, joint_ang, fitted_joint_ang_poly, fitted_joint_ang_hyst):
    """4. 각 모델별 오차(Error = Actual - Predicted)를 시계열로 비교하는 함수"""
    cols = ['abad_deg', 'mcp_deg', 'pip_deg']
    titles = ['Abduction/Adduction Error', 'MCP Joint Error', 'PIP Joint Error']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Error Analysis: Actual - Predicted', fontsize=16)

    for i, col in enumerate(cols):
        # 1. Raw Error (실제값 - 원본 명령값)
        err_raw = joint_ang[col] - motor_ang[col]
        axes[i].plot(joint_ang['t'], err_raw, 
                     label='Raw Error (Actual-Cmd)', linestyle='--', color='gray', alpha=0.5)
        
        # 2. 다항식 피팅 오차
        err_poly = joint_ang[col] - fitted_joint_ang_poly[col]
        axes[i].plot(joint_ang['t'], err_poly, 
                     label='2nd-order Poly Error', linestyle='-', color='red', alpha=0.8)
        
        # 3. 히스테리시스 피팅 오차
        err_hyst = joint_ang[col] - fitted_joint_ang_hyst[col]
        axes[i].plot(joint_ang['t'], err_hyst, 
                     label='Hysteresis Error', linestyle='-', color='green', alpha=0.8)

        # 0점 기준선 (오차가 0인 이상적인 지점)
        axes[i].axhline(0, color='black', linewidth=1, linestyle=':')

        axes[i].set_ylabel('Error (deg)')
        axes[i].set_title(f'{titles[i]}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle=':', alpha=0.6)

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def fit_joint_angles(motor_ang, joint_ang, order_abad=3, order_mcp=3, order_pip=3):
    """다변량 2차/3차 다항식 피팅을 수행하는 함수 (polyfitn 적용)"""
    cols = ['abad_deg', 'mcp_deg', 'pip_deg']
    fitted_data_poly = pd.DataFrame()
    fitted_data_poly['t'] = motor_ang['t']  # 시간축 복사
    
    fitted_data_hyst = pd.DataFrame()
    fitted_data_hyst['t'] = motor_ang['t']  # 시간축 복사

    # 1. 독립 변수 행렬 X 구성 (N, 3)
    # 순서: abad_deg(0), mcp_deg(1), pip_deg(2)
    X = motor_ang[cols].values 
    
    # 2. Abad 피팅 (모터 1개 참조: X[:, 0:1])
    y_abad = joint_ang['abad_deg'].values
    mdl_abad = polyfitn(X[:, 0:1], y_abad, order_abad)
    fitted_data_poly['abad_deg'] = polyvaln(mdl_abad, X[:, 0:1])
    print(f"[*] abad_deg Fit Coeffs (Order {order_abad}): {mdl_abad.coefficients}")
    
    # 3. MCP 피팅 (모터 2개 참조: X[:, 0:2])
    y_mcp = joint_ang['mcp_deg'].values
    mdl_mcp = polyfitn(X[:, 0:2], y_mcp, order_mcp)
    fitted_data_poly['mcp_deg'] = polyvaln(mdl_mcp, X[:, 0:2])
    print(f"[*] mcp_deg Fit Coeffs (Order {order_mcp}): {mdl_mcp.coefficients}")
    
    # 4. PIP 피팅 (모터 3개 참조: X[:, 0:3])
    y_pip = joint_ang['pip_deg'].values
    mdl_pip = polyfitn(X[:, 0:3], y_pip, order_pip)
    fitted_data_poly['pip_deg'] = polyvaln(mdl_pip, X[:, 0:3])
    print(f"[*] pip_deg Fit Coeffs (Order {order_pip}): {mdl_pip.coefficients}")


    # 2. 히스테리시스 모델 피팅 및 예측
    result = fit_hysteresis_model(motor_ang[cols].values, joint_ang[cols].values)
    fitted_data_hyst[cols] = predict_hysteresis(result, motor_ang[cols].values)

    return fitted_data_poly, fitted_data_hyst




# --- 실행 흐름 ---
if __name__ == "__main__":
    try:
        # 1. 데이터 로드
        m_data, j_data = read_csv_motor_and_joint()
        
        # 2. 2차 피팅 수행
        f_data_poly, f_data_hyst = fit_joint_angles(m_data, j_data)

        
        # 3. 파라미터 MSE, Peak Error 계산 및 출력
        mse_raw = np.mean((j_data[['abad_deg', 'mcp_deg', 'pip_deg']].values - m_data[['abad_deg', 'mcp_deg', 'pip_deg']].values) ** 2)
        mse_poly = np.mean((j_data[['abad_deg', 'mcp_deg', 'pip_deg']].values - f_data_poly[['abad_deg', 'mcp_deg', 'pip_deg']].values) ** 2)
        mse_hyst = np.mean((j_data[['abad_deg', 'mcp_deg', 'pip_deg']].values - f_data_hyst[['abad_deg', 'mcp_deg', 'pip_deg']].values) ** 2)
        
        print("\n" + "="*45)
        print(f"| {'Model Type':<25} | {'MSE':<13} |")
        print("-" * 45)
        print(f"| {'Raw Command vs Actual':<25} | {mse_raw:>13.4f} |")
        print(f"| {'2nd-order Poly Fit':<25} | {mse_poly:>13.4f} |")
        print(f"| {'Hysteresis Fit':<25} | {mse_hyst:>13.4f} |")
        print("="*45 + "\n")
        
        # 4. 결과 시각화 (3개 데이터 비교)
        plot_comparison(m_data, j_data, f_data_poly, f_data_hyst)

        # 5. 오차(Error) 시각화 비교
        plot_errors(m_data, j_data, f_data_poly, f_data_hyst)

        
    except Exception as e:
        print(f"오류 발생: {e}")



