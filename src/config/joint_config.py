"""
관절 제어 관련 설정
"""

# ========================================
# 🦾 관절 제어 설정
# ========================================
class JointConfig:
    """관절 제어 관련 설정"""
    # 관절 개수
    EFFECTIVE_JOINTS = 49      # UI에서 제어할 관절 수 (0~48)
    CONTROL_JOINTS = 6         # UI 슬라이더로 제어할 관절 수
    
    # 관절 범위
    JOINT_MIN_ANGLE = -3.14159  # -π
    JOINT_MAX_ANGLE = 3.14159   # π  
    JOINT_STEP = 0.001           # 슬라이더 스텝
    
    # 관절 이름 템플릿
    JOINT_NAME_TEMPLATE = 'joint{}'  # joint1, joint2, ...
    
    # 초기화 설정
    MAX_INITIALIZATION_ATTEMPTS = 5
    INITIALIZATION_DELAY_STEPS = 3  # 몇 step 후에 초기화 시도