"""
ALLEX Digital Twin 초기화 및 설정 관리
"""

from isaacsim.core.utils.viewports import set_camera_view

from ..utils.constants import (
    TOTAL_JOINTS, EFFECTIVE_JOINTS, DEFAULT_CAMERA_EYE, DEFAULT_CAMERA_TARGET, DEFAULT_CAMERA_PRIM_PATH
)

class ALLEXInitializer:
    """ALLEX 디지털 트윈 초기화 및 설정을 담당하는 클래스"""
    
    def __init__(self):
        """초기화 파라미터 설정"""
        # 🆕 59자유도 전체 지원
        self._total_joints = TOTAL_JOINTS
        self._effective_joints = EFFECTIVE_JOINTS
        
        # 관절 위치 초기화
        self._target_joint_positions = [0.0] * self._total_joints
        self._controlled_joint_indices = list(range(self._total_joints))
        
        # Coupled Joint 설정
        self._coupled_joints = {}

    def setup_camera_view(self):
        """카메라 뷰 설정"""
        set_camera_view(
            eye=DEFAULT_CAMERA_EYE, 
            target=DEFAULT_CAMERA_TARGET, 
            camera_prim_path=DEFAULT_CAMERA_PRIM_PATH
        )

    def initialize_joint_positions(self, articulation):
        """관절 위치 초기화 (안전한 버전)"""
        if articulation is None:
            print("⚠️ Articulation 객체가 None입니다")
            return
            
        try:
            # 🆕 안전한 관절 위치 조회
            current_positions = None
            try:
                current_positions = articulation.get_joint_positions()
            except Exception as pos_error:
                print(f"⚠️ 관절 위치 조회 실패: {pos_error}")
                # 기본값으로 초기화
                self._target_joint_positions = [0.0] * self._total_joints
                return
            
            # None 체크 추가
            if current_positions is None:
                print("⚠️ 관절 위치가 None입니다. 기본값으로 초기화합니다")
                self._target_joint_positions = [0.0] * self._total_joints
                return
                
            # 길이 체크 추가
            if len(current_positions) > 0:
                # 실제 관절 개수로 목표 위치 배열 크기 조정
                actual_joint_count = len(current_positions)
                self._target_joint_positions = list(current_positions)

                # 부족한 부분은 0으로 채움
                while len(self._target_joint_positions) < self._total_joints:
                    self._target_joint_positions.append(0.0)

            else:
                print("⚠️ 관절 위치 배열이 비어있습니다. 기본값으로 초기화합니다")
                self._target_joint_positions = [0.0] * self._total_joints
                
        except Exception as e:
            print(f"❌ 관절 위치 초기화 실패: {e}")
            print("🔄 기본값으로 초기화합니다")
            self._target_joint_positions = [0.0] * self._total_joints

    def reset(self, articulation):
        """시스템 리셋"""
        # 현재 관절 위치로 리셋 (0이 아닌)
        if articulation is not None:
            current_positions = articulation.get_joint_positions()
            if len(current_positions) > 0:
                self._target_joint_positions = list(current_positions)

    @property
    def target_joint_positions(self):
        """목표 관절 위치 반환"""
        return self._target_joint_positions

    @target_joint_positions.setter 
    def target_joint_positions(self, positions):
        """목표 관절 위치 설정"""
        self._target_joint_positions = positions
