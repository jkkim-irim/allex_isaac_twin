"""
ALLEX Digital Twin 관절 제어
"""

import json
import os
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from ..config.ros2_config import ROS2Config


class ALLEXJointController:
    """ALLEX Real2Sim 관절 매핑 및 제어"""

    def __init__(self):
        self._coupled_joints = {}
        self._ros2_subscriber_active = False
        self._topic_mode = ROS2Config.DEFAULT_TOPIC_MODE

        # current/desired 분리 저장 (각 그룹별)
        self.Hand_R_current = [0.0] * len(ROS2Config.R_HAND_JOINT_INDICES)
        self.Hand_R_desired = [0.0] * len(ROS2Config.R_HAND_JOINT_INDICES)
        self.Hand_L_current = [0.0] * len(ROS2Config.L_HAND_JOINT_INDICES)
        self.Hand_L_desired = [0.0] * len(ROS2Config.L_HAND_JOINT_INDICES)
        self.Arm_R_current = [0.0] * len(ROS2Config.R_ARM_JOINT_INDICES)
        self.Arm_R_desired = [0.0] * len(ROS2Config.R_ARM_JOINT_INDICES)
        self.Arm_L_current = [0.0] * len(ROS2Config.L_ARM_JOINT_INDICES)
        self.Arm_L_desired = [0.0] * len(ROS2Config.L_ARM_JOINT_INDICES)
        self.Waist_current = [0.0] * len(ROS2Config.WAIST_JOINT_INDICES)
        self.Waist_desired = [0.0] * len(ROS2Config.WAIST_JOINT_INDICES)
        self.Neck_current = [0.0] * len(ROS2Config.NECK_JOINT_INDICES)
        self.Neck_desired = [0.0] * len(ROS2Config.NECK_JOINT_INDICES)

    # ========================================
    # 설정
    # ========================================
    def load_coupled_joint_config(self, config_path=None):
        if config_path is None:
            config_path = os.getenv('ALLEX_JOINT_CONFIG')

        if not config_path or not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self._coupled_joints = {
            int(master): {
                'slave_joint': info['slave_joint'],
                'ratio': info['ratio'],
            }
            for master, info in config.get('coupled_joints', {}).items()
        }

    def set_ros2_subscriber_status(self, is_active):
        self._ros2_subscriber_active = is_active

    def set_topic_mode(self, topic_mode):
        self._topic_mode = topic_mode

    # ========================================
    # ROS2 콜백 (ros2_callbacks.py 흡수)
    # ========================================
    def on_joint_data_received(self, joint_values, joint_names, group_name, mode):
        """ROS2 토픽에서 수신한 관절 데이터 처리 — 노드에서 직접 호출"""
        target_group = self._determine_joint_group(group_name)
        if target_group:
            self.update_joint_group(target_group, joint_values, mode)

    def _determine_joint_group(self, group_name):
        """토픽 그룹명 → 내부 관절 그룹명 변환"""
        # 손가락별 세부 매핑
        if "Hand_R_thumb" in group_name:
            return 'hand_right_thumb'
        elif "Hand_R_index" in group_name:
            return 'hand_right_index'
        elif "Hand_R_middle" in group_name:
            return 'hand_right_middle'
        elif "Hand_R_ring" in group_name:
            return 'hand_right_ring'
        elif "Hand_R_little" in group_name:
            return 'hand_right_little'
        elif "Hand_L_thumb" in group_name:
            return 'hand_left_thumb'
        elif "Hand_L_index" in group_name:
            return 'hand_left_index'
        elif "Hand_L_middle" in group_name:
            return 'hand_left_middle'
        elif "Hand_L_ring" in group_name:
            return 'hand_left_ring'
        elif "Hand_L_little" in group_name:
            return 'hand_left_little'
        elif "Arm_R" in group_name:
            return 'right_arm'
        elif "Arm_L" in group_name:
            return 'left_arm'
        elif "waist" in group_name.lower():
            return 'waist'
        elif "neck" in group_name.lower():
            return 'neck'
        return None

    # ========================================
    # 관절 그룹 업데이트
    # ========================================
    def update_joint_group(self, group_name, joint_values, mode='current'):
        """특정 관절 그룹 값 업데이트"""
        safe_values = list(joint_values)

        # 손가락별 업데이트
        if group_name.startswith('hand_'):
            parts = group_name.split('_')
            if len(parts) >= 3:
                hand_side = parts[1]   # right / left
                finger_name = parts[2]  # thumb / index / ...
                self._update_hand_finger(hand_side, finger_name, safe_values, mode)
                return

        # 팔, 허리, 목 업데이트
        group_attr_mapping = {
            'right_hand': f'Hand_R_{mode}',
            'left_hand': f'Hand_L_{mode}',
            'right_arm': f'Arm_R_{mode}',
            'left_arm': f'Arm_L_{mode}',
            'waist': f'Waist_{mode}',
            'neck': f'Neck_{mode}',
        }

        if group_name in group_attr_mapping:
            setattr(self, group_attr_mapping[group_name], safe_values)

    def _update_hand_finger(self, hand_side, finger_name, joint_values, mode):
        """손가락별 관절 업데이트"""
        finger_mapping = {
            'thumb': [0, 5, 10],
            'index': [1, 6, 11],
            'middle': [2, 7, 12],
            'ring': [3, 8, 13],
            'little': [4, 9, 14],
        }

        if finger_name not in finger_mapping:
            return

        if hand_side == 'right':
            hand_position = getattr(self, f"Hand_R_{mode}")
        elif hand_side == 'left':
            hand_position = getattr(self, f"Hand_L_{mode}")
        else:
            return

        indices = finger_mapping[finger_name]
        for i, idx in enumerate(indices):
            if i < len(joint_values) and idx < len(hand_position):
                hand_position[idx] = joint_values[i]

    # ========================================
    # 통합 목표 위치 계산
    # ========================================
    def get_unified_target_positions(self):
        """모든 관절 그룹을 통합한 59-joint 목표 위치 반환"""
        target_positions = [0.0] * 59

        data_suffix = 'current' if self._topic_mode == ROS2Config.TOPIC_MODE_CURRENT else 'desired'

        self._map_group_to_positions('right_hand', getattr(self, f'Hand_R_{data_suffix}'), target_positions)
        self._map_group_to_positions('left_hand', getattr(self, f'Hand_L_{data_suffix}'), target_positions)
        self._map_group_to_positions('right_arm', getattr(self, f'Arm_R_{data_suffix}'), target_positions)
        self._map_group_to_positions('left_arm', getattr(self, f'Arm_L_{data_suffix}'), target_positions)
        self._map_group_to_positions('waist', getattr(self, f'Waist_{data_suffix}'), target_positions)
        self._map_group_to_positions('neck', getattr(self, f'Neck_{data_suffix}'), target_positions)

        return self.apply_coupled_joints(target_positions)

    def _map_group_to_positions(self, group_name, group_values, target_positions):
        """관절 그룹 값을 전체 위치 배열에 매핑 (degree → radian)"""
        if group_name not in ROS2Config.ALL_JOINT_INDICES:
            return
        indices = ROS2Config.ALL_JOINT_INDICES[group_name]
        for i, idx in enumerate(indices):
            if i < len(group_values) and idx < len(target_positions):
                target_positions[idx] = np.radians(group_values[i])

    def apply_coupled_joints(self, joint_positions, total_joints=59):
        """Coupled Joint 값 계산 — master joint에서 slave joint 도출"""
        extended = list(joint_positions)
        while len(extended) < total_joints:
            extended.append(0.0)

        for master_idx, info in self._coupled_joints.items():
            if master_idx < len(joint_positions):
                slave_idx = info['slave_joint']
                if slave_idx < len(extended):
                    extended[slave_idx] = joint_positions[master_idx] * info['ratio']

        return extended

    # ========================================
    # 시뮬레이션 루프 제너레이터
    # ========================================
    def create_joint_control_generator(self, articulation, get_target_positions_func):
        """관절 제어 제너레이터 — 매 physics step마다 목표 위치 적용"""
        while True:
            if articulation is not None and self._ros2_subscriber_active:
                target_positions = get_target_positions_func()
                articulation.apply_action(ArticulationAction(target_positions))
            yield

    def get_coupled_joints_info(self):
        return dict(self._coupled_joints)
