"""
ALLEX Digital Twin 관절 제어
"""

import json
import os
import threading
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

        # ----- Real torque buffers (position 과 동일 shape) -----
        # 실제 로봇도 motor current 기반으로 slave torque 를 계산/로깅한다는 전제.
        # slave abbr 자리도 포함되어 있어 master→slave 폴백 없이 그대로 매핑된다.
        self.Hand_R_torque = [0.0] * len(ROS2Config.R_HAND_JOINT_INDICES)
        self.Hand_L_torque = [0.0] * len(ROS2Config.L_HAND_JOINT_INDICES)
        self.Arm_R_torque = [0.0] * len(ROS2Config.R_ARM_JOINT_INDICES)
        self.Arm_L_torque = [0.0] * len(ROS2Config.L_ARM_JOINT_INDICES)
        self.Waist_torque = [0.0] * len(ROS2Config.WAIST_JOINT_INDICES)
        self.Neck_torque = [0.0] * len(ROS2Config.NECK_JOINT_INDICES)

        # Torque write/read 동기화용 Lock (ROS2 executor thread 와 physics thread 간)
        self._torque_lock = threading.Lock()

        # persistent snapshot dict — get_torque_snapshot 호출마다 재할당 피함
        self._torque_snapshot: dict = {}
        for names in (
            ROS2Config.R_HAND_JOINT_NAMES,
            ROS2Config.L_HAND_JOINT_NAMES,
            ROS2Config.R_ARM_JOINT_NAMES,
            ROS2Config.L_ARM_JOINT_NAMES,
            ROS2Config.WAIST_JOINT_NAMES,
            ROS2Config.NECK_JOINT_NAMES,
        ):
            for abbr in names:
                self._torque_snapshot[abbr] = 0.0

    # ========================================
    # 설정
    # ========================================
    def load_coupled_joint_config(self, config_path=None):
        if config_path is None:
            config_path = os.getenv('ALLEX_JOINT_CONFIG')

        if not config_path or not os.path.exists(config_path):
            # 환경변수 미설정 시 extension 내부 기본 경로로 폴백
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "joint_config.json")
            if os.path.exists(default_path):
                config_path = default_path
            else:
                print(f"[ALLEX] Joint config not found: {config_path} (default: {default_path})")
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
        """모든 관절 그룹을 통합한 60-joint 목표 위치 반환"""
        target_positions = [0.0] * 60

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

    def apply_coupled_joints(self, joint_positions, total_joints=60):
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
    def create_joint_control_generator(self, articulation, get_target_positions_func,
                                        is_external_active_fn=None,
                                        torque_plot_fn=None):
        """관절 제어 제너레이터 — 매 physics step마다 목표 위치 적용.

        is_external_active_fn: optional callable returning True when an external
        driver (e.g. trajectory playback) wants to push targets even if the ROS2
        subscriber is off.
        """
        while True:
            active = self._ros2_subscriber_active
            if not active and is_external_active_fn is not None:
                try:
                    active = bool(is_external_active_fn())
                except Exception:
                    active = False
            if articulation is not None and active:
                target_positions = get_target_positions_func()
                articulation.apply_action(ArticulationAction(target_positions))
            if torque_plot_fn is not None:
                try:
                    torque_plot_fn()
                except Exception as exc:
                    print(f"[ALLEX][TorquePlot] hook failed: {exc}")
            yield

    def get_coupled_joints_info(self):
        return dict(self._coupled_joints)

    # ========================================
    # Torque (real) 수신 처리
    # ========================================
    def on_torque_data_received(self, values, joint_names, group_name):
        """Real torque 토픽 수신 콜백.

        joint_names 는 약어(L11 …) 리스트. group_name 은 OUTBOUND_TORQUE_TOPIC_TO_JOINTS 의 key.
        실제 로봇은 master/slave 구분 없이 motor current 기반 slave torque 를 로깅하므로
        해당 그룹 버퍼에 그대로 저장한다 (mimic 폴백 없음).
        ROS2 executor thread 에서 호출되므로 Lock 으로 보호.
        """
        target_group = self._determine_joint_group(group_name)
        if target_group is None:
            return
        with self._torque_lock:
            self._update_torque_group(target_group, list(values), joint_names)

    def _update_torque_group(self, group_name, torque_values, joint_names):
        """position 의 update_joint_group 과 대칭인 torque 저장 로직."""
        # 손가락 세부 그룹
        if group_name.startswith('hand_'):
            parts = group_name.split('_')
            if len(parts) >= 3:
                hand_side = parts[1]
                finger_name = parts[2]
                self._update_hand_finger_torque(hand_side, finger_name, torque_values)
                return

        group_attr_mapping = {
            'right_hand': 'Hand_R_torque',
            'left_hand': 'Hand_L_torque',
            'right_arm': 'Arm_R_torque',
            'left_arm': 'Arm_L_torque',
            'waist': 'Waist_torque',
            'neck': 'Neck_torque',
        }
        if group_name in group_attr_mapping:
            setattr(self, group_attr_mapping[group_name], torque_values)

    def _update_hand_finger_torque(self, hand_side, finger_name, torque_values):
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
            buf = self.Hand_R_torque
        elif hand_side == 'left':
            buf = self.Hand_L_torque
        else:
            return
        indices = finger_mapping[finger_name]
        for i, idx in enumerate(indices):
            if i < len(torque_values) and idx < len(buf):
                buf[idx] = torque_values[i]

    def get_torque_snapshot(self):
        """약어(L11 …) → torque 값 dict 를 반환.

        visualizer 는 HAND_JOINT_TORQUE_RING_MAP 의 dof_abbr 로 조회한다.
        모든 abbr(40개 hand + arm/waist/neck 포함) 가 항상 dict key 에 존재.
        실제 torque 토픽이 아직 없는 abbr 는 0.0 으로 유지된다.

        master/slave 구분 없이 그룹 버퍼를 그대로 반환한다 — slave abbr 은
        실제 로봇 측에서 motor current 기반으로 계산된 값이 해당 그룹에
        채워져 있다는 전제.

        반환 dict 는 persistent buffer 이며, 호출자는 per-key 단일 float 조회만
        수행한다고 가정한다 (Python dict 의 per-key read 는 GIL 하에서 atomic).
        dict 전체 스냅샷이 필요하면 호출자가 `dict(snapshot)` 으로 복사할 것.
        """
        groups = [
            (ROS2Config.R_HAND_JOINT_NAMES, self.Hand_R_torque),
            (ROS2Config.L_HAND_JOINT_NAMES, self.Hand_L_torque),
            (ROS2Config.R_ARM_JOINT_NAMES, self.Arm_R_torque),
            (ROS2Config.L_ARM_JOINT_NAMES, self.Arm_L_torque),
            (ROS2Config.WAIST_JOINT_NAMES, self.Waist_torque),
            (ROS2Config.NECK_JOINT_NAMES, self.Neck_torque),
        ]
        snapshot = self._torque_snapshot
        with self._torque_lock:
            for names, values in groups:
                for i, abbr in enumerate(names):
                    if i < len(values):
                        snapshot[abbr] = float(values[i])
                    else:
                        snapshot[abbr] = 0.0
        return snapshot
