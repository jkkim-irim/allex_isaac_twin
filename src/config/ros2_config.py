"""
ROS2 통신 관련 설정
"""

import os


class ROS2Topics:
    """ROS2 토픽 이름들"""
    ROBOT_STATE = '/robot_state'


class ROS2QoS:
    """ROS2 QoS 설정"""
    HISTORY_DEPTH = 10


class ROS2Config:
    """ROS2 통신 전반적인 설정"""
    NODE_NAME = 'isaac_sim_integrated_ros2'
    BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
    DOMAIN_ID = int(os.environ.get('ROS_DOMAIN_ID', '77'))
    RMW_IMPLEMENTATION = os.environ.get('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')

    INIT_TIMEOUT = 5.0
    SHUTDOWN_TIMEOUT = 1.0
    THREAD_DAEMON = True
    EXECUTOR_TIMEOUT = 0.1

    # 토픽 모드 설정
    TOPIC_MODE_CURRENT = 'joint_positions_deg'
    TOPIC_MODE_DESIRED = 'joint_ang_target_deg'
    DEFAULT_TOPIC_MODE = TOPIC_MODE_CURRENT

    TOPIC_SUFFIXES = {
        TOPIC_MODE_CURRENT: 'joint_positions_deg',
        TOPIC_MODE_DESIRED: 'joint_ang_target_deg',
    }

    TOPIC_MODE_DISPLAY_NAMES = {
        TOPIC_MODE_CURRENT: 'Current',
        TOPIC_MODE_DESIRED: 'Desired',
    }

    # 관절 인덱스 (dof_names 순서 기준, 2026-04-22 확인)
    # passive joints (제어 안 함): [2]=Waist_Upper_Pitch, [59]=Waist_Pitch_Dummy,
    #   L DIP: 13,17,21,25,29 / R DIP: 42,46,50,54,58
    L_ARM_JOINT_INDICES  = [3, 4, 5, 6, 7, 8, 9]    # LSP,LSR,LSY,LEP,LWY,LWR,LWP
    R_ARM_JOINT_INDICES  = [32, 33, 34, 35, 36, 37, 38]  # RSP,RSR,RSY,REP,RWY,RWR,RWP
    WAIST_JOINT_INDICES  = [0, 1]                    # WY, WP(Lower_Pitch)
    NECK_JOINT_INDICES   = [30, 31]                  # NP, NY
    # L_HAND 순서: L11,L21,L31,L41,L51, L12,L22,L32,L42,L52, L13,L23,L33,L43,L53
    L_HAND_JOINT_INDICES = [26, 10, 18, 22, 14,  27, 11, 19, 23, 15,  28, 12, 20, 24, 16]
    # R_HAND 순서: R11,R21,R31,R41,R51, R12,R22,R32,R42,R52, R13,R23,R33,R43,R53
    R_HAND_JOINT_INDICES = [55, 39, 47, 51, 43,  56, 40, 48, 52, 44,  57, 41, 49, 53, 45]

    # 관절 이름
    R_HAND_JOINT_NAMES = [
        "R11", "R21", "R31", "R41", "R51",
        "R12", "R22", "R32", "R42", "R52",
        "R13", "R23", "R33", "R43", "R53",
    ]
    L_HAND_JOINT_NAMES = [
        "L11", "L21", "L31", "L41", "L51",
        "L12", "L22", "L32", "L42", "L52",
        "L13", "L23", "L33", "L43", "L53",
    ]
    R_ARM_JOINT_NAMES = ["RSP", "RSR", "RSY", "REP", "RWY", "RWR", "RWP"]
    L_ARM_JOINT_NAMES = ["LSP", "LSR", "LSY", "LEP", "LWY", "LWR", "LWP"]
    WAIST_JOINT_NAMES = ["WY", "WP"]
    NECK_JOINT_NAMES = ["NP", "NY"]

    ALL_JOINT_INDICES = {
        'right_hand': R_HAND_JOINT_INDICES,
        'left_hand': L_HAND_JOINT_INDICES,
        'right_arm': R_ARM_JOINT_INDICES,
        'left_arm': L_ARM_JOINT_INDICES,
        'waist': WAIST_JOINT_INDICES,
        'neck': NECK_JOINT_INDICES,
    }

    ALL_JOINT_NAMES = {
        'right_hand': R_HAND_JOINT_NAMES,
        'left_hand': L_HAND_JOINT_NAMES,
        'right_arm': R_ARM_JOINT_NAMES,
        'left_arm': L_ARM_JOINT_NAMES,
        'waist': WAIST_JOINT_NAMES,
        'neck': NECK_JOINT_NAMES,
    }

    # 14개 토픽 관절 매핑
    OUTBOUND_TOPIC_TO_JOINTS = {
        "Arm_R_theOne": ["RSP", "RSR", "RSY", "REP", "RWY", "RWR", "RWP"],
        "Hand_R_thumb_wir": ["R11", "R12", "R13"],
        "Hand_R_index_wir": ["R21", "R22", "R23"],
        "Hand_R_middle_wir": ["R31", "R32", "R33"],
        "Hand_R_ring_wir": ["R41", "R42", "R43"],
        "Hand_R_little_wir": ["R51", "R52", "R53"],
        "Arm_L_theOne": ["LSP", "LSR", "LSY", "LEP", "LWY", "LWR", "LWP"],
        "Hand_L_thumb_wir": ["L11", "L12", "L13"],
        "Hand_L_index_wir": ["L21", "L22", "L23"],
        "Hand_L_middle_wir": ["L31", "L32", "L33"],
        "Hand_L_ring_wir": ["L41", "L42", "L43"],
        "Hand_L_little_wir": ["L51", "L52", "L53"],
        "theOne_waist": ["WY", "WP"],
        "theOne_neck": ["NP", "NY"],
    }

    # ========================================
    # Torque 토픽 (real2sim 토크 시각화용)
    # TODO: rosbag torque 토픽 패턴 확정 시 채울 것.
    #       형식은 OUTBOUND_TOPIC_TO_JOINTS 와 동일하게
    #         { "<topic_key>": ["R11", "R12", ...], ... } 로 두면
    #       ros2_node 의 enable_torque_subscribers() 가 자동으로 subscribe 한다.
    # ========================================
    OUTBOUND_TORQUE_TOPIC_TO_JOINTS: dict = {}

    @classmethod
    def get_torque_topics(cls):
        """Real torque 토픽 목록 반환.

        OUTBOUND_TORQUE_TOPIC_TO_JOINTS 가 비어있으면 빈 리스트.
        각 원소는 (topic_full_name, {joint_names, group_name}) 튜플.
        """
        if not cls.OUTBOUND_TORQUE_TOPIC_TO_JOINTS:
            return []

        entries = []
        for group_name, joint_names in cls.OUTBOUND_TORQUE_TOPIC_TO_JOINTS.items():
            # TODO: 실제 토픽 네임스페이스/suffix 확정 시 아래 규칙 수정.
            topic = f"/robot_outbound_data/{group_name}/joint_torque"
            entries.append((topic, {
                'joint_names': joint_names,
                'group_name': group_name,
            }))
        return entries

    @classmethod
    def get_outbound_topics_by_mode(cls, topic_mode):
        """특정 모드에 맞는 14개 outbound 토픽 반환"""
        if topic_mode not in cls.TOPIC_SUFFIXES:
            raise ValueError(f"Invalid topic mode: {topic_mode}")

        suffix = cls.TOPIC_SUFFIXES[topic_mode]
        outbound_topics = {}

        for group_name, joint_names in cls.OUTBOUND_TOPIC_TO_JOINTS.items():
            topic = f"/robot_outbound_data/{group_name}/{suffix}"
            outbound_topics[topic] = {
                'joint_names': joint_names,
                'group_name': group_name,
            }

        return outbound_topics

    @classmethod
    def get_available_topic_modes(cls):
        return list(cls.TOPIC_SUFFIXES.keys())

    @classmethod
    def is_valid_topic_mode(cls, topic_mode):
        return topic_mode in cls.TOPIC_SUFFIXES

    @classmethod
    def get_topic_mode_display_name(cls, topic_mode):
        return cls.TOPIC_MODE_DISPLAY_NAMES.get(topic_mode, topic_mode)
