"""
ALLEX Digital Twin ROS2 통합 관리자
"""

import os
import threading

from .ros2_node import create_allex_ros2_node
from ..config import ROS2Config
from ..core.joint_controller import ALLEXJointController


class ROS2IntegratedManager:
    """ROS2 통신을 통합 관리하는 클래스"""

    def __init__(self, scenario_ref=None):
        self._ros2_node = None
        self._executor = None
        self._ros_thread = None
        self._joint_controller = None
        self._initialized = False
        self._scenario_ref = scenario_ref
        self._topic_mode = ROS2Config.DEFAULT_TOPIC_MODE

    # ========================================
    # 초기화 및 정리
    # ========================================
    def initialize(self):
        if self._initialized:
            return True

        self.cleanup()

        # ROS2 환경 설정 (bridge 활성화 이전에)
        os.environ['ROS_DOMAIN_ID'] = str(ROS2Config.DOMAIN_ID)
        os.environ['RMW_IMPLEMENTATION'] = ROS2Config.RMW_IMPLEMENTATION

        # Enable Isaac Sim's ROS2 DDS bridge extension; rclpy itself is provided
        # by the system ROS2 install (e.g. /opt/ros/jazzy). Isaac Sim 6.0.0 does
        # not bundle rclpy — do not purge system paths here.
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension(ROS2Config.BRIDGE_EXTENSION)

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        print(f"[ALLEX][ROS2] using rclpy from: {getattr(rclpy, '__file__', '<unknown>')}")

        if not rclpy.ok():
            rclpy.init()

        # Scenario의 JointController 공유
        if self._scenario_ref and hasattr(self._scenario_ref, '_joint_controller'):
            self._joint_controller = self._scenario_ref._joint_controller
        else:
            self._joint_controller = ALLEXJointController()

        # ROS2 노드 생성 — joint_controller에 직접 콜백 연결
        self._ros2_node = create_allex_ros2_node(
            joint_callback=self._joint_controller.on_joint_data_received,
            torque_callback=self._joint_controller.on_torque_data_received,
        )

        if self._ros2_node:
            self._ros2_node.set_topic_mode(self._topic_mode)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._ros2_node)
        self._start_executor_thread()

        self._initialized = True
        print(f"ROS2 Manager initialized (Domain ID: {ROS2Config.DOMAIN_ID})")
        return True

    def cleanup(self):
        if self._executor:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._ros2_node:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                pass
            self._ros2_node = None

        self._joint_controller = None

        if self._ros_thread and self._ros_thread.is_alive():
            try:
                self._ros_thread.join(timeout=1.0)
            except Exception:
                pass
        self._ros_thread = None

        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

        self._initialized = False

    def _start_executor_thread(self):
        def safe_spin():
            try:
                self._executor.spin()
            except Exception as e:
                print(f"ROS2 Executor error: {e}")

        self._ros_thread = threading.Thread(target=safe_spin, daemon=ROS2Config.THREAD_DAEMON)
        self._ros_thread.start()

    # ========================================
    # 토픽 모드 관리
    # ========================================
    def toggle_topic_mode(self):
        if not self._initialized or not self._ros2_node:
            return False, "Topic Mode: ERROR (No Manager)"

        if self._topic_mode == ROS2Config.TOPIC_MODE_CURRENT:
            new_mode = ROS2Config.TOPIC_MODE_DESIRED
        else:
            new_mode = ROS2Config.TOPIC_MODE_CURRENT

        success = self.set_topic_mode(new_mode)
        if success:
            mode_display = ROS2Config.get_topic_mode_display_name(new_mode)
            return True, f"Topic Mode: {mode_display}"
        return False, "Topic Mode: FAILED"

    def set_topic_mode(self, topic_mode):
        if not self._initialized or not self._ros2_node:
            return False

        if not ROS2Config.is_valid_topic_mode(topic_mode):
            return False

        success = self._ros2_node.set_topic_mode(topic_mode)
        if success:
            self._topic_mode = topic_mode
            if self._joint_controller:
                self._joint_controller.set_topic_mode(topic_mode)
        return success

    def get_current_topic_mode(self):
        if self._initialized and self._ros2_node:
            node_mode = self._ros2_node.get_current_topic_mode()
            self._topic_mode = node_mode
            return node_mode
        return self._topic_mode

    # ========================================
    # Subscriber 제어
    # ========================================
    def toggle_subscriber(self):
        if not self._initialized or not self._ros2_node:
            return False, "Subscriber: ERROR (No Manager)"

        success = self._ros2_node.toggle_subscriber()

        if self._ros2_node.is_subscriber_enabled():
            current_mode = self.get_current_topic_mode()
            mode_display = ROS2Config.get_topic_mode_display_name(current_mode)
            status = f"Subscriber: ON ({mode_display})"
            if self._joint_controller:
                self._joint_controller.set_ros2_subscriber_status(True)
        else:
            status = "Subscriber: OFF"
            if self._joint_controller:
                self._joint_controller.set_ros2_subscriber_status(False)

        return success, status

    # ========================================
    # 상태 확인
    # ========================================
    def is_initialized(self):
        return self._initialized

    def get_joint_controller(self):
        return self._joint_controller

    def get_status_summary(self):
        if not self._initialized or not self._ros2_node:
            return {
                'initialized': False,
                'subscriber': 'OFF',
                'topic_mode': self._topic_mode,
                'topic_mode_display': ROS2Config.get_topic_mode_display_name(self._topic_mode),
            }

        status = self._ros2_node.get_status_summary()
        status['initialized'] = True
        return status
