"""
ALLEX Digital Twin ROS2 노드
"""

from ..config import ROS2Config, ROS2QoS


def create_allex_ros2_node(joint_callback=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import Float64MultiArray

    class ALLEXRos2Node(Node):
        """ALLEX Real2Sim ROS2 노드 — joint_position 토픽 구독 전용"""

        def __init__(self, joint_callback=None):
            super().__init__(ROS2Config.NODE_NAME)
            self._joint_callback = joint_callback
            self._current_topic_mode = ROS2Config.DEFAULT_TOPIC_MODE
            self._current_subscribers = []
            self._desired_subscribers = []
            self._subscriber_enabled = False
            self.get_logger().info(f'ALLEX ROS2 Node Ready (Domain ID: {ROS2Config.DOMAIN_ID})')

        # ========================================
        # 토픽 모드 관리
        # ========================================
        def set_topic_mode(self, topic_mode):
            if not ROS2Config.is_valid_topic_mode(topic_mode):
                self.get_logger().error(f"Invalid topic mode: {topic_mode}")
                return False

            if self._current_topic_mode == topic_mode:
                return True

            was_enabled = self._subscriber_enabled
            if was_enabled:
                self.disable_subscriber()

            old_mode = self._current_topic_mode
            self._current_topic_mode = topic_mode

            if was_enabled:
                success = self.enable_subscriber()
                if not success:
                    self._current_topic_mode = old_mode
                    self.enable_subscriber()
                    return False

            mode_display = ROS2Config.get_topic_mode_display_name(topic_mode)
            self.get_logger().info(f"Topic mode changed to: {mode_display}")
            return True

        def get_current_topic_mode(self):
            return self._current_topic_mode

        # ========================================
        # Subscriber 관리
        # ========================================
        def enable_subscriber(self):
            """28개 Outbound Subscribers 활성화 (14 current + 14 desired)"""
            if self._subscriber_enabled:
                return True

            current_topics = ROS2Config.get_outbound_topics_by_mode(ROS2Config.TOPIC_MODE_CURRENT)
            desired_topics = ROS2Config.get_outbound_topics_by_mode(ROS2Config.TOPIC_MODE_DESIRED)
            qos = self._create_qos_profile()

            self._current_subscribers = []
            for topic, info in current_topics.items():
                sub = self.create_subscription(
                    Float64MultiArray, topic,
                    self._make_callback(info['joint_names'], info['group_name'], mode='current'),
                    qos,
                )
                self._current_subscribers.append(sub)

            self._desired_subscribers = []
            for topic, info in desired_topics.items():
                sub = self.create_subscription(
                    Float64MultiArray, topic,
                    self._make_callback(info['joint_names'], info['group_name'], mode='desired'),
                    qos,
                )
                self._desired_subscribers.append(sub)

            self._subscriber_enabled = True
            total = len(self._current_subscribers) + len(self._desired_subscribers)
            self.get_logger().info(f"{total} Outbound Subscribers ENABLED (Current + Desired)")
            return True

        def disable_subscriber(self):
            for sub in self._current_subscribers:
                self.destroy_subscription(sub)
            self._current_subscribers.clear()

            for sub in self._desired_subscribers:
                self.destroy_subscription(sub)
            self._desired_subscribers.clear()

            self._subscriber_enabled = False
            self.get_logger().info("Outbound Subscribers DISABLED")
            return True

        def toggle_subscriber(self):
            if self._subscriber_enabled:
                return self.disable_subscriber()
            return self.enable_subscriber()

        def _make_callback(self, joint_names, group_name, mode):
            def callback(msg):
                if self._joint_callback:
                    self._joint_callback(msg.data, joint_names, group_name, mode)
            return callback

        # ========================================
        # 헬퍼
        # ========================================
        def _create_qos_profile(self):
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=ROS2QoS.HISTORY_DEPTH,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )

        def is_subscriber_enabled(self):
            return self._subscriber_enabled

        def get_subscriber_count(self):
            return len(self._current_subscribers) + len(self._desired_subscribers)

        def get_status_summary(self):
            mode_display = ROS2Config.get_topic_mode_display_name(self._current_topic_mode)
            total = self.get_subscriber_count()
            return {
                'subscriber': f'ON ({total})' if self._subscriber_enabled else 'OFF',
                'topic_mode': self._current_topic_mode,
                'topic_mode_display': mode_display,
            }

    return ALLEXRos2Node(joint_callback=joint_callback)
