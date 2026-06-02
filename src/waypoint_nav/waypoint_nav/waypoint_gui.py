import math
import os

import yaml

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from python_qt_binding import QtCore, QtWidgets


WAYPOINT_NAMES = ('station', 'container1', 'container2', 'container3')


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qz, qw


class WaypointGuiNode(Node):
    def __init__(self):
        super().__init__('waypoint_gui')

        self.declare_parameter('waypoint_file', '/root/maps/waypoints.yaml')
        self.declare_parameter('pose_topic', '/amcl_pose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('navigate_action', 'navigate_to_pose')
        self.declare_parameter('ignore_goal_yaw', False)

        self.waypoint_file = self.get_parameter('waypoint_file').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        navigate_action = self.get_parameter('navigate_action').value

        self.current_pose = None
        self.current_frame = self.map_frame
        self.status_text = 'Waiting for current pose...'
        self.distance_remaining = None

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.pose_topic,
            self.pose_callback,
            10,
        )
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            navigate_action,
        )
        self.goal_handle = None

        self.get_logger().info('Waypoint GUI node started.')
        self.get_logger().info(f'Pose topic: {self.pose_topic}')
        self.get_logger().info(f'Waypoint file: {self.waypoint_file}')

    def pose_callback(self, msg):
        pose = msg.pose.pose
        self.current_frame = msg.header.frame_id or self.map_frame
        self.current_pose = {
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'yaw': float(quaternion_to_yaw(pose.orientation)),
        }

    def load_data(self):
        try:
            with open(self.waypoint_file, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            data = None

        if not isinstance(data, dict):
            data = {}
        if not isinstance(data.get('waypoints'), dict):
            data['waypoints'] = {}
        return data

    def save_waypoint(self, name):
        if self.current_pose is None:
            self.status_text = (
                f'Cannot save {name}: current pose not received.'
            )
            self.get_logger().warn(self.status_text)
            return False

        data = self.load_data()
        data['waypoints'][name] = dict(self.current_pose)

        directory = os.path.dirname(self.waypoint_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.waypoint_file, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)

        pose = self.current_pose
        self.status_text = (
            f'Saved {name}: x={pose["x"]:.3f}, y={pose["y"]:.3f}, '
            f'yaw={pose["yaw"]:.3f}'
        )
        self.get_logger().info(self.status_text)
        return True

    def get_waypoint(self, name):
        data = self.load_data()
        waypoint = data['waypoints'].get(name)
        if not waypoint:
            self.status_text = f'Waypoint {name} is not saved yet.'
            self.get_logger().warn(self.status_text)
            return None

        try:
            return {
                'x': float(waypoint['x']),
                'y': float(waypoint['y']),
                'yaw': float(waypoint['yaw']),
            }
        except (KeyError, TypeError, ValueError):
            self.status_text = f'Waypoint {name} has invalid data.'
            self.get_logger().error(self.status_text)
            return None

    def navigate_to_waypoint(self, name):
        waypoint = self.get_waypoint(name)
        if waypoint is None:
            return False

        if not self.action_client.wait_for_server(timeout_sec=0.5):
            self.status_text = (
                'Nav2 navigate_to_pose action server is not available.'
            )
            self.get_logger().error(self.status_text)
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.pose.position.x = waypoint['x']
        goal_msg.pose.pose.position.y = waypoint['y']
        goal_msg.pose.pose.position.z = 0.0

        goal_yaw = waypoint['yaw']
        ignore_goal_yaw = bool(self.get_parameter('ignore_goal_yaw').value)
        if ignore_goal_yaw and self.current_pose is not None:
            goal_yaw = self.current_pose['yaw']

        qz, qw = yaw_to_quaternion(goal_yaw)
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.distance_remaining = None
        self.status_text = f'Sending goal: {name}'
        self.get_logger().info(
            f'Sending goal [{name}] x={waypoint["x"]:.3f}, '
            f'y={waypoint["y"]:.3f}, yaw={goal_yaw:.3f}'
        )

        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        future.add_done_callback(
            lambda done: self.goal_response_callback(done, name)
        )
        return True

    def cancel_navigation(self):
        if self.goal_handle is None:
            self.status_text = 'No active navigation goal.'
            return

        future = self.goal_handle.cancel_goal_async()
        future.add_done_callback(self.cancel_callback)
        self.status_text = 'Cancel requested.'

    def goal_response_callback(self, future, name):
        try:
            self.goal_handle = future.result()
        except Exception as exc:
            self.status_text = f'Failed to send goal: {exc}'
            self.get_logger().error(self.status_text)
            return

        if not self.goal_handle.accepted:
            self.status_text = f'Goal rejected: {name}'
            self.get_logger().error(self.status_text)
            self.goal_handle = None
            return

        self.status_text = f'Goal accepted: {name}'
        self.get_logger().info(self.status_text)
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done: self.result_callback(done, name)
        )

    def feedback_callback(self, feedback_msg):
        self.distance_remaining = feedback_msg.feedback.distance_remaining

    def result_callback(self, future, name):
        try:
            result = future.result()
        except Exception as exc:
            self.status_text = f'Navigation failed: {exc}'
            self.get_logger().error(self.status_text)
            self.goal_handle = None
            return

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.status_text = f'Navigation succeeded: {name}'
        elif result.status == GoalStatus.STATUS_CANCELED:
            self.status_text = f'Navigation canceled: {name}'
        else:
            self.status_text = (
                f'Navigation failed: {name}, status={result.status}'
            )
        self.get_logger().info(self.status_text)
        self.goal_handle = None
        self.distance_remaining = None

    def cancel_callback(self, future):
        try:
            cancel_response = future.result()
            count = len(cancel_response.goals_canceling)
        except Exception as exc:
            self.status_text = f'Cancel failed: {exc}'
            self.get_logger().error(self.status_text)
            return

        self.status_text = (
            'Navigation cancel accepted.' if count else 'No goal canceled.'
        )
        self.get_logger().info(self.status_text)
        self.goal_handle = None
        self.distance_remaining = None


class WaypointWindow(QtWidgets.QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle('TurtleBot3 Burger Waypoints')
        self.setMinimumSize(720, 360)

        main_layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('TurtleBot3 Burger Waypoints')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setObjectName('titleLabel')
        main_layout.addWidget(title)

        columns = QtWidgets.QHBoxLayout()
        main_layout.addLayout(columns, 1)

        save_group = QtWidgets.QGroupBox('Save Current Position')
        save_layout = QtWidgets.QVBoxLayout(save_group)
        self.pose_label = QtWidgets.QLabel(
            'x: ---.---   y: ---.---   yaw: ---.---'
        )
        self.pose_label.setAlignment(QtCore.Qt.AlignCenter)
        save_layout.addWidget(self.pose_label)
        for name in WAYPOINT_NAMES:
            button = QtWidgets.QPushButton(f'Save {name}')
            button.setMinimumHeight(44)
            button.clicked.connect(
                lambda checked=False, n=name: self.save_waypoint(n)
            )
            save_layout.addWidget(button)
        save_layout.addStretch(1)

        move_group = QtWidgets.QGroupBox('Move To Waypoint')
        move_layout = QtWidgets.QVBoxLayout(move_group)
        for name in WAYPOINT_NAMES:
            button = QtWidgets.QPushButton(f'Go {name}')
            button.setMinimumHeight(44)
            button.clicked.connect(
                lambda checked=False, n=name: self.go_waypoint(n)
            )
            move_layout.addWidget(button)
        cancel_button = QtWidgets.QPushButton('Cancel Navigation')
        cancel_button.setMinimumHeight(44)
        cancel_button.clicked.connect(self.node.cancel_navigation)
        move_layout.addWidget(cancel_button)
        move_layout.addStretch(1)

        columns.addWidget(save_group, 1)
        columns.addWidget(move_group, 1)

        self.status_label = QtWidgets.QLabel('')
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        self.setStyleSheet("""
            QWidget {
                font-size: 16px;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: 700;
                padding: 10px;
            }
            QGroupBox {
                font-weight: 700;
                border: 1px solid #9aa0a6;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                font-weight: 600;
                padding: 8px 12px;
            }
        """)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(100)

    def save_waypoint(self, name):
        self.node.save_waypoint(name)
        self.refresh()

    def go_waypoint(self, name):
        self.node.navigate_to_waypoint(name)
        self.refresh()

    def refresh(self):
        pose = self.node.current_pose
        if pose is None:
            self.pose_label.setText('x: ---.---   y: ---.---   yaw: ---.---')
        else:
            self.pose_label.setText(
                f'x: {pose["x"]:.3f}   y: {pose["y"]:.3f}   '
                f'yaw: {pose["yaw"]:.3f}'
            )

        status = self.node.status_text
        if self.node.distance_remaining is not None:
            status = (
                f'{status}   Distance remaining: '
                f'{self.node.distance_remaining:.2f} m'
            )
        self.status_label.setText(status)


def main(args=None):
    rclpy.init(args=args)
    app = QtWidgets.QApplication([])
    node = WaypointGuiNode()
    window = WaypointWindow(node)
    window.show()

    spin_timer = QtCore.QTimer()
    spin_timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    spin_timer.start(20)

    try:
        app.exec_()
    finally:
        spin_timer.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
