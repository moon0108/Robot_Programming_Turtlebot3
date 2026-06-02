import math
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        self.declare_parameter('waypoint_file', '/root/maps/waypoints.yaml')
        self.waypoint_file = self.get_parameter('waypoint_file').value

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.sub = self.create_subscription(
            String,
            '/target_name',
            self.target_callback,
            10
        )

        self.waypoints = self.load_waypoints()

        self.get_logger().info('Waypoint navigator started.')
        self.get_logger().info(f'Loaded waypoints: {list(self.waypoints.keys())}')

    def load_waypoints(self):
        try:
            with open(self.waypoint_file, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().error(f'Waypoint file not found: {self.waypoint_file}')
            return {}

        if data is None or 'waypoints' not in data:
            self.get_logger().error('No waypoints found in YAML.')
            return {}

        return data['waypoints']

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def target_callback(self, msg):
        target_name = msg.data.strip()
        self.waypoints = self.load_waypoints()

        if target_name not in self.waypoints:
            self.get_logger().error(f'Unknown waypoint: {target_name}')
            self.get_logger().info(f'Available: {list(self.waypoints.keys())}')
            return

        wp = self.waypoints[target_name]
        self.send_goal(target_name, wp)

    def send_goal(self, name, wp):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available.')
            return

        goal_msg = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = 'map'

        pose.pose.position.x = float(wp['x'])
        pose.pose.position.y = float(wp['y'])
        pose.pose.position.z = 0.0

        qz, qw = self.yaw_to_quaternion(float(wp['yaw']))
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal_msg.pose = pose

        self.get_logger().info(
            f'Sending goal [{name}] x={wp["x"]:.3f}, y={wp["y"]:.3f}, yaw={wp["yaw"]:.3f}'
        )

        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            return

        self.get_logger().info('Goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        distance = feedback.distance_remaining
        self.get_logger().info(f'Distance remaining: {distance:.2f} m')

    def result_callback(self, future):
        result = future.result().result
        status = future.result().status

        self.get_logger().info(f'Navigation finished. status={status}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
