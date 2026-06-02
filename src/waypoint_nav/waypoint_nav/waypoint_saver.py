import math
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class WaypointSaver(Node):
    def __init__(self):
        super().__init__('waypoint_saver')

        self.declare_parameter('waypoint_file', '/root/maps/waypoints.yaml')
        self.declare_parameter('waypoint_name', 'station')

        self.waypoint_file = self.get_parameter('waypoint_file').value
        self.waypoint_name = self.get_parameter('waypoint_name').value

        self.sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.get_logger().info('Waypoint saver started.')
        self.get_logger().info(f'Current waypoint name: {self.waypoint_name}')
        self.get_logger().info(f'Save file: {self.waypoint_file}')

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def goal_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.orientation)

        try:
            with open(self.waypoint_file, 'r') as f:
                data = yaml.safe_load(f)
                if data is None:
                    data = {}
        except FileNotFoundError:
            data = {}

        if 'waypoints' not in data:
            data['waypoints'] = {}

        data['waypoints'][self.waypoint_name] = {
            'x': float(x),
            'y': float(y),
            'yaw': float(yaw)
        }

        with open(self.waypoint_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        self.get_logger().info(
            f'Saved [{self.waypoint_name}] x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
