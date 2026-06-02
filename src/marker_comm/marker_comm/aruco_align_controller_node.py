import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ArucoAlignControllerNode(Node):
    def __init__(self):
        super().__init__('aruco_align_controller')

        self.declare_parameter('align_error_topic', '/aruco/align_error')
        self.declare_parameter('active_topic', '/aruco_align/active')
        self.declare_parameter('done_topic', '/aruco_align/done')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('target_distance', 0.2)
        self.declare_parameter('align_start_distance', 1.0)
        self.declare_parameter('z_diff_threshold', 0.01)
        self.declare_parameter('center_x_threshold', 0.02)
        self.declare_parameter('center_x_offset', -0.01)
        self.declare_parameter('distance_threshold', 0.04)
        self.declare_parameter('odom_step_distance_threshold', 0.05)
        self.declare_parameter('min_valid_marker_distance', 0.05)
        self.declare_parameter('z_diff_gain', 1.8)
        self.declare_parameter('max_yaw_align_speed', 0.08)
        self.declare_parameter('center_x_gain', 1.2)
        self.declare_parameter('distance_gain', 0.25)   
        self.declare_parameter('approach_linear_speed', 0.04)
        self.declare_parameter('single_marker_x_gain', 0.9)
        self.declare_parameter('single_marker_search_angular_speed', 0.14)
        self.declare_parameter('single_marker_hold_distance', 0.2)
        self.declare_parameter('left_marker_id', 0)
        self.declare_parameter('right_marker_id', 1)
        self.declare_parameter('enable_odom_steps', True)
        self.declare_parameter('odom_turn_gain', 1.6)
        self.declare_parameter('odom_turn_tolerance', 0.04)
        self.declare_parameter('odom_drive_speed', 0.1)
        self.declare_parameter('odom_drive_tolerance', 0.015)
        self.declare_parameter('min_odom_drive_speed', 0.025)
        self.declare_parameter('odom_step_distance_scale', 1.0)
        self.declare_parameter('min_odom_step_distance', 0.005)
        self.declare_parameter('max_odom_step_distance', 0.8)
        self.declare_parameter('max_odom_step_heading', 0.80)
        self.declare_parameter('lateral_odom_step_heading', 0.75)
        self.declare_parameter('reverse_step_distance', 0.08)
        self.declare_parameter('search_linear_speed', -0.025)
        self.declare_parameter('search_angular_speed', 0.0)
        self.declare_parameter('linear_direction', 1.0)
        self.declare_parameter('angular_direction', 1.0)
        self.declare_parameter('max_linear_speed', 0.08)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('marker_timeout_sec', 0.5)
        self.declare_parameter('zero_cmd_publish_count', 5)
        self.declare_parameter('zero_cmd_publish_period_sec', 0.02)
        self.declare_parameter('start_active', False)
        self.declare_parameter('enable_motion', False)

        align_error_topic = self.get_parameter('align_error_topic').value
        active_topic = self.get_parameter('active_topic').value
        done_topic = self.get_parameter('done_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        control_rate_hz = float(self.get_parameter('control_rate_hz').value)

        self.target_distance = float(self.get_parameter('target_distance').value)
        self.align_start_distance = float(self.get_parameter('align_start_distance').value)
        self.z_diff_threshold = float(self.get_parameter('z_diff_threshold').value)
        self.center_x_threshold = float(self.get_parameter('center_x_threshold').value)
        self.center_x_offset = float(self.get_parameter('center_x_offset').value)
        self.distance_threshold = float(self.get_parameter('distance_threshold').value)
        self.odom_step_distance_threshold = float(
            self.get_parameter('odom_step_distance_threshold').value
        )
        self.min_valid_marker_distance = float(self.get_parameter('min_valid_marker_distance').value)
        self.z_diff_gain = float(self.get_parameter('z_diff_gain').value)
        self.max_yaw_align_speed = float(self.get_parameter('max_yaw_align_speed').value)
        self.center_x_gain = float(self.get_parameter('center_x_gain').value)
        self.distance_gain = float(self.get_parameter('distance_gain').value)
        self.approach_linear_speed = float(self.get_parameter('approach_linear_speed').value)
        self.single_marker_x_gain = float(self.get_parameter('single_marker_x_gain').value)
        self.single_marker_search_angular_speed = float(
            self.get_parameter('single_marker_search_angular_speed').value
        )
        self.single_marker_hold_distance = float(
            self.get_parameter('single_marker_hold_distance').value
        )
        self.left_marker_id = int(self.get_parameter('left_marker_id').value)
        self.right_marker_id = int(self.get_parameter('right_marker_id').value)
        self.enable_odom_steps = bool(self.get_parameter('enable_odom_steps').value)
        self.odom_turn_gain = float(self.get_parameter('odom_turn_gain').value)
        self.odom_turn_tolerance = float(self.get_parameter('odom_turn_tolerance').value)
        self.odom_drive_speed = float(self.get_parameter('odom_drive_speed').value)
        self.odom_drive_tolerance = float(self.get_parameter('odom_drive_tolerance').value)
        self.min_odom_drive_speed = float(self.get_parameter('min_odom_drive_speed').value)
        self.odom_step_distance_scale = float(
            self.get_parameter('odom_step_distance_scale').value
        )
        self.min_odom_step_distance = float(self.get_parameter('min_odom_step_distance').value)
        self.max_odom_step_distance = float(self.get_parameter('max_odom_step_distance').value)
        self.max_odom_step_heading = float(self.get_parameter('max_odom_step_heading').value)
        self.lateral_odom_step_heading = float(
            self.get_parameter('lateral_odom_step_heading').value
        )
        self.reverse_step_distance = float(self.get_parameter('reverse_step_distance').value)
        self.search_linear_speed = float(self.get_parameter('search_linear_speed').value)
        self.search_angular_speed = float(self.get_parameter('search_angular_speed').value)
        self.linear_direction = float(self.get_parameter('linear_direction').value)
        self.angular_direction = float(self.get_parameter('angular_direction').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.marker_timeout_sec = float(self.get_parameter('marker_timeout_sec').value)
        self.zero_cmd_publish_count = int(
            self.get_parameter('zero_cmd_publish_count').value
        )
        self.zero_cmd_publish_period_sec = float(
            self.get_parameter('zero_cmd_publish_period_sec').value
        )
        self.alignment_active = bool(self.get_parameter('start_active').value)
        self.enable_motion = bool(self.get_parameter('enable_motion').value)

        self.latest_error = None
        self.latest_error_time = None
        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None
        self.odom_step_state = 'vision'
        self.odom_step = {}
        self.last_mode = None
        self.alignment_done = False
        self.done_published = False

        self.error_sub = self.create_subscription(
            Float32MultiArray,
            align_error_topic,
            self.align_error_callback,
            10,
        )
        self.active_sub = self.create_subscription(
            Bool,
            active_topic,
            self.active_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.done_pub = self.create_publisher(Bool, done_topic, 10)
        self.timer = self.create_timer(1.0 / control_rate_hz, self.control_loop)

        self.get_logger().info(f'Subscribed align error: {align_error_topic}')
        self.get_logger().info(f'Subscribed active command: {active_topic}')
        self.get_logger().info(f'Subscribed odom: {odom_topic}')
        self.get_logger().info(f'Publishing cmd_vel: {cmd_vel_topic}')
        self.get_logger().info(f'Publishing alignment done: {done_topic}')
        self.get_logger().info(
            f'target_distance={self.target_distance:.2f} m, '
            f'align_start_distance={self.align_start_distance:.2f} m, '
            f'start_active={self.alignment_active}, '
            f'enable_motion={self.enable_motion}'
        )
        self.get_logger().info(f'min_valid_marker_distance={self.min_valid_marker_distance:.2f} m')
        self.get_logger().info(
            f'linear_direction={self.linear_direction:+.1f}, '
            f'angular_direction={self.angular_direction:+.1f}'
        )
        if not self.enable_motion:
            self.get_logger().warn('Motion is disabled. Set -p enable_motion:=true to publish movement.')

    def odom_callback(self, msg):
        pose = msg.pose.pose
        self.odom_x = pose.position.x
        self.odom_y = pose.position.y
        self.odom_yaw = yaw_from_quaternion(pose.orientation)

    def active_callback(self, msg):
        was_active = self.alignment_active
        self.alignment_active = bool(msg.data)
        if not self.alignment_active:
            self.odom_step_state = 'vision'
            self.odom_step = {}
            self.alignment_done = False
            self.done_published = False
            if was_active:
                self.publish_zero_cmd_vel_burst()
        elif not was_active:
            self.alignment_done = False
            self.done_published = False
        if self.alignment_active != was_active:
            state = 'active' if self.alignment_active else 'inactive'
            self.get_logger().info(f'Alignment {state}.')

    def align_error_callback(self, msg):
        if len(msg.data) < 8:
            self.get_logger().warn('Ignoring old align_error format. Rebuild and restart aruco_pose.')
            return

        raw_center_x = float(msg.data[0])
        raw_single_marker_x = float(msg.data[6])
        self.latest_error = {
            'center_x': raw_center_x - self.center_x_offset,
            'center_z': float(msg.data[1]),
            'z_diff': float(msg.data[2]),
            'both_visible': float(msg.data[3]) >= 0.5,
            'visible_count': int(round(float(msg.data[4]))),
            'single_marker_id': int(round(float(msg.data[5]))),
            'single_marker_x': raw_single_marker_x - self.center_x_offset,
            'single_marker_z': float(msg.data[7]),
        }
        if not self.is_error_valid(self.latest_error):
            self.get_logger().warn(
                'Invalid align_error received; treating marker as lost. '
                f'data={list(msg.data)}',
                throttle_duration_sec=1.0,
            )
            self.latest_error['both_visible'] = False
            self.latest_error['visible_count'] = 0
        self.latest_error_time = self.get_clock().now()

    def is_error_valid(self, error):
        if error['both_visible'] and error['center_z'] < self.min_valid_marker_distance:
            return False
        if error['visible_count'] == 1 and error['single_marker_z'] < self.min_valid_marker_distance:
            return False
        return True

    def control_loop(self):
        twist = Twist()
        mode = 'stop'

        if not self.alignment_active:
            mode = 'inactive'
            self.log_mode_change(mode, twist)
            return
        elif self.alignment_done:
            mode = 'aligned_done'
            self.log_mode_change(mode, twist)
            return
        elif self.odom_step_state != 'vision':
            mode, twist = self.odom_step_control()
        elif self.is_error_stale():
            mode = 'lost'
            twist.linear.x = self.search_linear_speed
            twist.angular.z = self.search_angular_speed
        elif self.latest_error['both_visible']:
            mode, twist = self.two_marker_control(self.latest_error)
        elif self.latest_error['visible_count'] == 1:
            mode, twist = self.single_marker_control(self.latest_error)
        else:
            mode = 'search'
            twist.linear.x = self.search_linear_speed
            twist.angular.z = self.search_angular_speed

        if mode == 'aligned':
            self.alignment_done = True
            self.publish_done_once()
            self.publish_zero_cmd_vel_burst()
            self.log_mode_change(mode, Twist())
            return

        twist.linear.x *= self.linear_direction
        twist.angular.z *= self.angular_direction
        twist.linear.x = clamp(twist.linear.x, -self.max_linear_speed, self.max_linear_speed)
        twist.angular.z = clamp(twist.angular.z, -self.max_angular_speed, self.max_angular_speed)

        if not self.enable_motion:
            twist = Twist()
            self.log_mode_change(mode, twist)
            return

        self.cmd_pub.publish(twist)
        self.log_mode_change(mode, twist)

    def publish_zero_cmd_vel_burst(self):
        if not rclpy.ok():
            return
        count = max(1, self.zero_cmd_publish_count)
        period = max(0.0, self.zero_cmd_publish_period_sec)
        for index in range(count):
            try:
                self.cmd_pub.publish(Twist())
            except RCLError:
                return
            if period > 0.0 and index < count - 1:
                time.sleep(period)

    def publish_done_once(self):
        if self.done_published:
            return
        self.done_published = True
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)
        self.get_logger().info('Alignment done published.')

    def has_odom(self):
        return self.odom_x is not None and self.odom_y is not None and self.odom_yaw is not None

    def is_error_stale(self):
        if self.latest_error is None or self.latest_error_time is None:
            return True
        age = (self.get_clock().now() - self.latest_error_time).nanoseconds / 1e9
        return age > self.marker_timeout_sec

    def is_two_marker_aligned(self, error):
        if not error['both_visible']:
            return False
        distance_error = error['center_z'] - self.target_distance
        return (
            error['center_z'] <= self.align_start_distance
            and abs(error['z_diff']) <= self.z_diff_threshold
            and abs(error['center_x']) <= self.center_x_threshold
            and abs(distance_error) <= self.distance_threshold
        )

    def two_marker_control(self, error):
        twist = Twist()
        distance_error = error['center_z'] - self.target_distance

        if error['center_z'] > self.align_start_distance:
            twist.linear.x = self.approach_linear_speed
            twist.angular.z = -self.center_x_gain * error['center_x']
            return 'two_marker_approach', twist

        if distance_error < -self.distance_threshold and self.enable_odom_steps and self.has_odom():
            self.start_reverse_step()
            return 'odom_reverse_start', twist

        if abs(error['z_diff']) > self.z_diff_threshold:
            twist.angular.z = clamp(
                self.z_diff_gain * error['z_diff'],
                -self.max_yaw_align_speed,
                self.max_yaw_align_speed,
            )
            return 'two_marker_yaw', twist

        if abs(error['center_x']) > self.center_x_threshold:
            if self.enable_odom_steps and self.has_odom():
                self.start_lateral_odom_step(error)
                return 'odom_lateral_start', twist
            return 'two_marker_center_wait_odom', twist

        if (
            self.enable_odom_steps
            and self.has_odom()
            and abs(distance_error) > self.odom_step_distance_threshold
        ):
            self.start_odom_alignment_step(error)
            return 'odom_step_start', twist

        if abs(distance_error) > self.distance_threshold:
            twist.linear.x = self.distance_gain * distance_error
            return 'two_marker_distance', twist

        return 'aligned', twist

    def start_odom_alignment_step(self, error):
        distance_error = error['center_z'] - self.target_distance
        if (
            abs(error['center_x']) > self.center_x_threshold
            and abs(distance_error) <= self.distance_threshold
        ):
            self.start_lateral_odom_step(error)
            return

        forward_error = max(0.0, error['center_z'] - self.target_distance)
        heading = -math.atan2(error['center_x'], max(forward_error, 0.05))
        heading = clamp(heading, -self.max_odom_step_heading, self.max_odom_step_heading)
        step_distance = math.hypot(error['center_x'], forward_error)
        step_distance *= self.odom_step_distance_scale
        step_distance = clamp(
            step_distance,
            self.min_odom_step_distance,
            self.max_odom_step_distance,
        )
        self.odom_step = {
            'start_x': self.odom_x,
            'start_y': self.odom_y,
            'start_yaw': self.odom_yaw,
            'move_yaw': normalize_angle(self.odom_yaw + heading),
            'return_yaw': self.odom_yaw,
            'distance': step_distance,
            'step_kind': 'alignment',
        }
        self.odom_step_state = 'turn_to_move'

    def start_lateral_odom_step(self, error):
        heading_size = clamp(
            abs(self.lateral_odom_step_heading),
            self.odom_turn_tolerance,
            math.pi * 0.5,
        )
        heading = -math.copysign(heading_size, error['center_x'])
        step_distance = abs(error['center_x']) / max(math.sin(heading_size), 0.1)
        step_distance *= self.odom_step_distance_scale
        step_distance = clamp(
            step_distance,
            self.min_odom_step_distance,
            self.max_odom_step_distance,
        )
        self.odom_step = {
            'start_x': self.odom_x,
            'start_y': self.odom_y,
            'start_yaw': self.odom_yaw,
            'move_yaw': normalize_angle(self.odom_yaw + heading),
            'return_yaw': self.odom_yaw,
            'distance': step_distance,
            'step_kind': 'lateral',
        }
        self.odom_step_state = 'turn_to_move'

    def odom_step_control(self):
        twist = Twist()
        if not self.has_odom():
            self.odom_step_state = 'vision'
            return 'odom_wait', twist

        if (
            self.latest_error is not None
            and not self.is_error_stale()
            and self.is_two_marker_aligned(self.latest_error)
        ):
            self.odom_step_state = 'vision'
            self.odom_step = {}
            return 'aligned', twist

        if self.odom_step_state == 'turn_to_move':
            yaw_error = normalize_angle(self.odom_step['move_yaw'] - self.odom_yaw)
            if abs(yaw_error) <= self.odom_turn_tolerance:
                self.odom_step['start_x'] = self.odom_x
                self.odom_step['start_y'] = self.odom_y
                self.odom_step_state = 'drive_step'
                return 'odom_drive_step', twist
            twist.angular.z = self.odom_turn_gain * yaw_error
            step_kind = self.odom_step.get('step_kind', 'alignment')
            return f'odom_turn_to_{step_kind}', twist

        if self.odom_step_state == 'drive_step':
            traveled = math.hypot(
                self.odom_x - self.odom_step['start_x'],
                self.odom_y - self.odom_step['start_y'],
            )
            remaining = self.odom_step['distance'] - traveled
            if remaining <= self.odom_drive_tolerance:
                self.odom_step_state = self.odom_step.get('after_drive_state', 'turn_back')
                return self.odom_step_state, twist
            drive_direction = self.odom_step.get('drive_direction', 1.0)
            drive_speed = clamp(
                remaining,
                self.min_odom_drive_speed,
                self.odom_drive_speed,
            )
            twist.linear.x = drive_direction * drive_speed
            return 'odom_drive_step', twist

        if self.odom_step_state == 'turn_back':
            yaw_error = normalize_angle(self.odom_step['return_yaw'] - self.odom_yaw)
            if abs(yaw_error) <= self.odom_turn_tolerance:
                self.odom_step_state = 'vision'
                self.odom_step = {}
                return 'odom_verify', twist
            twist.angular.z = self.odom_turn_gain * yaw_error
            return 'odom_turn_back', twist

        if self.odom_step_state == 'single_marker_search':
            if (
                self.latest_error is not None
                and not self.is_error_stale()
                and self.latest_error['both_visible']
            ):
                self.odom_step_state = 'vision'
                self.odom_step = {}
                return 'single_marker_recovered', twist

            marker_id = self.odom_step.get('single_marker_id', -1)
            if marker_id == self.left_marker_id:
                twist.angular.z = self.single_marker_search_angular_speed
            elif marker_id == self.right_marker_id:
                twist.angular.z = -self.single_marker_search_angular_speed
            else:
                twist.angular.z = self.search_angular_speed
            return f'single_marker_search_{marker_id}', twist

        if self.odom_step_state == 'single_marker_reverse_search':
            if (
                self.latest_error is not None
                and not self.is_error_stale()
                and self.latest_error['both_visible']
            ):
                self.odom_step_state = 'vision'
                self.odom_step = {}
                return 'single_marker_recovered', twist

            traveled = math.hypot(
                self.odom_x - self.odom_step['start_x'],
                self.odom_y - self.odom_step['start_y'],
            )
            marker_id = self.odom_step.get('single_marker_id', -1)
            if traveled >= self.odom_step['distance']:
                self.odom_step_state = 'single_marker_search'
                return f'single_marker_search_{marker_id}', twist

            twist.linear.x = -self.odom_drive_speed
            return f'single_marker_reverse_straight_{marker_id}', twist

        self.odom_step_state = 'vision'
        self.odom_step = {}
        return 'odom_reset', twist

    def start_reverse_step(self, after_drive_state='vision', marker_id=None):
        step_distance = clamp(
            self.reverse_step_distance,
            self.min_odom_step_distance,
            self.max_odom_step_distance,
        )
        self.odom_step = {
            'start_x': self.odom_x,
            'start_y': self.odom_y,
            'move_yaw': self.odom_yaw,
            'distance': step_distance,
            'drive_direction': -1.0,
            'after_drive_state': after_drive_state,
        }
        if marker_id is not None:
            self.odom_step['single_marker_id'] = marker_id
        self.odom_step_state = 'drive_step'

    def single_marker_control(self, error):
        twist = Twist()
        marker_id = error['single_marker_id']

        if error['single_marker_z'] <= self.single_marker_hold_distance:
            if self.enable_odom_steps and self.has_odom():
                self.start_single_marker_reverse_search_step(marker_id)
                return f'single_marker_recover_start_{marker_id}', twist
            return f'single_marker_hold_{marker_id}', twist

        if marker_id == self.left_marker_id:
            twist.angular.z = -self.single_marker_search_angular_speed
        elif marker_id == self.right_marker_id:
            twist.angular.z = self.single_marker_search_angular_speed
        else:
            twist.angular.z = -self.single_marker_x_gain * error['single_marker_x']

        return f'single_marker_{marker_id}', twist

    def start_single_marker_reverse_search_step(self, marker_id):
        step_distance = clamp(
            self.reverse_step_distance,
            self.min_odom_step_distance,
            self.max_odom_step_distance,
        )
        self.odom_step = {
            'start_x': self.odom_x,
            'start_y': self.odom_y,
            'distance': step_distance,
            'single_marker_id': marker_id,
        }
        self.odom_step_state = 'single_marker_reverse_search'

    def log_mode_change(self, mode, twist):
        if mode == self.last_mode:
            return
        self.last_mode = mode
        self.get_logger().info(
            f'mode={mode}, linear.x={twist.linear.x:+.3f}, angular.z={twist.angular.z:+.3f}'
        )


def main(args=None):
    node = None
    rclpy.init(args=args)
    try:
        node = ArucoAlignControllerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        if node is not None:
            try:
                node.publish_zero_cmd_vel_burst()
                node.destroy_node()
            except RCLError:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
