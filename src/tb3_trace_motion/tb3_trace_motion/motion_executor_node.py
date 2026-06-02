import copy
import json
import math
import os
import threading
import time
from typing import Dict, Optional

import yaml

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


DEFAULT_PATHS = {
    'paths': {
        'station_to_container1': [
            {'type': 'move', 'distance': 0.30, 'speed': 0.05},
            {'type': 'rotate', 'angle': 1.57, 'angular_speed': 0.25},
        ],
        'station_to_container2': [
            {'type': 'wait', 'duration': 0.0},
        ],
        'station_to_container3': [
            {'type': 'wait', 'duration': 0.0},
        ],
        'container1_to_station': [
            {'type': 'rotate', 'angle': -1.57, 'angular_speed': 0.25},
            {'type': 'move', 'distance': 0.30, 'speed': 0.05},
        ],
        'container2_to_station': [
            {'type': 'wait', 'duration': 0.0},
        ],
        'container3_to_station': [
            {'type': 'wait', 'duration': 0.0},
        ],
        'test_forward_back': [
            {'type': 'move', 'distance': 0.20, 'speed': 0.05},
            {'type': 'wait', 'duration': 1.0},
            {'type': 'move', 'distance': -0.20, 'speed': 0.05},
        ],
    }
}


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MotionExecutor(Node):
    def __init__(self):
        super().__init__('tb3_motion_executor')

        self.declare_parameter(
            'motion_file',
            '/root/maps/tb3_motion_paths.yaml',
        )
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('command_topic', '/tb3_motion/command')
        self.declare_parameter('status_topic', '/tb3_motion/status')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('default_linear_speed', 0.05)
        self.declare_parameter('default_angular_speed', 0.25)
        self.declare_parameter('distance_tolerance_m', 0.01)
        self.declare_parameter('angle_tolerance_rad', 0.02)

        self.motion_file = self.get_parameter('motion_file').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        command_topic = self.get_parameter('command_topic').value
        status_topic = self.get_parameter('status_topic').value

        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.running = False
        self.active_path_name = None
        self.active_thread = None
        self.pose: Optional[Dict[str, float]] = None
        self.previous_yaw = None
        self.yaw_unwrapped = 0.0
        self.paths = {}

        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            20,
        )
        self.create_subscription(
            String,
            command_topic,
            self.command_callback,
            10,
        )

        self.create_service(
            Trigger,
            '/tb3_motion/stop',
            self.stop_callback,
        )
        self.create_service(
            Trigger,
            '/tb3_motion/reload',
            self.reload_callback,
        )

        self._ensure_motion_file()
        self.load_paths()
        self.publish_status('IDLE', 'motion executor ready')

        self.get_logger().info(f'Motion file: {self.motion_file}')
        self.get_logger().info(f'Odom topic: {odom_topic}')
        self.get_logger().info(f'Command topic: {command_topic}')

    def _ensure_motion_file(self) -> None:
        if os.path.exists(self.motion_file):
            return

        directory = os.path.dirname(self.motion_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.motion_file, 'w', encoding='utf-8') as yaml_file:
            yaml.safe_dump(DEFAULT_PATHS, yaml_file, sort_keys=False)

    def load_paths(self) -> bool:
        try:
            with open(self.motion_file, 'r', encoding='utf-8') as yaml_file:
                data = yaml.safe_load(yaml_file) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(f'Failed to load motion file: {exc}')
            return False

        paths = data.get('paths', {})
        if not isinstance(paths, dict):
            self.get_logger().error('Motion file must contain a "paths" map')
            return False

        with self.lock:
            self.paths = paths
        self.get_logger().info(f'Loaded motion paths: {list(paths.keys())}')
        return True

    def odom_callback(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        yaw = quaternion_to_yaw(pose.orientation)

        with self.lock:
            if self.previous_yaw is None:
                self.yaw_unwrapped = yaw
            else:
                self.yaw_unwrapped += normalize_angle(yaw - self.previous_yaw)

            self.previous_yaw = yaw
            self.pose = {
                'x': float(pose.position.x),
                'y': float(pose.position.y),
                'yaw': yaw,
                'yaw_unwrapped': self.yaw_unwrapped,
                'time': time.monotonic(),
            }

    def command_callback(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {'command': 'run', 'path': text}

        command = str(payload.get('command', 'run')).strip().lower()
        if command == 'run':
            path_name = str(payload.get('path', '')).strip()
            steps = payload.get('steps')
            if isinstance(steps, list):
                self.start_steps(path_name or 'inline_path', steps)
            else:
                self.start_path(path_name)
        elif command == 'stop':
            self.request_stop('stop command received')
        elif command == 'reload':
            self.load_paths()
            self.publish_status('IDLE', 'motion file reloaded')
        else:
            self.publish_status('ERROR', f'unknown command: {command}')

    def stop_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ):
        del request
        self.request_stop('stop service called')
        response.success = True
        response.message = 'stop requested'
        return response

    def reload_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ):
        del request
        ok = self.load_paths()
        response.success = ok
        response.message = 'motion file reloaded' if ok else 'reload failed'
        self.publish_status('IDLE' if ok else 'ERROR', response.message)
        return response

    def start_path(self, path_name: str) -> bool:
        if not path_name:
            self.publish_status('ERROR', 'missing path name')
            return False

        with self.lock:
            if self.running:
                self.publish_status(
                    'BUSY',
                    f'already running {self.active_path_name}',
                )
                return False
            steps = copy.deepcopy(self.paths.get(path_name))

        if steps is None:
            self.publish_status('ERROR', f'unknown path: {path_name}')
            return False
        if not isinstance(steps, list) or not steps:
            self.publish_status('ERROR', f'path has no steps: {path_name}')
            return False

        return self.start_steps(path_name, steps)

    def start_steps(self, path_name: str, steps: list) -> bool:
        if not isinstance(steps, list) or not steps:
            self.publish_status('ERROR', f'path has no steps: {path_name}')
            return False

        with self.lock:
            if self.running:
                self.publish_status(
                    'BUSY',
                    f'already running {self.active_path_name}',
                )
                return False

        self.cancel_event.clear()
        with self.lock:
            self.running = True
            self.active_path_name = path_name

        self.active_thread = threading.Thread(
            target=self._run_path,
            args=(path_name, copy.deepcopy(steps)),
            daemon=True,
        )
        self.active_thread.start()
        return True

    def request_stop(self, message: str) -> None:
        self.cancel_event.set()
        self.publish_zero_cmd_vel()
        self.publish_status('STOPPING', message)

    def _run_path(self, path_name: str, steps: list) -> None:
        try:
            if not self.wait_for_odom(timeout_sec=5.0):
                self.publish_status('ERROR', 'odom not received')
                return

            total = len(steps)
            self.publish_status(
                'RUNNING',
                f'starting {path_name}',
                total=total,
            )
            for index, step in enumerate(steps, start=1):
                if self.cancel_event.is_set():
                    self.publish_status('CANCELED', 'motion canceled')
                    return

                if not isinstance(step, dict):
                    self.publish_status('ERROR', f'invalid step {index}')
                    return

                ok = self.execute_step(path_name, step, index, total)
                if not ok:
                    if self.cancel_event.is_set():
                        self.publish_status('CANCELED', 'motion canceled')
                    else:
                        self.publish_status('ERROR', f'step {index} failed')
                    return

            self.publish_zero_cmd_vel()
            self.publish_status('DONE', f'finished {path_name}', total=total)
        finally:
            self.publish_zero_cmd_vel()
            with self.lock:
                self.running = False
                self.active_path_name = None

    def wait_for_odom(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            with self.lock:
                if self.pose is not None:
                    return True
            time.sleep(0.05)
        return False

    def execute_step(
        self,
        path_name: str,
        step: dict,
        index: int,
        total: int,
    ) -> bool:
        step_type = str(step.get('type', '')).strip().lower()
        if step_type in ('move', 'forward', 'backward'):
            distance = float(step.get('distance', 0.0))
            if step_type == 'forward':
                distance = abs(distance)
            elif step_type == 'backward':
                distance = -abs(distance)
            speed = float(
                step.get(
                    'speed',
                    self.get_parameter('default_linear_speed').value,
                )
            )
            return self.move_distance(
                path_name,
                index,
                total,
                distance,
                speed,
            )

        if step_type in ('rotate', 'turn'):
            angle = float(step.get('angle', 0.0))
            angular_speed = float(
                step.get(
                    'angular_speed',
                    self.get_parameter('default_angular_speed').value,
                )
            )
            return self.rotate_angle(
                path_name,
                index,
                total,
                angle,
                angular_speed,
            )

        if step_type == 'wait':
            duration = float(step.get('duration', 0.0))
            return self.wait_step(path_name, index, total, duration)

        self.get_logger().error(f'Unknown step type: {step_type}')
        return False

    def move_distance(
        self,
        path_name: str,
        index: int,
        total: int,
        distance: float,
        speed: float,
    ) -> bool:
        target = abs(distance)
        if target <= 0.0:
            return True

        speed = abs(speed)
        if speed <= 0.0:
            self.get_logger().error('move speed must be positive')
            return False

        direction = 1.0 if distance >= 0.0 else -1.0
        tolerance = float(self.get_parameter('distance_tolerance_m').value)
        rate = self._rate_sleep_sec()

        with self.lock:
            start_pose = copy.deepcopy(self.pose)

        twist = Twist()
        twist.linear.x = direction * speed
        last_status_time = 0.0

        while rclpy.ok() and not self.cancel_event.is_set():
            with self.lock:
                current_pose = copy.deepcopy(self.pose)
            if current_pose is None:
                time.sleep(rate)
                continue

            traveled = math.hypot(
                current_pose['x'] - start_pose['x'],
                current_pose['y'] - start_pose['y'],
            )
            remaining = max(target - traveled, 0.0)

            now = time.monotonic()
            if now - last_status_time > 0.5:
                self.publish_status(
                    'RUNNING',
                    f'{path_name} step {index}/{total}: move {distance:.3f} m',
                    step_index=index,
                    step_count=total,
                    progress=traveled,
                    remaining=remaining,
                )
                last_status_time = now

            if traveled + tolerance >= target:
                break

            self.cmd_vel_pub.publish(twist)
            time.sleep(rate)

        self.publish_zero_cmd_vel()
        return not self.cancel_event.is_set()

    def rotate_angle(
        self,
        path_name: str,
        index: int,
        total: int,
        angle: float,
        angular_speed: float,
    ) -> bool:
        target = abs(angle)
        if target <= 0.0:
            return True

        angular_speed = abs(angular_speed)
        if angular_speed <= 0.0:
            self.get_logger().error('angular_speed must be positive')
            return False

        direction = 1.0 if angle >= 0.0 else -1.0
        tolerance = float(self.get_parameter('angle_tolerance_rad').value)
        rate = self._rate_sleep_sec()

        with self.lock:
            start_yaw = float(self.pose['yaw_unwrapped'])

        twist = Twist()
        twist.angular.z = direction * angular_speed
        last_status_time = 0.0

        while rclpy.ok() and not self.cancel_event.is_set():
            with self.lock:
                current_pose = copy.deepcopy(self.pose)
            if current_pose is None:
                time.sleep(rate)
                continue

            turned = abs(float(current_pose['yaw_unwrapped']) - start_yaw)
            remaining = max(target - turned, 0.0)

            now = time.monotonic()
            if now - last_status_time > 0.5:
                self.publish_status(
                    'RUNNING',
                    (
                        f'{path_name} step {index}/{total}: '
                        f'rotate {angle:.3f} rad'
                    ),
                    step_index=index,
                    step_count=total,
                    progress=turned,
                    remaining=remaining,
                )
                last_status_time = now

            if turned + tolerance >= target:
                break

            self.cmd_vel_pub.publish(twist)
            time.sleep(rate)

        self.publish_zero_cmd_vel()
        return not self.cancel_event.is_set()

    def wait_step(
        self,
        path_name: str,
        index: int,
        total: int,
        duration: float,
    ) -> bool:
        if duration <= 0.0:
            return True

        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self.cancel_event.is_set():
            self.publish_zero_cmd_vel()
            remaining = max(deadline - time.monotonic(), 0.0)
            self.publish_status(
                'RUNNING',
                f'{path_name} step {index}/{total}: wait {duration:.2f} s',
                step_index=index,
                step_count=total,
                remaining=remaining,
            )
            time.sleep(min(0.2, remaining))

        self.publish_zero_cmd_vel()
        return not self.cancel_event.is_set()

    def _rate_sleep_sec(self) -> float:
        rate_hz = float(self.get_parameter('rate_hz').value)
        return 1.0 / max(rate_hz, 1.0)

    def publish_zero_cmd_vel(self) -> None:
        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass

    def publish_status(
        self,
        state: str,
        message: str,
        step_index: Optional[int] = None,
        step_count: Optional[int] = None,
        total: Optional[int] = None,
        progress: Optional[float] = None,
        remaining: Optional[float] = None,
    ) -> None:
        with self.lock:
            path_name = self.active_path_name
            running = self.running

        payload = {
            'state': state,
            'message': message,
            'path': path_name,
            'running': running,
        }
        if step_index is not None:
            payload['step_index'] = step_index
        if step_count is not None:
            payload['step_count'] = step_count
        if total is not None:
            payload['step_count'] = total
        if progress is not None:
            payload['progress'] = progress
        if remaining is not None:
            payload['remaining'] = remaining

        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        try:
            self.status_pub.publish(msg)
        except Exception:
            return
        if state in ('ERROR', 'DONE', 'CANCELED', 'IDLE', 'BUSY', 'STOPPING'):
            self.get_logger().info(msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = MotionExecutor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.request_stop('shutdown')
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
