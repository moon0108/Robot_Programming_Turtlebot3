import os
import signal
import subprocess
import time
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String


class ArucoProcessManager(Node):
    """Start marker_comm ArUco nodes only while alignment is requested."""

    def __init__(self):
        super().__init__('aruco_process_manager')

        self.declare_parameter('active_topic', '/aruco_align/active')
        self.declare_parameter('shutdown_topic', '/aruco_align/shutdown')
        self.declare_parameter('done_topic', '/aruco_align/done')
        self.declare_parameter('status_topic', '/aruco_process/status')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('aruco_params_file', '')
        self.declare_parameter('image_topic', '/camera/image_decompressed')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('dictionary', 'DICT_4X4_50')
        self.declare_parameter('marker_ids', '0,1')
        self.declare_parameter('marker_size', 0.03)
        self.declare_parameter('startup_active_republish_count', 20)
        self.declare_parameter('startup_active_republish_period_sec', 0.2)
        self.declare_parameter('stop_timeout_sec', 2.0)
        self.declare_parameter('zero_cmd_publish_count', 5)
        self.declare_parameter('zero_cmd_publish_period_sec', 0.03)
        self.declare_parameter('cleanup_external_processes', True)

        self.active_topic = str(self.get_parameter('active_topic').value)
        self.shutdown_topic = str(self.get_parameter('shutdown_topic').value)
        self.done_topic = str(self.get_parameter('done_topic').value)
        self.status_topic = str(self.get_parameter('status_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.aruco_params_file = str(self.get_parameter('aruco_params_file').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.dictionary = str(self.get_parameter('dictionary').value)
        self.marker_ids = str(self.get_parameter('marker_ids').value)
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.republish_count = int(
            self.get_parameter('startup_active_republish_count').value
        )
        self.republish_period = float(
            self.get_parameter('startup_active_republish_period_sec').value
        )
        self.stop_timeout = float(self.get_parameter('stop_timeout_sec').value)
        self.zero_cmd_publish_count = int(
            self.get_parameter('zero_cmd_publish_count').value
        )
        self.zero_cmd_publish_period_sec = float(
            self.get_parameter('zero_cmd_publish_period_sec').value
        )
        self.cleanup_external_processes_enabled = bool(
            self.get_parameter('cleanup_external_processes').value
        )

        self.pose_process: Optional[subprocess.Popen] = None
        self.controller_process: Optional[subprocess.Popen] = None
        self.active_republish_remaining = 0
        self.stopping = False

        self.active_pub = self.create_publisher(Bool, self.active_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(Bool, self.active_topic, self.active_callback, 10)
        self.create_subscription(Bool, self.shutdown_topic, self.shutdown_callback, 10)
        self.create_subscription(Bool, self.done_topic, self.done_callback, 10)
        self.republish_timer = self.create_timer(
            self.republish_period,
            self.republish_active,
        )

        self.get_logger().info(
            'ArUco process manager ready. Nodes will start on active=true.'
        )
        self.get_logger().info(f'ArUco done topic: {self.done_topic}')
        if self.cleanup_external_processes_enabled:
            self.cleanup_external_aruco_processes('manager startup')

    def active_callback(self, msg: Bool) -> None:
        if msg.data:
            self.start_aruco_nodes()
        else:
            self.stop_aruco_nodes('active=false')

    def shutdown_callback(self, msg: Bool) -> None:
        if msg.data:
            self.stop_aruco_nodes('shutdown=true')

    def done_callback(self, msg: Bool) -> None:
        if msg.data:
            self.stop_aruco_nodes('alignment done')

    def start_aruco_nodes(self) -> None:
        self.cleanup_finished_processes()
        if self.pose_process is not None and self.controller_process is not None:
            return
        self.cleanup_external_aruco_processes('before start')

        self.get_logger().info('Starting ArUco pose and alignment controller nodes.')
        self.pose_process = self.start_process(self.pose_command(), 'aruco_pose')
        self.controller_process = self.start_process(
            self.controller_command(),
            'aruco_align_controller',
        )
        self.active_republish_remaining = self.republish_count
        self.publish_status('STARTED')

    def stop_aruco_nodes(self, reason: str) -> None:
        if self.stopping:
            return
        if self.pose_process is None and self.controller_process is None:
            self.active_republish_remaining = 0
            self.cleanup_external_aruco_processes(reason)
            self.publish_status('STOPPED')
            return

        self.stopping = True
        self.active_republish_remaining = 0
        self.publish_active(False)
        self.publish_zero_cmd_vel_burst()
        self.get_logger().info(f'Stopping ArUco nodes: {reason}')
        self.stop_process(self.controller_process, 'aruco_align_controller')
        self.stop_process(self.pose_process, 'aruco_pose')
        self.controller_process = None
        self.pose_process = None
        self.cleanup_external_aruco_processes(reason)
        self.publish_zero_cmd_vel_burst()
        self.publish_status('STOPPED')
        self.stopping = False

    def start_process(self, command: List[str], label: str) -> subprocess.Popen:
        env = os.environ.copy()
        process = subprocess.Popen(
            command,
            env=env,
            preexec_fn=os.setsid,
        )
        self.get_logger().info(f'{label} started with pid={process.pid}')
        return process

    def stop_process(
        self,
        process: Optional[subprocess.Popen],
        label: str,
    ) -> None:
        if process is None:
            return
        if process.poll() is not None:
            self.get_logger().info(f'{label} already exited with code={process.returncode}')
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=self.stop_timeout)
            self.get_logger().info(f'{label} stopped.')
            return
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f'{label} did not stop in time; terminating.')
        except ProcessLookupError:
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=self.stop_timeout)
            self.get_logger().info(f'{label} terminated.')
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f'{label} did not terminate; killing.')
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=self.stop_timeout)
        except ProcessLookupError:
            return

    def cleanup_finished_processes(self) -> None:
        if self.pose_process is not None and self.pose_process.poll() is not None:
            self.get_logger().warn(
                f'aruco_pose exited with code={self.pose_process.returncode}'
            )
            self.pose_process = None
        if (
            self.controller_process is not None
            and self.controller_process.poll() is not None
        ):
            self.get_logger().warn(
                'aruco_align_controller exited with '
                f'code={self.controller_process.returncode}'
            )
            self.controller_process = None

    def cleanup_external_aruco_processes(self, reason: str) -> None:
        if not self.cleanup_external_processes_enabled:
            return

        process_groups = self.find_external_aruco_process_groups()
        if not process_groups:
            return

        pids = sorted({pid for group in process_groups.values() for pid in group})
        self.get_logger().warn(
            'Stopping unmanaged ArUco processes '
            f'({reason}): pids={pids}'
        )
        self.publish_zero_cmd_vel_burst()

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            for pgid in process_groups:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    continue
                except PermissionError as exc:
                    self.get_logger().warn(
                        f'Cannot signal unmanaged ArUco process group {pgid}: {exc}'
                    )

            if self.wait_for_pids_to_exit(pids, self.stop_timeout):
                break

        remaining = [pid for pid in pids if self.pid_exists(pid)]
        if remaining:
            self.get_logger().warn(
                f'Unmanaged ArUco processes still visible after cleanup: {remaining}'
            )
        else:
            self.get_logger().info('Unmanaged ArUco processes stopped.')

        self.publish_zero_cmd_vel_burst()

    def find_external_aruco_process_groups(self) -> dict:
        managed_pgids = set()
        for process in (self.pose_process, self.controller_process):
            if process is None or process.poll() is not None:
                continue
            try:
                managed_pgids.add(os.getpgid(process.pid))
            except ProcessLookupError:
                continue

        current_pid = os.getpid()
        parent_pid = os.getppid()
        groups = {}
        try:
            result = subprocess.run(
                ['ps', '-eo', 'pid=,pgid=,args='],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            self.get_logger().warn(f'Cannot inspect process list: {exc}')
            return groups

        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                pgid = int(parts[1])
            except ValueError:
                continue
            command = parts[2]
            if pid in (current_pid, parent_pid) or pgid in managed_pgids:
                continue
            if self.is_aruco_process_command(command):
                groups.setdefault(pgid, []).append(pid)
        return groups

    def is_aruco_process_command(self, command: str) -> bool:
        command = ' '.join(command.split())
        if 'aruco_process_manager' in command:
            return False
        return (
            'ros2 run marker_comm aruco_pose' in command
            or 'marker_comm/lib/marker_comm/aruco_pose' in command
            or 'ros2 run marker_comm aruco_align_controller' in command
            or 'marker_comm/lib/marker_comm/aruco_align_controller' in command
        )

    def wait_for_pids_to_exit(self, pids: List[int], timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() <= deadline:
            if not any(self.pid_exists(pid) for pid in pids):
                return True
            time.sleep(0.05)
        return not any(self.pid_exists(pid) for pid in pids)

    def pid_exists(self, pid: int) -> bool:
        return os.path.exists(f'/proc/{pid}')

    def republish_active(self) -> None:
        self.cleanup_finished_processes()
        if self.active_republish_remaining <= 0:
            return
        if self.pose_process is None or self.controller_process is None:
            self.active_republish_remaining = 0
            return
        self.publish_active(True)
        self.active_republish_remaining -= 1

    def publish_active(self, active: bool) -> None:
        if not rclpy.ok():
            return
        msg = Bool()
        msg.data = active
        try:
            self.active_pub.publish(msg)
        except RCLError:
            pass

    def publish_status(self, state: str) -> None:
        if not rclpy.ok():
            return
        msg = String()
        msg.data = state
        try:
            self.status_pub.publish(msg)
        except RCLError:
            pass

    def publish_zero_cmd_vel_burst(self) -> None:
        if not rclpy.ok():
            return
        count = max(1, self.zero_cmd_publish_count)
        period = max(0.0, self.zero_cmd_publish_period_sec)
        for index in range(count):
            try:
                self.cmd_vel_pub.publish(Twist())
            except RCLError:
                return
            if period > 0.0 and index < count - 1:
                time.sleep(period)

    def pose_command(self) -> List[str]:
        command = ['ros2', 'run', 'marker_comm', 'aruco_pose']
        self.add_ros_args(command, 'aruco_pose')
        command.extend([
            '-p',
            f'image_topic:={self.image_topic}',
            '-p',
            f'camera_info_topic:={self.camera_info_topic}',
            '-p',
            f'dictionary:={self.dictionary}',
            '-p',
            f'marker_ids:={self.marker_ids}',
            '-p',
            f'marker_size:={self.marker_size}',
        ])
        return command

    def controller_command(self) -> List[str]:
        command = ['ros2', 'run', 'marker_comm', 'aruco_align_controller']
        self.add_ros_args(command, 'aruco_align_controller')
        return command

    def add_ros_args(self, command: List[str], node_name: str) -> None:
        command.extend(['--ros-args', '-r', f'__node:={node_name}'])
        if self.aruco_params_file:
            command.extend(['--params-file', self.aruco_params_file])

    def destroy_node(self) -> bool:
        self.stop_aruco_nodes('manager shutdown')
        return super().destroy_node()


def main(args=None):
    node = None
    rclpy.init(args=args)
    try:
        node = ArucoProcessManager()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except RCLError:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
