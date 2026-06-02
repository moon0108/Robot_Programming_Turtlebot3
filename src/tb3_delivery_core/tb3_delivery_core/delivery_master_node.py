import copy
import json
import math
import os
import threading
import time
import traceback
from typing import Dict, Optional, Tuple

import yaml

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import Trigger
from tb3_delivery_interfaces.action import ContainerStatus


class DeliveryMaster(Node):
    DEFAULT_PARAMS = {
        'nav_timeout_sec': 90.0,
        'aruco_center_x_tolerance_m': 0.02,
        'aruco_z_diff_tolerance_m': 0.01,
        'aruco_target_center_z_m': 0.20,
        'aruco_center_z_tolerance_m': 0.04,
        'aruco_center_x_offset_m': -0.01,
        'aruco_required_stable_count': 5,
        'aruco_timeout_sec': 20.0,
        'aruco_use_controller_status': True,
        'pickup_linear_x': 0.04,
        'pickup_duration_sec': 1.5,
        'lift_action_delay_sec': 3.4,
        'object_wait_timeout_sec': 30.0,
        'min_yolo_confidence': 0.3,
        'yolo_duplicate_window_sec': 0.1,
        'yolo_snapshot_service_timeout_sec': 12.0,
        'status_publish_period_sec': 1.0,
        'movement_mode': 'trace',
        'motion_timeout_sec': 120.0,
    }

    def __init__(self):
        super().__init__('delivery_master')

        self.lock = threading.RLock()
        self.thread_context = threading.local()
        self.cancel_event = threading.Event()
        self.yolo_event = threading.Event()
        self.motion_status_event = threading.Event()
        self.aruco_status_event = threading.Event()

        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('mission_params_file', '')

        waypoints_file = self.get_parameter('waypoints_file').value
        mission_params_file = self.get_parameter('mission_params_file').value
        self.waypoints_file = waypoints_file or self._default_config_path('waypoints.yaml')
        self.mission_params_file = mission_params_file or self._default_config_path(
            'mission_params.yaml'
        )

        self.waypoints = self._load_yaml_file(self.waypoints_file, 'waypoints')
        self.params = copy.deepcopy(self.DEFAULT_PARAMS)
        self.params.update(self._load_yaml_file(self.mission_params_file, 'mission params'))
        self.params['aruco_required_stable_count'] = int(
            self.params['aruco_required_stable_count']
        )

        self.state = 'IDLE'
        self.required: Dict[str, Dict[str, int]] = {}
        self.delivered: Dict[str, Dict[str, int]] = {}
        self.current_object: Optional[str] = None
        self.target_container: Optional[str] = None
        self.target_waypoint: Optional[str] = None
        self.current_location = 'station'
        self.last_status_message = 'delivery master ready'

        self.latest_yolo = None
        self.yolo_seq = 0
        self.latest_aruco = None
        self.aruco_seq = 0
        self.latest_aruco_status = None
        self.aruco_status_seq = 0
        self.latest_aruco_done = None
        self.aruco_done_seq = 0
        self.last_bad_aruco_log_time = 0.0
        self.latest_motion_status = None
        self.motion_status_seq = 0

        self.mission_running = False
        self.mission_thread = None
        self.debug_running = False
        self.debug_thread = None
        self.run_id = 0
        self.emergency_stop_active = False
        self.active_nav_goal_handle = None
        self.active_motion_path = None

        self.status_pub = self.create_publisher(String, '/delivery/status', 10)
        self.aruco_active_pub = self.create_publisher(Bool, '/aruco_align/active', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.motion_command_pub = self.create_publisher(
            String,
            '/tb3_motion/command',
            10,
        )

        self.create_subscription(String, '/delivery/order', self.order_callback, 10)
        self.create_subscription(String, '/delivery/debug_step', self.debug_step_callback, 10)
        self.create_subscription(String, '/yolo/best_object', self.yolo_callback, 10)
        self.create_subscription(String, '/yolo/detections', self.yolo_detections_callback, 10)
        self.create_subscription(
            Float32MultiArray,
            '/aruco/align_error',
            self.aruco_callback,
            10,
        )
        self.create_subscription(Bool, '/emergency_stop', self.emergency_stop_callback, 10)
        self.create_subscription(
            String,
            '/aruco_align/status',
            self.aruco_status_callback,
            10,
        )
        self.create_subscription(
            Bool,
            '/aruco_align/done',
            self.aruco_done_callback,
            10,
        )
        self.create_subscription(
            String,
            '/tb3_motion/status',
            self.motion_status_callback,
            10,
        )

        self.start_service = self.create_service(
            Trigger,
            '/delivery/start',
            self.start_callback,
        )
        self.cancel_service = self.create_service(
            Trigger,
            '/delivery/cancel',
            self.cancel_callback,
        )
        self.reset_service = self.create_service(
            Trigger,
            '/delivery/reset',
            self.reset_callback,
        )

        self.lift_push_client = self.create_client(Trigger, '/lift_push')
        self.lift_return_client = self.create_client(Trigger, '/lift_return')
        self.lift_stop_client = self.create_client(Trigger, '/lift_stop')
        self.motion_stop_client = self.create_client(Trigger, '/tb3_motion/stop')
        self.motion_reload_client = self.create_client(Trigger, '/tb3_motion/reload')
        self.yolo_snapshot_client = self.create_client(Trigger, '/yolo/snapshot')
        self.nav_action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.container_status_action = ActionServer(
            self,
            ContainerStatus,
            '/container_status',
            execute_callback=self.container_status_execute_callback,
            goal_callback=self.container_status_goal_callback,
            cancel_callback=self.container_status_cancel_callback,
        )

        period = max(0.1, float(self.params['status_publish_period_sec']))
        self.status_timer = self.create_timer(period, self.status_timer_callback)

        self.get_logger().info(f'Loaded waypoints from {self.waypoints_file}')
        self.get_logger().info(f'Loaded mission params from {self.mission_params_file}')
        self.get_logger().info(f'Available waypoints: {list(self.waypoints.keys())}')
        self.get_logger().info('Container status action server: /container_status')
        self.publish_status()

    def container_status_goal_callback(
        self,
        goal_request: ContainerStatus.Goal,
    ) -> GoalResponse:
        del goal_request
        return GoalResponse.ACCEPT

    def container_status_cancel_callback(self, goal_handle) -> CancelResponse:
        del goal_handle
        return CancelResponse.ACCEPT

    def container_status_execute_callback(self, goal_handle) -> ContainerStatus.Result:
        period = float(goal_handle.request.feedback_period_sec)
        if period <= 0.0:
            period = max(0.1, float(self.params['status_publish_period_sec']))

        self.get_logger().info(
            f'Container status action accepted, feedback_period_sec={period:.2f}'
        )

        feedback = ContainerStatus.Feedback()
        while rclpy.ok():
            snapshot = self._container_status_snapshot()
            feedback.all_filled = bool(snapshot['all_filled'])
            feedback.progress = float(snapshot['progress'])
            feedback.current_status_json = json.dumps(
                snapshot,
                sort_keys=True,
                ensure_ascii=False,
            )
            goal_handle.publish_feedback(feedback)

            if feedback.all_filled:
                goal_handle.succeed()
                result = ContainerStatus.Result()
                result.all_filled = True
                result.final_status_json = feedback.current_status_json
                result.message = 'all containers filled'
                self.get_logger().info('Container status action succeeded')
                return result

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = ContainerStatus.Result()
                result.all_filled = False
                result.final_status_json = feedback.current_status_json
                result.message = 'container status action canceled'
                self.get_logger().info(result.message)
                return result

            self._sleep_action_period(period, goal_handle)

        goal_handle.abort()
        result = ContainerStatus.Result()
        result.all_filled = False
        result.final_status_json = json.dumps(
            self._container_status_snapshot(),
            sort_keys=True,
            ensure_ascii=False,
        )
        result.message = 'ROS shutdown before all containers were filled'
        return result

    def _sleep_action_period(self, period_sec: float, goal_handle) -> None:
        deadline = time.monotonic() + period_sec
        while time.monotonic() < deadline and rclpy.ok():
            if goal_handle.is_cancel_requested:
                return
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))

    def _container_status_snapshot(self) -> dict:
        with self.lock:
            containers = {}
            total_required = 0
            total_delivered = 0

            for container, objects in self.required.items():
                container_required = 0
                container_delivered = 0
                items = {}
                for object_class, required_count in objects.items():
                    required_count = int(required_count)
                    delivered_count = int(
                        self.delivered.get(container, {}).get(object_class, 0)
                    )
                    remaining_count = max(required_count - delivered_count, 0)
                    items[object_class] = {
                        'required': required_count,
                        'delivered': delivered_count,
                        'remaining': remaining_count,
                        'filled': remaining_count == 0,
                    }
                    container_required += required_count
                    container_delivered += min(delivered_count, required_count)

                total_required += container_required
                total_delivered += container_delivered
                containers[container] = {
                    'items': items,
                    'required_total': container_required,
                    'delivered_total': container_delivered,
                    'remaining_total': max(container_required - container_delivered, 0),
                    'filled': container_required > 0
                    and container_delivered >= container_required,
                }

            all_filled = self._all_done_locked()
            progress = (
                float(total_delivered) / float(total_required)
                if total_required > 0
                else 0.0
            )
            return {
                'state': self.state,
                'message': self.last_status_message,
                'containers': containers,
                'current_object': self.current_object,
                'target_container': self.target_container,
                'target_waypoint': self.target_waypoint,
                'current_location': self.current_location,
                'mission_running': self.mission_running,
                'debug_running': self.debug_running,
                'active_motion_path': self.active_motion_path,
                'total_required': total_required,
                'total_delivered': total_delivered,
                'total_remaining': max(total_required - total_delivered, 0),
                'progress': progress,
                'all_filled': all_filled,
            }

    def _default_config_path(self, filename: str) -> str:
        try:
            share_dir = get_package_share_directory('tb3_delivery_core')
            installed_path = os.path.join(share_dir, 'config', filename)
            if os.path.exists(installed_path):
                return installed_path
        except PackageNotFoundError:
            pass

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(package_root, 'config', filename)

    def _load_yaml_file(self, path: str, label: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as yaml_file:
                data = yaml.safe_load(yaml_file) or {}
        except FileNotFoundError:
            self.get_logger().error(f'{label} file not found: {path}')
            return {}
        except yaml.YAMLError as exc:
            self.get_logger().error(f'Failed to parse {label} file {path}: {exc}')
            return {}

        if not isinstance(data, dict):
            self.get_logger().error(f'{label} file must contain a YAML map: {path}')
            return {}
        return data

    def order_callback(self, msg: String) -> None:
        with self.lock:
            if self.mission_running:
                self.last_status_message = 'order ignored while mission is running'
                self.get_logger().warn(self.last_status_message)
                self.publish_status()
                return

        try:
            required = self._parse_order(msg.data)
        except ValueError as exc:
            self.get_logger().error(f'Invalid order JSON: {exc}')
            with self.lock:
                self.required = {}
                self.delivered = {}
                self.current_object = None
                self.target_container = None
                self.target_waypoint = None
            self._set_state('ERROR', f'invalid order: {exc}')
            return

        delivered = {
            container: {object_class: 0 for object_class in objects}
            for container, objects in required.items()
        }

        with self.lock:
            self.required = required
            self.delivered = delivered
            self.current_object = None
            self.target_container = None
            self.target_waypoint = None
            self.cancel_event.clear()

        self.get_logger().info(f'Order received: {required}')
        self._set_state('ORDER_RECEIVED', 'order received')

    def debug_step_callback(self, msg: String) -> None:
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._set_state('ERROR', f'invalid debug command JSON: {exc}')
            return

        if not isinstance(command, dict):
            self._set_state('ERROR', 'debug command must be a JSON object')
            return

        step = str(command.get('step', '')).strip()
        if not step:
            self._set_state('ERROR', 'debug command missing step')
            return

        if step == 'stop_all':
            self._request_cancel('debug stop all requested')
            return

        with self.lock:
            if self.emergency_stop_active:
                self.last_status_message = 'debug step ignored: emergency stop is active'
                self.publish_status()
                return
            if self.mission_running:
                self.last_status_message = 'debug step ignored while mission is running'
                self.publish_status()
                return
            if self.debug_running:
                self.last_status_message = 'debug step ignored while another debug step is running'
                self.publish_status()
                return

            self.cancel_event.clear()
            self.run_id += 1
            run_id = self.run_id
            self.debug_running = True
            self.debug_thread = threading.Thread(
                target=self._debug_step_main,
                args=(run_id, command),
                daemon=True,
            )
            self.debug_thread.start()

    def _parse_order(self, data: str) -> Dict[str, Dict[str, int]]:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc

        if not isinstance(payload, dict):
            raise ValueError('top-level order must be a JSON object')

        required: Dict[str, Dict[str, int]] = {}
        for container, objects in payload.items():
            container_name = str(container).strip()
            if not container_name:
                raise ValueError('container name cannot be empty')
            if container_name not in self.waypoints:
                raise ValueError(f'unknown waypoint/container: {container_name}')
            if not isinstance(objects, dict):
                raise ValueError(f'{container_name} must contain an object map')

            parsed_objects = {}
            for object_class, quantity in objects.items():
                object_name = str(object_class).strip()
                if not object_name:
                    raise ValueError(f'{container_name} contains an empty object name')
                try:
                    count = int(quantity)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'{container_name}.{object_name} quantity must be an integer'
                    ) from exc
                if count < 0:
                    raise ValueError(
                        f'{container_name}.{object_name} quantity cannot be negative'
                    )
                if count > 0:
                    parsed_objects[object_name] = count

            if parsed_objects:
                required[container_name] = parsed_objects

        if not required:
            raise ValueError('order has no positive item quantities')

        return required

    def yolo_callback(self, msg: String) -> None:
        parsed = self._parse_yolo(msg.data)
        if parsed is None:
            return

        object_class, confidence = parsed
        self._record_yolo_object(object_class, confidence, source='best_object')

    def yolo_detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Ignoring malformed /yolo/detections JSON')
            return

        if not isinstance(payload, list):
            self.get_logger().warn('/yolo/detections must be a JSON list')
            return

        best_detection = None
        best_confidence = -1.0
        for item in payload:
            parsed = self._parse_yolo_detection_item(item)
            if parsed is None:
                continue
            object_class, confidence = parsed
            if confidence > best_confidence:
                best_detection = object_class
                best_confidence = confidence

        if best_detection is None:
            return

        self._record_yolo_object(best_detection, best_confidence, source='detections')

    def _record_yolo_object(
        self,
        object_class: str,
        confidence: float,
        source: str,
    ) -> None:
        now = time.monotonic()
        with self.lock:
            previous = copy.deepcopy(self.latest_yolo)
            duplicate_window = float(self.params.get('yolo_duplicate_window_sec', 0.1))
            if (
                previous is not None
                and previous.get('class') == object_class
                and abs(float(previous.get('confidence', 0.0)) - confidence) < 1e-6
                and now - float(previous.get('time', 0.0)) < duplicate_window
            ):
                return

            self.yolo_seq += 1
            self.latest_yolo = {
                'class': object_class,
                'confidence': confidence,
                'source': source,
                'seq': self.yolo_seq,
                'time': now,
            }
        self.yolo_event.set()
        self.get_logger().debug(
            f'YOLO object received from {source}: '
            f'{object_class} confidence={confidence:.3f}'
        )

    def _parse_yolo(self, data: str) -> Optional[Tuple[str, float]]:
        text = data.strip()
        if not text:
            return None

        object_class = text
        confidence = 1.0
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            object_class = str(
                payload.get('class')
                or payload.get('object_class')
                or payload.get('label')
                or ''
            ).strip()
            try:
                confidence = float(payload.get('confidence', 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
        elif isinstance(payload, str):
            object_class = payload.strip()

        if not object_class:
            self.get_logger().warn('YOLO message did not contain an object class')
            return None

        if confidence < float(self.params['min_yolo_confidence']):
            self.get_logger().info(
                f'Ignoring {object_class}: confidence {confidence:.3f} below threshold'
            )
            return None

        return object_class, confidence

    def _parse_yolo_detection_item(self, item) -> Optional[Tuple[str, float]]:
        if not isinstance(item, dict):
            return None

        object_class = str(
            item.get('class')
            or item.get('object_class')
            or item.get('label')
            or item.get('class_name')
            or ''
        ).strip()
        if not object_class:
            return None

        try:
            confidence = float(item.get('confidence', 1.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < float(self.params['min_yolo_confidence']):
            return None
        return object_class, confidence

    def aruco_callback(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 8:
            now = time.monotonic()
            if now - self.last_bad_aruco_log_time > 5.0:
                self.get_logger().warn(
                    f'Ignoring malformed /aruco/align_error with {len(data)} values'
                )
                self.last_bad_aruco_log_time = now
            return

        with self.lock:
            self.aruco_seq += 1
            self.latest_aruco = {
                'data': data,
                'seq': self.aruco_seq,
                'time': time.monotonic(),
            }

    def aruco_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {
                'state': 'UNKNOWN',
                'message': msg.data,
            }

        with self.lock:
            self.aruco_status_seq += 1
            payload['seq'] = self.aruco_status_seq
            payload['time'] = time.monotonic()
            self.latest_aruco_status = payload

        self.aruco_status_event.set()

    def aruco_done_callback(self, msg: Bool) -> None:
        with self.lock:
            self.aruco_done_seq += 1
            self.latest_aruco_done = {
                'done': bool(msg.data),
                'seq': self.aruco_done_seq,
                'time': time.monotonic(),
            }

    def motion_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {
                'state': 'UNKNOWN',
                'message': msg.data,
            }

        with self.lock:
            self.motion_status_seq += 1
            payload['seq'] = self.motion_status_seq
            payload['time'] = time.monotonic()
            self.latest_motion_status = payload

        self.motion_status_event.set()

    def emergency_stop_callback(self, msg: Bool) -> None:
        if msg.data:
            with self.lock:
                first_event = not self.emergency_stop_active
                self.emergency_stop_active = True
            if first_event:
                self.get_logger().warn('Emergency stop ON')
            self._request_cancel('emergency stop')
        else:
            with self.lock:
                self.emergency_stop_active = False
                self.last_status_message = 'emergency stop cleared'
            self.get_logger().info('Emergency stop OFF')
            self.publish_status()

    def start_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        with self.lock:
            if self.emergency_stop_active:
                response.success = False
                response.message = 'emergency stop is active'
                return response
            if self.mission_running:
                response.success = False
                response.message = 'mission already running'
                return response
            if self.debug_running:
                response.success = False
                response.message = 'debug step is running'
                return response
            if not self.required:
                response.success = False
                response.message = 'no order received'
                return response

            self.cancel_event.clear()
            self.run_id += 1
            run_id = self.run_id
            self.mission_running = True
            self.mission_thread = threading.Thread(
                target=self._mission_main,
                args=(run_id,),
                daemon=True,
            )
            self.mission_thread.start()

        response.success = True
        response.message = 'mission started'
        self.get_logger().info(response.message)
        return response

    def cancel_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self._request_cancel('mission canceled by request')
        response.success = True
        response.message = 'mission cancel requested'
        return response

    def reset_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self.cancel_event.set()
        self._safe_stop()
        self._cancel_active_nav_goal()
        self._call_motion_stop_async()
        self._call_lift_stop_async()

        with self.lock:
            self.run_id += 1
            self.mission_running = False
            self.debug_running = False
            self.required = {}
            self.delivered = {}
            self.current_object = None
            self.target_container = None
            self.target_waypoint = None
            self.current_location = 'station'
            self.latest_yolo = None
            self.yolo_seq = 0
            self.latest_aruco = None
            self.aruco_seq = 0
            self.latest_aruco_status = None
            self.aruco_status_seq = 0
            self.latest_aruco_done = None
            self.aruco_done_seq = 0
            self.latest_motion_status = None
            self.motion_status_seq = 0
            self.active_motion_path = None
            self.last_status_message = 'reset complete'
            self.state = 'IDLE'

        self.get_logger().info('Mission reset')
        self.publish_status()
        response.success = True
        response.message = 'mission reset'
        return response

    def _debug_step_main(self, run_id: int, command: dict) -> None:
        self.thread_context.run_id = run_id
        step = str(command.get('step', '')).strip()
        try:
            ok = self._execute_debug_step(step, command)
            if self._cancel_requested():
                self._finish_canceled()
            elif ok:
                self._set_state('DEBUG_DONE', f'debug step complete: {step}')
            elif self.state not in ('ERROR', 'CANCELED'):
                self._fail(f'debug step failed: {step}')
        except Exception as exc:  # noqa: BLE001 - debug must still leave robot stopped.
            self.get_logger().error(
                f'Unhandled debug step error: {exc}\n{traceback.format_exc()}'
            )
            self._fail(f'unhandled debug step error: {exc}')
        finally:
            self._safe_stop()
            with self.lock:
                if self._is_run_current(run_id):
                    self.debug_running = False
                    self.active_nav_goal_handle = None
                    self.active_motion_path = None
            if hasattr(self.thread_context, 'run_id'):
                del self.thread_context.run_id

    def _execute_debug_step(self, step: str, command: dict) -> bool:
        if step == 'motion_path':
            path_name = str(command.get('path', '')).strip()
            if not path_name:
                self._fail('debug motion_path missing path')
                return False
            source, target = self._locations_from_motion_path(path_name)
            with self.lock:
                if source is not None:
                    self.current_location = source
                self.current_object = None
                self.target_container = None
                self.target_waypoint = target
            self._set_state(
                'DEBUG_MOTION',
                f'running debug motion path {path_name}',
                target_waypoint=target,
            )
            return self._run_motion_path_command(path_name, final_location=target)

        if step == 'aruco_align':
            location = str(command.get('location', '')).strip()
            if not location:
                with self.lock:
                    location = self.current_location or 'station'
            if location not in self.waypoints:
                self._fail(f'unknown debug ArUco location: {location}')
                return False
            self._set_state(
                'DEBUG_ARUCO_ALIGN',
                f'running debug ArUco alignment at {location}',
                target_waypoint=location,
            )
            return self.align_with_aruco(location)

        if step == 'yolo_snapshot':
            return self._run_yolo_snapshot_debug(command)

        if step == 'lift_push':
            self._set_state('DEBUG_LIFT_PUSH', 'calling /lift_push')
            ok, message = self._call_trigger_service(self.lift_push_client, '/lift_push')
            if not ok:
                self._fail(f'/lift_push failed: {message}')
            return ok

        if step == 'lift_return':
            self._set_state('DEBUG_LIFT_RETURN', 'calling /lift_return')
            ok, message = self._call_trigger_service(
                self.lift_return_client,
                '/lift_return',
            )
            if not ok:
                self._fail(f'/lift_return failed: {message}')
            return ok

        if step == 'lift_stop':
            self._set_state('DEBUG_LIFT_STOP', 'calling /lift_stop')
            ok, message = self._call_trigger_service(self.lift_stop_client, '/lift_stop')
            if not ok:
                self._fail(f'/lift_stop failed: {message}')
            return ok

        self._fail(f'unknown debug step: {step}')
        return False

    def _mission_main(self, run_id: int) -> None:
        self.thread_context.run_id = run_id
        try:
            if not self._is_run_current(run_id):
                return

            self._set_state(
                'MISSION_ACTIVE',
                'delivery mission started at station',
                target_waypoint='station',
            )
            with self.lock:
                self.current_location = self.current_location or 'station'

            while not self._cancel_requested():
                if self._all_done():
                    with self.lock:
                        self.current_object = None
                        self.target_container = None
                        self.target_waypoint = None
                    self._set_state('DONE', 'all deliveries completed')
                    self._safe_stop()
                    return

                selected = self._wait_for_needed_object()
                if selected is None:
                    self._finish_failed_step('YOLO object detection timed out')
                    return

                object_class, target_container = selected
                with self.lock:
                    self.current_object = object_class
                    self.target_container = target_container
                    self.target_waypoint = target_container

                self._set_state(
                    'STATION_TO_CONTAINER',
                    f'moving to {target_container}',
                    target_waypoint=target_container,
                )
                if not self.go_to_waypoint(target_container):
                    self._finish_failed_step(f'failed to move to {target_container}')
                    return

                self._set_state(
                    'ALIGN_CONTAINER',
                    f'aligning at {target_container}',
                    target_waypoint=target_container,
                )
                if not self.align_with_aruco(target_container):
                    self._finish_failed_step(f'ArUco alignment failed at {target_container}')
                    return

                if not self.perform_lift_sequence():
                    if self._cancel_requested():
                        self._finish_canceled()
                    elif self.state != 'ERROR':
                        self._fail('lift sequence failed')
                    return

                self._set_state(
                    'UPDATE_PROGRESS',
                    f'delivered {object_class} to {target_container}',
                    target_waypoint=target_container,
                )
                if not self._update_delivered_count(target_container, object_class):
                    self._fail(f'could not update progress for {target_container}/{object_class}')
                    return

                self._set_state('CHECK_DONE', 'checking remaining deliveries')
                with self.lock:
                    self.current_object = None
                    self.target_container = None
                    self.target_waypoint = 'station'

                if not self._return_to_station():
                    self._finish_failed_step('failed to return to station')
                    return

                if self._all_done():
                    with self.lock:
                        self.current_object = None
                        self.target_container = None
                        self.target_waypoint = None
                    self._set_state('DONE', 'all deliveries completed')
                    self._safe_stop()
                    return

                self._set_state(
                    'ALIGN_STATION',
                    'aligning at station before next detection',
                    target_waypoint='station',
                )
                if not self.align_with_aruco('station'):
                    if self._cancel_requested():
                        self._finish_canceled()
                        return
                    self._safe_stop()
                    self._call_motion_stop_async()
                    self._set_state(
                        'YOLO_DETECT',
                        'station ArUco alignment failed; continuing to YOLO detection',
                        target_waypoint='station',
                    )
                    continue

            self._finish_canceled()
        except Exception as exc:  # noqa: BLE001 - keep the robot safe on any mission error.
            self.get_logger().error(
                f'Unhandled mission error: {exc}\n{traceback.format_exc()}'
            )
            self._fail(f'unhandled mission error: {exc}')
        finally:
            self._safe_stop()
            with self.lock:
                if self._is_run_current(run_id):
                    self.mission_running = False
                    self.active_nav_goal_handle = None
                    self.active_motion_path = None
            if hasattr(self.thread_context, 'run_id'):
                del self.thread_context.run_id

    def _return_to_station(self) -> bool:
        self._set_state(
            'CONTAINER_TO_STATION',
            'returning to station',
            target_waypoint='station',
        )
        return self.go_to_waypoint('station')

    def _finish_failed_step(self, message: str) -> None:
        if self._cancel_requested():
            self._finish_canceled()
        else:
            self._fail(message)

    def _finish_canceled(self) -> None:
        self._safe_stop()
        self._call_motion_stop_async()
        self._call_lift_stop_async()
        self._set_state('CANCELED', 'mission canceled')

    def _request_cancel(self, message: str) -> None:
        self.cancel_event.set()
        self._safe_stop()
        self._cancel_active_nav_goal()
        self._call_motion_stop_async()
        self._call_lift_stop_async()
        self._set_state('CANCELED', message)

    def _fail(self, message: str) -> None:
        self.get_logger().error(message)
        self._safe_stop()
        self._call_motion_stop_async()
        self._call_lift_stop_async()
        self._set_state('ERROR', message)

    def _is_run_current(self, run_id: int) -> bool:
        with self.lock:
            return run_id == self.run_id

    def _cancel_requested(self) -> bool:
        run_id = getattr(self.thread_context, 'run_id', None)
        with self.lock:
            stale_run = run_id is not None and run_id != self.run_id
            return self.cancel_event.is_set() or self.emergency_stop_active or stale_run

    def go_to_waypoint(self, name: str) -> bool:
        movement_mode = str(self.params.get('movement_mode', 'trace')).lower()
        if movement_mode in ('trace', 'motion', 'cmd_vel'):
            return self.go_to_waypoint_motion(name)
        return self.go_to_waypoint_nav2(name)

    def go_to_waypoint_motion(self, name: str) -> bool:
        if name not in self.waypoints:
            self.get_logger().error(f'Unknown waypoint: {name}')
            return False

        with self.lock:
            current_location = self.current_location or 'station'

        if current_location == name:
            self.get_logger().info(f'Already at {name}; no motion path needed')
            with self.lock:
                self.current_location = name
            return True

        path_name = f'{current_location}_to_{name}'
        self.get_logger().info(f'Running trace motion path: {path_name}')

        if self._cancel_requested():
            return False

        return self._run_motion_path_command(path_name, final_location=name)

    def _run_motion_path_command(
        self,
        path_name: str,
        final_location: Optional[str] = None,
    ) -> bool:
        if self.motion_reload_client.wait_for_service(timeout_sec=0.5):
            reload_future = self.motion_reload_client.call_async(Trigger.Request())
            self._wait_for_future(reload_future, timeout_sec=2.0)
        else:
            self.get_logger().warn('/tb3_motion/reload service is not available')

        with self.lock:
            start_seq = self.motion_status_seq
            self.active_motion_path = path_name

        command = String()
        command.data = json.dumps({
            'command': 'run',
            'path': path_name,
        })
        self.motion_command_pub.publish(command)

        deadline = time.monotonic() + float(self.params['motion_timeout_sec'])
        while time.monotonic() < deadline:
            if self._cancel_requested():
                self._call_motion_stop_async()
                with self.lock:
                    self.active_motion_path = None
                return False

            self.motion_status_event.wait(timeout=0.1)
            self.motion_status_event.clear()

            with self.lock:
                status = copy.deepcopy(self.latest_motion_status)

            if status is None or status.get('seq', 0) <= start_seq:
                continue

            status_path = status.get('path')
            state = str(status.get('state', '')).upper()
            if status_path not in (path_name, None, ''):
                continue

            if state == 'DONE':
                if status_path != path_name:
                    continue
                self.get_logger().info(f'Trace motion path done: {path_name}')
                with self.lock:
                    if final_location is not None:
                        self.current_location = final_location
                    self.active_motion_path = None
                return True
            if state in ('ERROR', 'CANCELED', 'BUSY'):
                message = status.get('message', '')
                self.get_logger().error(
                    f'Trace motion path {path_name} failed: {state} {message}'
                )
                with self.lock:
                    self.active_motion_path = None
                return False

        self.get_logger().error(f'Trace motion path timed out: {path_name}')
        self._call_motion_stop_async()
        with self.lock:
            self.active_motion_path = None
        return False

    def _locations_from_motion_path(
        self,
        path_name: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        for source in self.waypoints:
            prefix = f'{source}_to_'
            if not path_name.startswith(prefix):
                continue
            target = path_name[len(prefix):]
            if target in self.waypoints:
                return source, target
        return None, None

    def go_to_waypoint_nav2(self, name: str) -> bool:
        waypoint = self.waypoints.get(name)
        if waypoint is None:
            self.get_logger().error(f'Unknown waypoint: {name}')
            return False

        if self._cancel_requested():
            return False

        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 /navigate_to_pose action server is not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = self._waypoint_to_pose(name, waypoint)

        self.get_logger().info(
            f'Sending Nav2 goal {name}: '
            f'x={float(waypoint["x"]):.3f}, y={float(waypoint["y"]):.3f}, '
            f'yaw={float(waypoint["yaw"]):.3f}'
        )

        try:
            send_future = self.nav_action_client.send_goal_async(goal)
            if not self._wait_for_future(send_future, timeout_sec=10.0):
                self.get_logger().error(f'Timed out sending Nav2 goal: {name}')
                return False

            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().error(f'Nav2 goal rejected: {name}')
                return False

            with self.lock:
                self.active_nav_goal_handle = goal_handle

            result_future = goal_handle.get_result_async()
            nav_timeout = float(self.params['nav_timeout_sec'])
            if not self._wait_for_future(result_future, timeout_sec=nav_timeout):
                self.get_logger().error(f'Nav2 goal timed out or was canceled: {name}')
                self._cancel_active_nav_goal()
                return False

            result = result_future.result()
            with self.lock:
                self.active_nav_goal_handle = None

            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f'Nav2 reached waypoint: {name}')
                with self.lock:
                    self.current_location = name
                return True

            self.get_logger().error(f'Nav2 failed for {name}, status={result.status}')
            return False
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Nav2 goal failed for {name}: {exc}')
            with self.lock:
                self.active_nav_goal_handle = None
            return False

    def _waypoint_to_pose(self, name: str, waypoint: dict) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(waypoint.get('frame_id', 'map'))
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(waypoint['x'])
        pose.pose.position.y = float(waypoint['y'])
        pose.pose.position.z = 0.0
        yaw = float(waypoint['yaw'])
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.get_logger().debug(f'Converted waypoint {name} to PoseStamped')
        return pose

    def align_with_aruco(self, location_name: str) -> bool:
        self.get_logger().info(f'Starting ArUco alignment at {location_name}')

        stable_count = 0
        required_count = int(self.params['aruco_required_stable_count'])
        timeout = float(self.params['aruco_timeout_sec'])
        deadline = time.monotonic() + timeout
        use_controller_status = bool(self.params.get('aruco_use_controller_status', True))

        with self.lock:
            last_seq = self.aruco_seq
            last_status_seq = self.aruco_status_seq
            last_done_seq = self.aruco_done_seq

        self._publish_aruco_active(True)

        try:
            while time.monotonic() < deadline:
                if self._cancel_requested():
                    return False

                with self.lock:
                    sample = copy.deepcopy(self.latest_aruco)
                    status = copy.deepcopy(self.latest_aruco_status)
                    done = copy.deepcopy(self.latest_aruco_done)

                if (
                    done is not None
                    and done.get('seq', 0) > last_done_seq
                    and done.get('done', False)
                ):
                    last_done_seq = done['seq']
                    self.get_logger().info(
                        f'ArUco controller done received at {location_name}'
                    )
                    return True

                if (
                    use_controller_status
                    and status is not None
                    and status.get('seq', 0) > last_status_seq
                ):
                    last_status_seq = status['seq']
                    if self._aruco_status_is_aligned(status):
                        self.get_logger().info(
                            f'ArUco controller reported aligned at {location_name}'
                        )
                        return True

                if sample is None or sample['seq'] == last_seq:
                    time.sleep(0.05)
                    continue

                last_seq = sample['seq']
                if self._aruco_sample_is_aligned(sample['data']):
                    stable_count += 1
                    self.get_logger().debug(
                        f'ArUco stable sample {stable_count}/{required_count}'
                    )
                    if stable_count >= required_count:
                        self.get_logger().info(
                            f'ArUco alignment complete at {location_name}'
                        )
                        return True
                else:
                    stable_count = 0

                time.sleep(0.02)

            self.get_logger().error(f'ArUco alignment timed out at {location_name}')
            return False
        finally:
            self._publish_aruco_active(False)

    def _aruco_sample_is_aligned(self, data) -> bool:
        center_x_m = float(data[0]) - float(self.params.get('aruco_center_x_offset_m', 0.0))
        center_z_m = float(data[1])
        z_diff_m = float(data[2])
        both_markers_visible = float(data[3])

        return (
            both_markers_visible >= 0.5
            and abs(center_x_m) < float(self.params['aruco_center_x_tolerance_m'])
            and abs(z_diff_m) < float(self.params['aruco_z_diff_tolerance_m'])
            and abs(center_z_m - float(self.params['aruco_target_center_z_m']))
            < float(self.params['aruco_center_z_tolerance_m'])
        )

    def _aruco_status_is_aligned(self, status: dict) -> bool:
        state = str(status.get('state', '')).upper()
        mode = str(status.get('mode', '')).lower()
        active = bool(status.get('active', True))
        aligned = bool(status.get('aligned', False))
        return active and (aligned or state == 'ALIGNED' or mode in ('aligned', 'aligned_done'))

    def _wait_for_needed_object(self) -> Optional[Tuple[str, str]]:
        self._set_state('YOLO_DETECT', 'waiting for YOLO object', target_waypoint='station')
        timeout = float(self.params['object_wait_timeout_sec'])
        deadline = time.monotonic() + timeout

        with self.lock:
            last_seq = self.yolo_seq

        while time.monotonic() < deadline:
            if self._cancel_requested():
                return None

            remaining = max(0.1, deadline - time.monotonic())
            if self.yolo_snapshot_client.wait_for_service(timeout_sec=0.1):
                self._set_state(
                    'YOLO_DETECT',
                    'running YOLO snapshot for object detection',
                    target_waypoint='station',
                )
                self._call_trigger_service(
                    self.yolo_snapshot_client,
                    '/yolo/snapshot',
                    timeout_sec=min(
                        remaining,
                        float(self.params.get('yolo_snapshot_service_timeout_sec', 12.0)),
                    ),
                )
            else:
                self.yolo_event.wait(timeout=0.1)
                self.yolo_event.clear()

            with self.lock:
                latest_yolo = copy.deepcopy(self.latest_yolo)

            if latest_yolo is None or latest_yolo['seq'] <= last_seq:
                continue

            last_seq = latest_yolo['seq']
            object_class = latest_yolo['class']
            with self.lock:
                self.current_object = object_class
                self.target_container = None

            self._set_state('DECIDE_TARGET', f'deciding target for {object_class}')
            target_container = self._find_target_container(object_class)
            if target_container is None:
                self.get_logger().info(
                    f'No container currently needs {object_class}; waiting again'
                )
                with self.lock:
                    self.current_object = None
                    self.target_container = None
                    self.target_waypoint = 'station'
                self._set_state(
                    'YOLO_DETECT',
                    f'{object_class} not needed; waiting for next object',
                    target_waypoint='station',
                )
                continue

            with self.lock:
                self.target_container = target_container
                self.target_waypoint = target_container
            self.get_logger().info(f'{object_class} assigned to {target_container}')
            return object_class, target_container

        self.get_logger().error('Timed out waiting for needed YOLO object')
        return None

    def _find_target_container(self, object_class: str) -> Optional[str]:
        with self.lock:
            required = copy.deepcopy(self.required)
            delivered = copy.deepcopy(self.delivered)

        for container, objects in required.items():
            if object_class not in objects:
                continue
            if delivered.get(container, {}).get(object_class, 0) < objects[object_class]:
                return container
        return None

    def pickup_approach(self) -> bool:
        self._publish_aruco_active(False)
        duration = float(self.params['pickup_duration_sec'])
        linear_x = float(self.params['pickup_linear_x'])
        return self._drive_cmd_vel_for(linear_x, duration)

    def _drive_cmd_vel_for(self, linear_x: float, duration: float) -> bool:
        deadline = time.monotonic() + duration

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = 0.0

        try:
            while time.monotonic() < deadline:
                if self._cancel_requested():
                    return False
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.1)
            return True
        finally:
            self._publish_zero_cmd_vel()

    def _run_yolo_snapshot_debug(self, command: dict) -> bool:
        self._set_state('DEBUG_YOLO_SNAPSHOT', 'requesting /yolo/snapshot')
        if self.yolo_snapshot_client.wait_for_service(timeout_sec=0.2):
            ok, message = self._call_trigger_service(
                self.yolo_snapshot_client,
                '/yolo/snapshot',
                timeout_sec=float(
                    self.params.get('yolo_snapshot_service_timeout_sec', 12.0)
                ),
            )
            if not ok:
                self._fail(f'/yolo/snapshot failed: {message}')
                return False
            self._set_state('DEBUG_YOLO_SNAPSHOT', message)
            return True

        with self.lock:
            latest_yolo = copy.deepcopy(self.latest_yolo)

        if latest_yolo is None:
            self._set_state(
                'DEBUG_YOLO_SNAPSHOT',
                'latest YOLO: none (/yolo/snapshot unavailable)',
            )
            return True

        self._set_state(
            'DEBUG_YOLO_SNAPSHOT',
            f'latest YOLO: {latest_yolo}; /yolo/snapshot unavailable so no drive sequence ran',
        )
        return True

    def perform_lift_sequence(self) -> bool:
        self._set_state('LIFT_PUSH', 'calling /lift_push')
        ok, message = self._call_trigger_service(self.lift_push_client, '/lift_push')
        if not ok:
            self._fail(f'/lift_push failed: {message}')
            return False
        self.get_logger().info(f'/lift_push response: {message}')

        if not self._sleep_with_cancel(float(self.params['lift_action_delay_sec'])):
            return False

        self._set_state('LIFT_RETURN', 'calling /lift_return')
        ok, message = self._call_trigger_service(self.lift_return_client, '/lift_return')
        if not ok:
            self._fail(f'/lift_return failed: {message}')
            return False
        self.get_logger().info(f'/lift_return response: {message}')

        return self._sleep_with_cancel(float(self.params['lift_action_delay_sec']))

    def _call_trigger_service(
        self,
        client,
        service_name: str,
        timeout_sec: float = 5.0,
    ) -> Tuple[bool, str]:
        if self._cancel_requested():
            return False, 'mission canceled'
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'{service_name} service is not available')
            return False, 'service unavailable'

        future = client.call_async(Trigger.Request())
        if not self._wait_for_future(future, timeout_sec=timeout_sec):
            return False, 'service call timed out or canceled'

        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

        if not response.success:
            return False, response.message
        return True, response.message

    def _update_delivered_count(self, container: str, object_class: str) -> bool:
        with self.lock:
            if container not in self.required:
                return False
            if object_class not in self.required[container]:
                return False

            current = self.delivered[container].get(object_class, 0)
            required = self.required[container][object_class]
            if current >= required:
                return False

            self.delivered[container][object_class] = current + 1

        self.publish_status(f'progress updated: {object_class} to {container}')
        return True

    def _all_done(self) -> bool:
        with self.lock:
            if not self.required:
                return False
            for container, objects in self.required.items():
                for object_class, required_count in objects.items():
                    delivered_count = self.delivered.get(container, {}).get(object_class, 0)
                    if delivered_count < required_count:
                        return False
        return True

    def _wait_for_future(self, future, timeout_sec: Optional[float]) -> bool:
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while rclpy.ok():
            if future.done():
                return True
            if self._cancel_requested():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _sleep_with_cancel(self, duration_sec: float) -> bool:
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            if self._cancel_requested():
                return False
            time.sleep(0.05)
        return True

    def _cancel_active_nav_goal(self) -> None:
        with self.lock:
            goal_handle = self.active_nav_goal_handle

        if goal_handle is None:
            return

        try:
            goal_handle.cancel_goal_async()
            self.get_logger().info('Requested active Nav2 goal cancellation')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Failed to cancel Nav2 goal: {exc}')

    def _call_lift_stop_async(self) -> None:
        if not self.lift_stop_client.service_is_ready():
            self.lift_stop_client.wait_for_service(timeout_sec=0.05)
        if not self.lift_stop_client.service_is_ready():
            self.get_logger().debug('/lift_stop service is not available')
            return

        try:
            self.lift_stop_client.call_async(Trigger.Request())
            self.get_logger().info('Requested /lift_stop')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Failed to call /lift_stop: {exc}')

    def _call_motion_stop_async(self) -> None:
        if not self.motion_stop_client.service_is_ready():
            self.motion_stop_client.wait_for_service(timeout_sec=0.05)
        if not self.motion_stop_client.service_is_ready():
            self.get_logger().debug('/tb3_motion/stop service is not available')
            return

        try:
            self.motion_stop_client.call_async(Trigger.Request())
            self.get_logger().info('Requested /tb3_motion/stop')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Failed to call /tb3_motion/stop: {exc}')

    def _safe_stop(self) -> None:
        self._publish_aruco_active(False)
        self._publish_zero_cmd_vel()

    def _publish_aruco_active(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        try:
            self.aruco_active_pub.publish(msg)
        except Exception:
            pass

    def _publish_zero_cmd_vel(self, count: int = 5, period_sec: float = 0.02) -> None:
        for index in range(max(1, count)):
            try:
                self.cmd_vel_pub.publish(Twist())
            except Exception:
                pass
            if period_sec > 0.0 and index < count - 1:
                time.sleep(period_sec)

    def _set_state(
        self,
        state: str,
        message: str = '',
        target_waypoint: Optional[str] = None,
    ) -> None:
        with self.lock:
            self.state = state
            if message:
                self.last_status_message = message
            if target_waypoint is not None:
                self.target_waypoint = target_waypoint

        if message:
            self.get_logger().info(f'{state}: {message}')
        else:
            self.get_logger().info(state)
        self.publish_status()

    def status_timer_callback(self) -> None:
        with self.lock:
            should_publish = self.mission_running
        if should_publish:
            self.publish_status()

    def publish_status(self, message: Optional[str] = None) -> None:
        with self.lock:
            if message is not None:
                self.last_status_message = message

            status = {
                'state': self.state,
                'current_object': self.current_object,
                'target_container': self.target_container,
                'target_waypoint': self.target_waypoint,
                'current_location': self.current_location,
                'mission_running': self.mission_running,
                'debug_running': self.debug_running,
                'active_motion_path': self.active_motion_path,
                'message': self.last_status_message,
                'progress': self._progress_snapshot_locked(),
                'all_done': self._all_done_locked(),
            }

        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        try:
            self.status_pub.publish(msg)
        except Exception:
            pass

    def _progress_snapshot_locked(self) -> dict:
        progress = {}
        for container, objects in self.required.items():
            progress[container] = {}
            for object_class, required_count in objects.items():
                delivered_count = self.delivered.get(container, {}).get(object_class, 0)
                progress[container][object_class] = {
                    'required': required_count,
                    'delivered': delivered_count,
                    'remaining': max(required_count - delivered_count, 0),
                }
        return progress

    def _all_done_locked(self) -> bool:
        if not self.required:
            return False
        for container, objects in self.required.items():
            for object_class, required_count in objects.items():
                delivered_count = self.delivered.get(container, {}).get(object_class, 0)
                if delivered_count < required_count:
                    return False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DeliveryMaster()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node._request_cancel('shutdown')
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
