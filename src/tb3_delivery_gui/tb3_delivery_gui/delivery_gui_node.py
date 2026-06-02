import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from python_qt_binding import QtCore, QtGui, QtWidgets
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tb3_delivery_interfaces.action import ContainerStatus


DEFAULT_CONTAINERS = ('container1', 'container2', 'container3')
DEFAULT_OBJECT_CLASSES = ('box',)
DEFAULT_DEBUG_PATHS = (
    'station_to_container1',
    'station_to_container2',
    'station_to_container3',
    'container1_to_station',
    'container2_to_station',
    'container3_to_station',
    'test_forward_back',
)


class DeliveryGuiRos(Node):
    def __init__(self):
        super().__init__('delivery_gui_node')
        self.declare_parameter('aruco_image_topic', '/aruco/annotated_image')
        self.declare_parameter('yolo_image_topic', '/yolo/image')

        self.status_text = 'Ready.'
        self.last_state = ''
        self.aruco_image_topic = str(self.get_parameter('aruco_image_topic').value)
        self.yolo_image_topic = str(self.get_parameter('yolo_image_topic').value)
        self.object_classes = list(DEFAULT_OBJECT_CLASSES)
        self.class_update_callbacks = []
        self.latest_yolo_best = None
        self.latest_yolo_detections = None
        self.latest_detected_classes = []
        self.latest_motion_status = None
        self.latest_aruco_status = None
        self.latest_container_action_status = None
        self.container_status_goal_handle = None
        self.container_status_result_future = None
        self.latest_aruco_image = None
        self.latest_yolo_image = None
        self._image_error_logged = set()
        self.bridge = CvBridge()
        self.orders_by_container = {
            container: {}
            for container in DEFAULT_CONTAINERS
        }

        self.order_pub = self.create_publisher(String, '/delivery/order', 10)
        self.emergency_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.aruco_active_pub = self.create_publisher(Bool, '/aruco_align/active', 10)
        self.motion_command_pub = self.create_publisher(
            String,
            '/tb3_motion/command',
            10,
        )
        self.debug_step_pub = self.create_publisher(String, '/delivery/debug_step', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(String, '/delivery/status', self.status_callback, 10)
        self.create_subscription(String, '/yolo/best_object', self.yolo_best_callback, 10)
        self.create_subscription(
            String,
            '/yolo/detections',
            self.yolo_detections_callback,
            10,
        )
        self.create_subscription(
            String,
            '/yolo/available_classes',
            self.available_classes_callback,
            10,
        )
        self.create_subscription(
            String,
            '/yolo/classes',
            self.detected_classes_callback,
            10,
        )
        self.create_subscription(
            String,
            '/tb3_motion/status',
            self.motion_status_callback,
            10,
        )
        self.create_subscription(
            String,
            '/aruco_align/status',
            self.aruco_status_callback,
            10,
        )
        self.create_subscription(
            Image,
            self.aruco_image_topic,
            self.aruco_image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.yolo_image_topic,
            self.yolo_image_callback,
            qos_profile_sensor_data,
        )

        self.start_client = self.create_client(Trigger, '/delivery/start')
        self.cancel_client = self.create_client(Trigger, '/delivery/cancel')
        self.reset_client = self.create_client(Trigger, '/delivery/reset')
        self.motion_reload_client = self.create_client(Trigger, '/tb3_motion/reload')
        self.motion_stop_client = self.create_client(Trigger, '/tb3_motion/stop')
        self.lift_push_client = self.create_client(Trigger, '/lift_push')
        self.lift_return_client = self.create_client(Trigger, '/lift_return')
        self.lift_stop_client = self.create_client(Trigger, '/lift_stop')
        self.container_status_client = ActionClient(
            self,
            ContainerStatus,
            '/container_status',
        )
        self.get_logger().info(f'Subscribed ArUco image: {self.aruco_image_topic}')
        self.get_logger().info(f'Subscribed YOLO image: {self.yolo_image_topic}')

    def add_class_update_callback(self, callback) -> None:
        self.class_update_callbacks.append(callback)

    def status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.status_text = msg.data
            return

        self.last_state = str(payload.get('state', ''))
        self.status_text = json.dumps(payload, indent=2, sort_keys=True)

    def yolo_best_callback(self, msg: String) -> None:
        self.latest_yolo_best = self._parse_json_or_text(msg.data)

    def yolo_detections_callback(self, msg: String) -> None:
        self.latest_yolo_detections = self._parse_json_or_text(msg.data)

    def available_classes_callback(self, msg: String) -> None:
        classes = self._parse_class_list(msg.data)
        if classes:
            self._replace_object_classes(classes)

    def detected_classes_callback(self, msg: String) -> None:
        classes = self._parse_class_list(msg.data)
        if classes:
            self.latest_detected_classes = classes
            self._merge_object_classes(classes)

    def motion_status_callback(self, msg: String) -> None:
        self.latest_motion_status = self._parse_json_or_text(msg.data)

    def aruco_status_callback(self, msg: String) -> None:
        self.latest_aruco_status = self._parse_json_or_text(msg.data)

    def aruco_image_callback(self, msg: Image) -> None:
        image = self._image_msg_to_qimage(msg, 'ArUco')
        if image is not None:
            self.latest_aruco_image = image

    def yolo_image_callback(self, msg: Image) -> None:
        image = self._image_msg_to_qimage(msg, 'YOLO')
        if image is not None:
            self.latest_yolo_image = image

    def _image_msg_to_qimage(self, msg: Image, source_name: str):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: BLE001
            if source_name not in self._image_error_logged:
                self.get_logger().warn(f'{source_name} image conversion failed: {exc}')
                self._image_error_logged.add(source_name)
            return None

        if frame.ndim != 3 or frame.shape[2] != 3:
            if source_name not in self._image_error_logged:
                self.get_logger().warn(
                    f'{source_name} image has unsupported shape: {frame.shape}'
                )
                self._image_error_logged.add(source_name)
            return None

        height, width, channels = frame.shape
        bytes_per_line = channels * width
        return QtGui.QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,
        ).copy()

    def _parse_json_or_text(self, data: str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data

    def _parse_class_list(self, data: str) -> list:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = [data]

        if isinstance(payload, dict):
            payload = (
                payload.get('classes')
                or payload.get('class_names')
                or payload.get('labels')
                or []
            )
        if isinstance(payload, str):
            payload = [payload]
        if not isinstance(payload, list):
            return []

        classes = []
        for item in payload:
            class_name = str(item).strip()
            if class_name and class_name not in classes:
                classes.append(class_name)
        return classes

    def _replace_object_classes(self, classes: list) -> None:
        normalized = [str(item).strip() for item in classes if str(item).strip()]
        if not normalized or normalized == self.object_classes:
            return
        self.object_classes = normalized
        self._notify_object_classes()

    def _merge_object_classes(self, classes: list) -> None:
        changed = False
        for class_name in classes:
            class_name = str(class_name).strip()
            if class_name and class_name not in self.object_classes:
                self.object_classes.append(class_name)
                changed = True
        if changed:
            self._notify_object_classes()

    def _notify_object_classes(self) -> None:
        for callback in self.class_update_callbacks:
            callback(list(self.object_classes))

    def set_orders(self, quantities, target_container):
        self.orders_by_container[target_container] = {
            object_class: quantity
            for object_class, quantity in quantities.items()
            if quantity > 0
        }

        if not self.orders_by_container[target_container]:
            self.status_text = f'{target_container} order cleared.'
            return

        self.status_text = f'{target_container} order saved.'

    def get_order_summary(self, target_container):
        orders = self.orders_by_container.get(target_container, {})
        if not orders:
            return f'{target_container}: no order'

        lines = [f'{target_container} orders']
        for object_class in self.object_classes:
            quantity = orders.get(object_class, 0)
            if quantity > 0:
                lines.append(f'{object_class}: {quantity}')
        return '\n'.join(lines)

    def build_order_payload(self) -> dict:
        order = {}
        for container in DEFAULT_CONTAINERS:
            items = {}
            orders = self.orders_by_container.get(container, {})
            for object_class in self.object_classes:
                quantity = int(orders.get(object_class, 0))
                if quantity > 0:
                    items[object_class] = quantity
            if items:
                order[container] = items
        return order

    def publish_orders(self) -> bool:
        order = self.build_order_payload()
        if not order:
            self.status_text = 'No orders saved.'
            return False

        msg = String()
        msg.data = json.dumps(order, ensure_ascii=False)
        self.order_pub.publish(msg)
        self.get_logger().info(f'Published order: {msg.data}')
        self.status_text = 'Order published. Starting mission...'
        return True

    def start_container_status_action(self) -> None:
        if self.container_status_goal_handle is not None:
            self.status_text = 'Container status action is already running.'
            return
        if not self.container_status_client.wait_for_server(timeout_sec=0.2):
            self.status_text = '/container_status action server is not available.'
            return

        goal = ContainerStatus.Goal()
        goal.feedback_period_sec = 10.0
        future = self.container_status_client.send_goal_async(
            goal,
            feedback_callback=self.container_status_feedback_callback,
        )
        future.add_done_callback(self.container_status_goal_response_callback)
        self.status_text = 'Container status action requested.'

    def cancel_container_status_action(self) -> None:
        goal_handle = self.container_status_goal_handle
        if goal_handle is None:
            self.status_text = 'No container status action is running.'
            return
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda _future: setattr(self, 'status_text', 'Container status cancel requested.')
        )

    def container_status_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.container_status_goal_handle = None
            self.status_text = 'Container status action rejected.'
            return

        self.container_status_goal_handle = goal_handle
        self.container_status_result_future = goal_handle.get_result_async()
        self.container_status_result_future.add_done_callback(
            self.container_status_result_callback
        )
        self.status_text = 'Container status action accepted.'

    def container_status_feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        payload = self._parse_json_or_text(feedback.current_status_json)
        self.latest_container_action_status = payload

    def container_status_result_callback(self, future) -> None:
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as exc:  # noqa: BLE001
            self.status_text = f'Container status action failed: {exc}'
            result = None

        self.container_status_goal_handle = None
        self.container_status_result_future = None

        if result is None:
            return

        self.latest_container_action_status = self._parse_json_or_text(
            result.final_status_json
        )
        self.status_text = (
            f'Container status action done: '
            f'all_filled={result.all_filled}, message="{result.message}"'
        )

    def start_mission(self) -> None:
        self.call_trigger_service(self.start_client, '/delivery/start')

    def cancel_mission(self) -> None:
        self.call_trigger_service(self.cancel_client, '/delivery/cancel')

    def reset_mission(self) -> None:
        self.call_trigger_service(self.reset_client, '/delivery/reset')

    def call_trigger_service(self, client, service_name: str) -> None:
        if not client.wait_for_service(timeout_sec=0.2):
            self.status_text = f'{service_name} is not available.'
            return

        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda done_future: self._trigger_done(done_future, service_name)
        )
        self.status_text = f'Calling {service_name}...'

    def _trigger_done(self, future, service_name: str) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.status_text = f'{service_name} failed: {exc}'
            return

        self.status_text = (
            f'{service_name}: success={response.success}, message="{response.message}"'
        )

    def publish_debug_step(self, step: str, **fields) -> None:
        payload = {'step': step}
        payload.update(fields)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_step_pub.publish(msg)
        self.status_text = f'Debug step requested: {json.dumps(payload, ensure_ascii=False)}'

    def run_motion_path(self, path_name: str) -> None:
        path_name = path_name.strip()
        if not path_name:
            self.status_text = 'Motion path is empty.'
            return
        self.publish_debug_step('motion_path', path=path_name)

    def stop_motion(self) -> None:
        self.stop_all_debug()

    def set_aruco_alignment(self, active: bool) -> None:
        if active:
            self.publish_debug_step('aruco_align')
            return
        self.publish_debug_step('stop_all')
        msg = Bool()
        msg.data = active
        self.aruco_active_pub.publish(msg)
        label = 'ON' if active else 'OFF'
        self.status_text = f'ArUco alignment {label} published.'
        if not active:
            self.publish_zero_cmd_vel()

    def show_yolo_snapshot(self) -> None:
        snapshot = {
            'best_object': self.latest_yolo_best,
            'detections': self.latest_yolo_detections,
            'detected_classes': self.latest_detected_classes,
            'available_classes': self.object_classes,
        }
        self.publish_debug_step('yolo_snapshot')
        self.status_text = json.dumps(
            snapshot,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    def show_debug_status(self) -> None:
        snapshot = {
            'motion_status': self.latest_motion_status,
            'aruco_status': self.latest_aruco_status,
            'yolo_best_object': self.latest_yolo_best,
            'container_action_status': self.latest_container_action_status,
        }
        self.status_text = json.dumps(
            snapshot,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    def publish_zero_cmd_vel(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def stop_all_debug(self) -> None:
        self.set_aruco_alignment(False)
        self.publish_zero_cmd_vel()
        if self.motion_stop_client.service_is_ready():
            self.motion_stop_client.call_async(Trigger.Request())
        if self.lift_stop_client.service_is_ready():
            self.lift_stop_client.call_async(Trigger.Request())
        self.status_text = (
            'Debug stop requested: ArUco off, cmd_vel zero, motion stop, lift stop.'
        )

    def publish_emergency_stop(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self.emergency_pub.publish(msg)
        label = 'ON' if active else 'OFF'
        self.get_logger().warn(f'Emergency Stop {label}')
        self.status_text = f'Emergency Stop {label} published.'


class DeliveryWindow(QtWidgets.QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle('TB3 Delivery Control')
        self.setMinimumSize(1120, 640)

        root_layout = QtWidgets.QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        controls = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(controls, 3)

        self.object_group = QtWidgets.QGroupBox('Objects')
        self.object_layout = QtWidgets.QGridLayout(self.object_group)
        self.quantity_inputs = {}
        layout.addWidget(self.object_group)

        form = QtWidgets.QFormLayout()
        self.target_container = QtWidgets.QComboBox()
        self.target_container.addItems(list(DEFAULT_CONTAINERS))
        form.addRow('Container', self.target_container)
        layout.addLayout(form)

        set_order = QtWidgets.QPushButton('Set Order')
        set_order.clicked.connect(self.set_order)
        layout.addWidget(set_order)

        self.order_summary = QtWidgets.QLabel('')
        self.order_summary.setObjectName('orderSummary')
        self.order_summary.setWordWrap(True)
        self.order_summary.setMinimumHeight(76)
        layout.addWidget(self.order_summary)

        self.container_action_summary = QtWidgets.QLabel('Container action: no feedback')
        self.container_action_summary.setObjectName('containerActionSummary')
        self.container_action_summary.setWordWrap(True)
        self.container_action_summary.setMinimumHeight(58)
        layout.addWidget(self.container_action_summary)

        button_grid = QtWidgets.QGridLayout()
        order_button = QtWidgets.QPushButton('Publish and Start')
        cancel_button = QtWidgets.QPushButton('Cancel')
        reset_button = QtWidgets.QPushButton('Reset')
        estop_on_button = QtWidgets.QPushButton('Emergency Stop ON')
        estop_off_button = QtWidgets.QPushButton('Emergency Stop OFF')

        order_button.clicked.connect(self.start_order)
        cancel_button.clicked.connect(self.ros_node.cancel_mission)
        reset_button.clicked.connect(self.ros_node.reset_mission)
        estop_on_button.clicked.connect(lambda: self.ros_node.publish_emergency_stop(True))
        estop_off_button.clicked.connect(lambda: self.ros_node.publish_emergency_stop(False))

        button_grid.addWidget(order_button, 0, 0)
        button_grid.addWidget(cancel_button, 0, 1)
        button_grid.addWidget(reset_button, 1, 0)
        button_grid.addWidget(estop_on_button, 1, 1)
        button_grid.addWidget(estop_off_button, 2, 0, 1, 2)
        layout.addLayout(button_grid)
        layout.addWidget(self.build_debug_group())

        self.status = QtWidgets.QLabel('Ready.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        root_layout.addWidget(self.build_camera_group(), 2)

        self.target_container.currentTextChanged.connect(
            lambda _text: self.update_order_summary()
        )
        self.target_container.currentTextChanged.connect(
            lambda _text: self.load_container_quantities()
        )
        self.ros_node.add_class_update_callback(self.set_object_classes)

        self.setStyleSheet("""
            QWidget { font-size: 15px; }
            QPushButton { min-height: 36px; font-weight: 600; }
            QComboBox, QSpinBox { min-height: 30px; }
            QGroupBox {
                font-weight: 700;
                border: 1px solid #9aa0a6;
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLabel#orderSummary {
                border: 1px solid #c5c9cf;
                border-radius: 6px;
                padding: 8px;
                background: #f7f8fa;
                font-weight: 500;
            }
            QLabel#containerActionSummary {
                border: 1px solid #c5c9cf;
                border-radius: 6px;
                padding: 8px;
                background: #eef4ff;
                font-weight: 500;
            }
            QLabel#cameraView {
                border: 1px solid #9aa0a6;
                border-radius: 6px;
                background: #202124;
                color: #f1f3f4;
                font-weight: 600;
            }
        """)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(100)
        self.rebuild_object_inputs()
        self.update_order_summary()
        self.load_container_quantities()

    def build_camera_group(self):
        group = QtWidgets.QGroupBox('Camera Views')
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.aruco_image_label = self.make_camera_label(
            f'Waiting for {self.ros_node.aruco_image_topic}'
        )
        self.yolo_image_label = self.make_camera_label(
            f'Waiting for {self.ros_node.yolo_image_topic}'
        )

        layout.addWidget(QtWidgets.QLabel('ArUco Marker'))
        layout.addWidget(self.aruco_image_label, 1)
        layout.addWidget(QtWidgets.QLabel('YOLO Object'))
        layout.addWidget(self.yolo_image_label, 1)
        return group

    def make_camera_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName('cameraView')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(320, 220)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        label.setWordWrap(True)
        return label

    def build_debug_group(self):
        group = QtWidgets.QGroupBox('Debug Steps')
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.debug_path_combo = QtWidgets.QComboBox()
        self.debug_path_combo.setEditable(True)
        self.debug_path_combo.addItems(list(DEFAULT_DEBUG_PATHS))

        run_path_button = QtWidgets.QPushButton('Run Motion Path')
        stop_motion_button = QtWidgets.QPushButton('Stop Motion')
        aruco_on_button = QtWidgets.QPushButton('Run ArUco Align')
        aruco_off_button = QtWidgets.QPushButton('ArUco Align OFF')
        yolo_snapshot_button = QtWidgets.QPushButton('YOLO Snapshot')
        debug_status_button = QtWidgets.QPushButton('Debug Status')
        lift_push_button = QtWidgets.QPushButton('Lift Push')
        lift_return_button = QtWidgets.QPushButton('Lift Return')
        lift_stop_button = QtWidgets.QPushButton('Lift Stop')
        stop_all_button = QtWidgets.QPushButton('Stop All Debug')

        layout.addWidget(QtWidgets.QLabel('Motion path'), 0, 0)
        layout.addWidget(self.debug_path_combo, 0, 1)
        layout.addWidget(run_path_button, 0, 2)
        layout.addWidget(stop_motion_button, 0, 3)

        layout.addWidget(aruco_on_button, 1, 0)
        layout.addWidget(aruco_off_button, 1, 1)
        layout.addWidget(yolo_snapshot_button, 1, 2)
        layout.addWidget(debug_status_button, 1, 3)

        layout.addWidget(lift_push_button, 2, 0)
        layout.addWidget(lift_return_button, 2, 1)
        layout.addWidget(lift_stop_button, 2, 2)
        layout.addWidget(stop_all_button, 2, 3)

        run_path_button.clicked.connect(
            lambda: self.ros_node.run_motion_path(self.debug_path_combo.currentText())
        )
        stop_motion_button.clicked.connect(self.ros_node.stop_motion)
        aruco_on_button.clicked.connect(lambda: self.ros_node.set_aruco_alignment(True))
        aruco_off_button.clicked.connect(lambda: self.ros_node.set_aruco_alignment(False))
        yolo_snapshot_button.clicked.connect(self.ros_node.show_yolo_snapshot)
        debug_status_button.clicked.connect(self.ros_node.show_debug_status)
        lift_push_button.clicked.connect(
            lambda: self.ros_node.publish_debug_step('lift_push')
        )
        lift_return_button.clicked.connect(
            lambda: self.ros_node.publish_debug_step('lift_return')
        )
        lift_stop_button.clicked.connect(
            lambda: self.ros_node.publish_debug_step('lift_stop')
        )
        stop_all_button.clicked.connect(self.ros_node.stop_all_debug)
        return group

    def set_object_classes(self, object_classes) -> None:
        del object_classes
        previous_quantities = {
            name: spin_box.value()
            for name, spin_box in self.quantity_inputs.items()
        }
        self.rebuild_object_inputs(previous_quantities)
        self.load_container_quantities()
        self.update_order_summary()

    def rebuild_object_inputs(self, previous_quantities=None) -> None:
        previous_quantities = previous_quantities or {}
        while self.object_layout.count():
            item = self.object_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.quantity_inputs = {}
        self.object_layout.addWidget(QtWidgets.QLabel('Object'), 0, 0)
        self.object_layout.addWidget(QtWidgets.QLabel('Quantity'), 0, 1)

        for row, object_name in enumerate(self.ros_node.object_classes, start=1):
            label = QtWidgets.QLabel(object_name)
            quantity = QtWidgets.QSpinBox()
            quantity.setMinimum(0)
            quantity.setMaximum(99)
            quantity.setValue(previous_quantities.get(object_name, 0))
            self.quantity_inputs[object_name] = quantity
            self.object_layout.addWidget(label, row, 0)
            self.object_layout.addWidget(quantity, row, 1)

    def set_order(self):
        quantities = {
            name: spin_box.value()
            for name, spin_box in self.quantity_inputs.items()
        }
        self.ros_node.set_orders(quantities, self.target_container.currentText())
        self.update_order_summary()

    def start_order(self):
        self.set_order()
        if self.ros_node.publish_orders():
            self.ros_node.start_container_status_action()
            QtCore.QTimer.singleShot(300, self.ros_node.start_mission)

    def update_order_summary(self):
        self.order_summary.setText(
            self.ros_node.get_order_summary(self.target_container.currentText())
        )

    def load_container_quantities(self):
        orders = self.ros_node.orders_by_container.get(self.target_container.currentText(), {})
        for object_name, spin_box in self.quantity_inputs.items():
            spin_box.blockSignals(True)
            spin_box.setValue(orders.get(object_name, 0))
            spin_box.blockSignals(False)

    def refresh(self):
        self.status.setText(self.ros_node.status_text)
        self.container_action_summary.setText(self.container_action_summary_text())
        self.update_camera_label(
            self.aruco_image_label,
            self.ros_node.latest_aruco_image,
            f'Waiting for {self.ros_node.aruco_image_topic}',
        )
        self.update_camera_label(
            self.yolo_image_label,
            self.ros_node.latest_yolo_image,
            f'Waiting for {self.ros_node.yolo_image_topic}',
        )

    def container_action_summary_text(self):
        status = self.ros_node.latest_container_action_status
        if not isinstance(status, dict):
            return 'Container action: no feedback'

        progress = float(status.get('progress', 0.0)) * 100.0
        all_filled = bool(status.get('all_filled', False))
        containers = status.get('containers', {})
        filled = []
        for name, container_status in containers.items():
            if isinstance(container_status, dict):
                mark = 'done' if container_status.get('filled', False) else 'open'
                remaining = container_status.get('remaining_total', 0)
                filled.append(f'{name}: {mark}, remaining {remaining}')

        lines = [
            f'Container action: {progress:.0f}% all_filled={all_filled}',
        ]
        if filled:
            lines.extend(filled)
        else:
            lines.append('waiting for order')
        return '\n'.join(lines)

    def update_camera_label(self, label, image, waiting_text):
        if image is None:
            label.setText(waiting_text)
            return

        pixmap = QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)


def main(args=None):
    rclpy.init(args=args)
    app = QtWidgets.QApplication([])
    node = DeliveryGuiRos()
    window = DeliveryWindow(node)
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
