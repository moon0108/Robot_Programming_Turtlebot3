import json
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:
    QtCore = None
    QtWidgets = None
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None


DEFAULT_CONTAINERS = ['container1', 'container2', 'container3']
DEFAULT_OBJECT_CLASSES = [
    'Fried-chicken-legs',
    'Haute-Cheese',
    'Pepero',
    'choco-heim',
]
DEFAULT_DEBUG_PATHS = [
    'station_to_container1',
    'station_to_container2',
    'station_to_container3',
    'container1_to_station',
    'container2_to_station',
    'container3_to_station',
    'test_forward_back',
]


class QtOrderNode(Node):
    def __init__(self, status_bridge):
        super().__init__('qt_order_node')
        self.status_bridge = status_bridge

        self.declare_parameter('containers', DEFAULT_CONTAINERS)
        self.declare_parameter('object_classes', DEFAULT_OBJECT_CLASSES)

        self.containers = self._string_list_parameter(
            'containers',
            DEFAULT_CONTAINERS,
        )
        self.object_classes = self._string_list_parameter(
            'object_classes',
            DEFAULT_OBJECT_CLASSES,
        )
        self.latest_yolo_best = None
        self.latest_yolo_detections = None
        self.latest_detected_classes = []
        self.latest_motion_status = None
        self.latest_aruco_status = None

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

        self.start_client = self.create_client(Trigger, '/delivery/start')
        self.cancel_client = self.create_client(Trigger, '/delivery/cancel')
        self.reset_client = self.create_client(Trigger, '/delivery/reset')
        self.motion_reload_client = self.create_client(Trigger, '/tb3_motion/reload')
        self.motion_stop_client = self.create_client(Trigger, '/tb3_motion/stop')
        self.lift_push_client = self.create_client(Trigger, '/lift_push')
        self.lift_return_client = self.create_client(Trigger, '/lift_return')
        self.lift_stop_client = self.create_client(Trigger, '/lift_stop')

    def status_callback(self, msg: String) -> None:
        text = msg.data
        try:
            text = json.dumps(json.loads(msg.data), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
        self.status_bridge.status_received.emit(text)

    def yolo_best_callback(self, msg: String) -> None:
        self.latest_yolo_best = self._parse_json_or_text(msg.data)

    def yolo_detections_callback(self, msg: String) -> None:
        self.latest_yolo_detections = self._parse_json_or_text(msg.data)

    def _string_list_parameter(self, name: str, fallback) -> list:
        value = self.get_parameter(name).value
        if value is None:
            return list(fallback)
        if isinstance(value, str):
            items = [item.strip() for item in value.split(',')]
        else:
            items = [str(item).strip() for item in value]
        items = [item for item in items if item]
        return items or list(fallback)

    def available_classes_callback(self, msg: String) -> None:
        classes = self._parse_class_list(msg.data)
        if not classes:
            return
        self.object_classes = classes
        self.status_bridge.object_classes_received.emit(list(self.object_classes))

    def detected_classes_callback(self, msg: String) -> None:
        classes = self._parse_class_list(msg.data)
        if not classes:
            return
        self.latest_detected_classes = classes
        changed = False
        for class_name in classes:
            if class_name not in self.object_classes:
                self.object_classes.append(class_name)
                changed = True
        if changed:
            self.status_bridge.object_classes_received.emit(list(self.object_classes))

    def motion_status_callback(self, msg: String) -> None:
        self.latest_motion_status = self._parse_json_or_text(msg.data)

    def aruco_status_callback(self, msg: String) -> None:
        self.latest_aruco_status = self._parse_json_or_text(msg.data)

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

    def publish_order_dict(self, order: dict) -> None:
        msg = String()
        msg.data = json.dumps(order, ensure_ascii=False)
        self.order_pub.publish(msg)
        self.get_logger().info(f'Published order: {msg.data}')
        self.status_bridge.status_received.emit('Order published.')

    def publish_order(self, order_text: str) -> None:
        try:
            order = json.loads(order_text)
        except json.JSONDecodeError as exc:
            self.status_bridge.status_received.emit(f'Invalid order JSON: {exc}')
            return

        msg = String()
        msg.data = json.dumps(order)
        self.order_pub.publish(msg)
        self.get_logger().info(f'Published order: {msg.data}')
        self.status_bridge.status_received.emit('Order published.')

    def publish_emergency_stop(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self.emergency_pub.publish(msg)
        label = 'ON' if active else 'OFF'
        self.get_logger().warn(f'Emergency Stop {label}')
        self.status_bridge.status_received.emit(f'Emergency Stop {label} published.')

    def call_trigger_service(self, client, service_name: str) -> None:
        if not client.wait_for_service(timeout_sec=0.2):
            self.status_bridge.status_received.emit(f'{service_name} is not available.')
            return

        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda done_future: self._trigger_done(done_future, service_name)
        )
        self.status_bridge.status_received.emit(f'Calling {service_name}...')

    def _trigger_done(self, future, service_name: str) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.status_bridge.status_received.emit(f'{service_name} failed: {exc}')
            return

        self.status_bridge.status_received.emit(
            f'{service_name}: success={response.success}, message="{response.message}"'
        )

    def publish_debug_step(self, step: str, **fields) -> None:
        payload = {'step': step}
        payload.update(fields)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_step_pub.publish(msg)
        self.status_bridge.status_received.emit(
            f'Debug step requested: {json.dumps(payload, ensure_ascii=False)}'
        )

    def run_motion_path(self, path_name: str) -> None:
        path_name = path_name.strip()
        if not path_name:
            self.status_bridge.status_received.emit('Motion path is empty.')
            return
        self.publish_debug_step('motion_path', path=path_name)

    def stop_motion(self) -> None:
        self.stop_all_debug()

    def set_aruco_alignment(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self.aruco_active_pub.publish(msg)
        label = 'ON' if active else 'OFF'
        self.status_bridge.status_received.emit(f'ArUco alignment {label} published.')

    def show_yolo_snapshot(self) -> None:
        self.publish_debug_step(
            'yolo_snapshot',
            linear_x=0.1,
            duration_sec=1.0,
        )
        self.status_bridge.status_received.emit(
            'YOLO snapshot requested. Detection result will arrive from delivery_master.'
        )

    def show_debug_status(self) -> None:
        snapshot = {
            'motion_status': self.latest_motion_status,
            'aruco_status': self.latest_aruco_status,
            'yolo_best_object': self.latest_yolo_best,
        }
        self.status_bridge.status_received.emit(
            json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def publish_zero_cmd_vel(self) -> None:
        for _ in range(5):
            self.cmd_vel_pub.publish(Twist())

    def stop_all_debug(self) -> None:
        self.set_aruco_alignment(False)
        self.publish_zero_cmd_vel()
        if self.motion_stop_client.service_is_ready():
            self.motion_stop_client.call_async(Trigger.Request())
        if self.lift_stop_client.service_is_ready():
            self.lift_stop_client.call_async(Trigger.Request())
        self.status_bridge.status_received.emit(
            'Debug stop requested: ArUco off, cmd_vel zero, motion stop, lift stop.'
        )


if QtWidgets is not None:

    class StatusBridge(QtCore.QObject):
        status_received = QtCore.pyqtSignal(str)
        object_classes_received = QtCore.pyqtSignal(object)


    class OrderWindow(QtWidgets.QWidget):
        def __init__(self, ros_node: QtOrderNode, status_bridge: StatusBridge):
            super().__init__()
            self.ros_node = ros_node
            self.setWindowTitle('TB3 Delivery Order')
            self.setMinimumSize(760, 720)
            self.object_classes = list(self.ros_node.object_classes)
            self.quantity_inputs = {}

            layout = QtWidgets.QVBoxLayout(self)

            self.order_group = QtWidgets.QGroupBox('Order')
            self.order_layout = QtWidgets.QGridLayout(self.order_group)
            self.order_layout.setContentsMargins(10, 10, 10, 10)
            self.order_layout.setSpacing(8)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.order_group)
            layout.addWidget(scroll, stretch=2)

            button_grid = QtWidgets.QGridLayout()
            self.publish_button = QtWidgets.QPushButton('Publish Order')
            self.start_button = QtWidgets.QPushButton('Start Mission')
            self.cancel_button = QtWidgets.QPushButton('Cancel Mission')
            self.reset_button = QtWidgets.QPushButton('Reset Mission')
            self.estop_on_button = QtWidgets.QPushButton('Emergency Stop ON')
            self.estop_off_button = QtWidgets.QPushButton('Emergency Stop OFF')

            button_grid.addWidget(self.publish_button, 0, 0)
            button_grid.addWidget(self.start_button, 0, 1)
            button_grid.addWidget(self.cancel_button, 1, 0)
            button_grid.addWidget(self.reset_button, 1, 1)
            button_grid.addWidget(self.estop_on_button, 2, 0)
            button_grid.addWidget(self.estop_off_button, 2, 1)
            layout.addLayout(button_grid)
            layout.addWidget(self.build_debug_group())

            status_label = QtWidgets.QLabel('Status')
            self.status_edit = QtWidgets.QTextEdit()
            self.status_edit.setReadOnly(True)
            layout.addWidget(status_label)
            layout.addWidget(self.status_edit, stretch=2)

            self.rebuild_order_inputs()
            self.publish_button.clicked.connect(self.publish_order)
            self.start_button.clicked.connect(
                lambda: self.ros_node.call_trigger_service(
                    self.ros_node.start_client,
                    '/delivery/start',
                )
            )
            self.cancel_button.clicked.connect(
                lambda: self.ros_node.call_trigger_service(
                    self.ros_node.cancel_client,
                    '/delivery/cancel',
                )
            )
            self.reset_button.clicked.connect(
                lambda: self.ros_node.call_trigger_service(
                    self.ros_node.reset_client,
                    '/delivery/reset',
                )
            )
            self.estop_on_button.clicked.connect(
                lambda: self.ros_node.publish_emergency_stop(True)
            )
            self.estop_off_button.clicked.connect(
                lambda: self.ros_node.publish_emergency_stop(False)
            )
            status_bridge.status_received.connect(self.set_status)
            status_bridge.object_classes_received.connect(self.set_object_classes)

        def build_debug_group(self):
            group = QtWidgets.QGroupBox('Debug Steps')
            layout = QtWidgets.QGridLayout(group)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            self.debug_path_combo = QtWidgets.QComboBox()
            self.debug_path_combo.setEditable(True)
            self.debug_path_combo.addItems(DEFAULT_DEBUG_PATHS)

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

        def publish_order(self) -> None:
            order = self.current_order()
            if not order:
                self.set_status('No positive order quantities.')
                return
            self.ros_node.publish_order_dict(order)

        def current_order(self) -> dict:
            order = {}
            for container, inputs in self.quantity_inputs.items():
                items = {}
                for object_class, spin_box in inputs.items():
                    quantity = spin_box.value()
                    if quantity > 0:
                        items[object_class] = quantity
                if items:
                    order[container] = items
            return order

        def set_object_classes(self, object_classes) -> None:
            object_classes = [str(item).strip() for item in object_classes if str(item).strip()]
            if not object_classes or object_classes == self.object_classes:
                return
            previous_order = self.current_order()
            self.object_classes = object_classes
            self.rebuild_order_inputs(previous_order)

        def set_status(self, text: str) -> None:
            self.status_edit.setPlainText(text)

        def rebuild_order_inputs(self, previous_order=None) -> None:
            previous_order = previous_order or self.current_order()
            while self.order_layout.count():
                item = self.order_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            self.quantity_inputs = {}
            self.order_layout.addWidget(QtWidgets.QLabel('Container'), 0, 0)
            for column, object_class in enumerate(self.object_classes, start=1):
                label = QtWidgets.QLabel(object_class)
                label.setAlignment(QtCore.Qt.AlignCenter)
                self.order_layout.addWidget(label, 0, column)

            for row, container in enumerate(self.ros_node.containers, start=1):
                self.order_layout.addWidget(QtWidgets.QLabel(container), row, 0)
                self.quantity_inputs[container] = {}
                for column, object_class in enumerate(self.object_classes, start=1):
                    quantity = QtWidgets.QSpinBox()
                    quantity.setRange(0, 99)
                    quantity.setValue(previous_order.get(container, {}).get(object_class, 0))
                    self.quantity_inputs[container][object_class] = quantity
                    self.order_layout.addWidget(quantity, row, column)


def main(args=None):
    if QtWidgets is None:
        print(
            'PyQt5 is required for qt_order_node but could not be imported: '
            f'{QT_IMPORT_ERROR}'
        )
        return

    rclpy.init(args=args)
    app = QtWidgets.QApplication(sys.argv)
    status_bridge = StatusBridge()
    node = QtOrderNode(status_bridge)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    window = OrderWindow(node, status_bridge)
    window.show()

    try:
        exit_code = app.exec_()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
