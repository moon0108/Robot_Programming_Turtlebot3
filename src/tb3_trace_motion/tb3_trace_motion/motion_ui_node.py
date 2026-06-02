import json
import os
import sys
import threading

import yaml

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tb3_trace_motion.motion_executor_node import DEFAULT_PATHS

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:
    QtCore = None
    QtWidgets = None
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None


class MotionUiNode(Node):
    def __init__(self, status_bridge):
        super().__init__('tb3_motion_ui')
        self.status_bridge = status_bridge

        self.declare_parameter(
            'motion_file',
            '/root/maps/tb3_motion_paths.yaml',
        )
        self.motion_file = self.get_parameter('motion_file').value

        self.command_pub = self.create_publisher(
            String,
            '/tb3_motion/command',
            10,
        )
        self.create_subscription(
            String,
            '/tb3_motion/status',
            self.status_callback,
            10,
        )

        self.stop_client = self.create_client(Trigger, '/tb3_motion/stop')
        self.reload_client = self.create_client(Trigger, '/tb3_motion/reload')

    def status_callback(self, msg: String) -> None:
        text = msg.data
        try:
            text = json.dumps(json.loads(msg.data), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
        self.status_bridge.status_received.emit(text)

    def publish_command(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload)
        self.command_pub.publish(msg)
        self.get_logger().info(f'Published command: {msg.data}')

    def call_trigger(self, client, service_name: str) -> None:
        if not client.wait_for_service(timeout_sec=0.2):
            self.status_bridge.status_received.emit(
                f'{service_name} unavailable'
            )
            return

        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda done_future: self._trigger_done(done_future, service_name)
        )

    def _trigger_done(self, future, service_name: str) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.status_bridge.status_received.emit(
                f'{service_name} failed: {exc}'
            )
            return

        self.status_bridge.status_received.emit(
            (
                f'{service_name}: success={response.success}, '
                f'message={response.message}'
            )
        )


if QtWidgets is not None:

    class StatusBridge(QtCore.QObject):
        status_received = QtCore.pyqtSignal(str)

    class MotionWindow(QtWidgets.QWidget):
        STEP_TYPES = ('move', 'rotate', 'wait')

        def __init__(
            self,
            ros_node: MotionUiNode,
            status_bridge: StatusBridge,
        ):
            super().__init__()
            self.ros_node = ros_node
            self.paths = {}
            self.loading_table = False

            self.setWindowTitle('TB3 Trace Motion')
            self.setMinimumSize(760, 560)

            main_layout = QtWidgets.QVBoxLayout(self)

            file_row = QtWidgets.QHBoxLayout()
            file_row.addWidget(QtWidgets.QLabel('Motion file'))
            self.file_label = QtWidgets.QLineEdit(self.ros_node.motion_file)
            self.file_label.setReadOnly(True)
            file_row.addWidget(self.file_label, stretch=1)
            main_layout.addLayout(file_row)

            path_row = QtWidgets.QHBoxLayout()
            path_row.addWidget(QtWidgets.QLabel('Path'))
            self.path_combo = QtWidgets.QComboBox()
            path_row.addWidget(self.path_combo, stretch=1)
            self.new_path_button = QtWidgets.QPushButton('New Path')
            self.delete_path_button = QtWidgets.QPushButton('Delete Path')
            path_row.addWidget(self.new_path_button)
            path_row.addWidget(self.delete_path_button)
            main_layout.addLayout(path_row)

            self.step_table = QtWidgets.QTableWidget(0, 3)
            self.step_table.setHorizontalHeaderLabels([
                'Type',
                'Distance / Angle / Duration',
                'Speed',
            ])
            self.step_table.horizontalHeader().setStretchLastSection(True)
            self.step_table.verticalHeader().setVisible(False)
            self.step_table.setSelectionBehavior(
                QtWidgets.QAbstractItemView.SelectRows
            )
            main_layout.addWidget(self.step_table, stretch=3)

            edit_row = QtWidgets.QHBoxLayout()
            self.add_move_button = QtWidgets.QPushButton('Add Move')
            self.add_rotate_button = QtWidgets.QPushButton('Add Rotate')
            self.add_wait_button = QtWidgets.QPushButton('Add Wait')
            self.remove_step_button = QtWidgets.QPushButton('Remove Step')
            self.up_button = QtWidgets.QPushButton('Up')
            self.down_button = QtWidgets.QPushButton('Down')
            for button in (
                self.add_move_button,
                self.add_rotate_button,
                self.add_wait_button,
                self.remove_step_button,
                self.up_button,
                self.down_button,
            ):
                edit_row.addWidget(button)
            main_layout.addLayout(edit_row)

            action_row = QtWidgets.QHBoxLayout()
            self.save_button = QtWidgets.QPushButton('Save YAML')
            self.reload_button = QtWidgets.QPushButton('Reload Executor')
            self.run_button = QtWidgets.QPushButton('Run Selected')
            self.stop_button = QtWidgets.QPushButton('Stop')
            action_row.addWidget(self.save_button)
            action_row.addWidget(self.reload_button)
            action_row.addWidget(self.run_button)
            action_row.addWidget(self.stop_button)
            main_layout.addLayout(action_row)

            self.status_edit = QtWidgets.QTextEdit()
            self.status_edit.setReadOnly(True)
            main_layout.addWidget(self.status_edit, stretch=2)

            self.path_combo.currentTextChanged.connect(self.load_selected_path)
            self.new_path_button.clicked.connect(self.new_path)
            self.delete_path_button.clicked.connect(self.delete_path)
            self.add_move_button.clicked.connect(
                lambda: self.add_step({
                    'type': 'move',
                    'distance': 0.20,
                    'speed': 0.05,
                })
            )
            self.add_rotate_button.clicked.connect(
                lambda: self.add_step(
                    {'type': 'rotate', 'angle': 1.57, 'angular_speed': 0.25}
                )
            )
            self.add_wait_button.clicked.connect(
                lambda: self.add_step({'type': 'wait', 'duration': 1.0})
            )
            self.remove_step_button.clicked.connect(self.remove_selected_step)
            self.up_button.clicked.connect(lambda: self.move_selected_step(-1))
            self.down_button.clicked.connect(
                lambda: self.move_selected_step(1)
            )
            self.save_button.clicked.connect(self.save_yaml)
            self.reload_button.clicked.connect(self.reload_executor)
            self.run_button.clicked.connect(self.run_selected)
            self.stop_button.clicked.connect(self.stop_motion)
            self.step_table.itemChanged.connect(self.table_changed)
            status_bridge.status_received.connect(self.set_status)

            self.load_yaml()

        def ensure_motion_file(self) -> None:
            if os.path.exists(self.ros_node.motion_file):
                return
            directory = os.path.dirname(self.ros_node.motion_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(
                self.ros_node.motion_file,
                'w',
                encoding='utf-8',
            ) as yaml_file:
                yaml.safe_dump(DEFAULT_PATHS, yaml_file, sort_keys=False)

        def load_yaml(self) -> None:
            self.ensure_motion_file()
            try:
                with open(
                    self.ros_node.motion_file,
                    'r',
                    encoding='utf-8',
                ) as yaml_file:
                    data = yaml.safe_load(yaml_file) or {}
            except (OSError, yaml.YAMLError) as exc:
                self.set_status(f'Failed to load YAML: {exc}')
                return

            paths = data.get('paths', {})
            if not isinstance(paths, dict):
                paths = {}

            self.paths = paths
            current = self.path_combo.currentText()
            self.path_combo.blockSignals(True)
            self.path_combo.clear()
            self.path_combo.addItems(sorted(self.paths.keys()))
            if current in self.paths:
                self.path_combo.setCurrentText(current)
            self.path_combo.blockSignals(False)
            self.load_selected_path(self.path_combo.currentText())

        def save_yaml(self) -> None:
            path_name = self.path_combo.currentText()
            if path_name:
                try:
                    self.paths[path_name] = self.steps_from_table()
                except ValueError as exc:
                    self.set_status(f'Cannot save: {exc}')
                    return

            directory = os.path.dirname(self.ros_node.motion_file)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(
                self.ros_node.motion_file,
                'w',
                encoding='utf-8',
            ) as yaml_file:
                yaml.safe_dump(
                    {'paths': self.paths},
                    yaml_file,
                    default_flow_style=False,
                    sort_keys=True,
                )

            self.set_status(f'Saved {self.ros_node.motion_file}')
            self.reload_executor()

        def load_selected_path(self, path_name: str) -> None:
            self.loading_table = True
            self.step_table.setRowCount(0)
            for step in self.paths.get(path_name, []):
                self.add_step(step)
            self.loading_table = False

        def add_step(self, step: dict) -> None:
            row = self.step_table.rowCount()
            self.step_table.insertRow(row)

            step_type = str(step.get('type', 'move'))
            type_combo = QtWidgets.QComboBox()
            type_combo.addItems(self.STEP_TYPES)
            if step_type in self.STEP_TYPES:
                type_combo.setCurrentText(step_type)
            type_combo.currentTextChanged.connect(self.type_changed)
            self.step_table.setCellWidget(row, 0, type_combo)

            value = self.step_value(step)
            speed = self.step_speed(step)
            self.step_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(f'{value:.4f}'),
            )
            self.step_table.setItem(row, 2, QtWidgets.QTableWidgetItem(speed))

        def step_value(self, step: dict) -> float:
            step_type = step.get('type')
            if step_type == 'rotate':
                return float(step.get('angle', 0.0))
            if step_type == 'wait':
                return float(step.get('duration', 0.0))
            return float(step.get('distance', 0.0))

        def step_speed(self, step: dict) -> str:
            step_type = step.get('type')
            if step_type == 'rotate':
                return f'{float(step.get("angular_speed", 0.25)):.4f}'
            if step_type == 'wait':
                return ''
            return f'{float(step.get("speed", 0.05)):.4f}'

        def steps_from_table(self) -> list:
            steps = []
            for row in range(self.step_table.rowCount()):
                type_combo = self.step_table.cellWidget(row, 0)
                step_type = type_combo.currentText()
                value_item = self.step_table.item(row, 1)
                speed_item = self.step_table.item(row, 2)
                value_text = value_item.text().strip() if value_item else ''
                speed_text = speed_item.text().strip() if speed_item else ''

                try:
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(
                        f'row {row + 1} value must be numeric'
                    ) from exc

                if step_type == 'move':
                    speed = self.parse_speed(speed_text, row, default=0.05)
                    steps.append({
                        'type': 'move',
                        'distance': value,
                        'speed': speed,
                    })
                elif step_type == 'rotate':
                    speed = self.parse_speed(speed_text, row, default=0.25)
                    steps.append({
                        'type': 'rotate',
                        'angle': value,
                        'angular_speed': speed,
                    })
                elif step_type == 'wait':
                    steps.append({'type': 'wait', 'duration': value})
            return steps

        def parse_speed(self, text: str, row: int, default: float) -> float:
            if not text:
                return default
            try:
                speed = float(text)
            except ValueError as exc:
                raise ValueError(
                    f'row {row + 1} speed must be numeric'
                ) from exc
            if speed <= 0.0:
                raise ValueError(f'row {row + 1} speed must be positive')
            return speed

        def new_path(self) -> None:
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                'New Path',
                'Path name',
            )
            if not ok:
                return
            name = name.strip()
            if not name:
                return
            if name in self.paths:
                self.set_status(f'Path already exists: {name}')
                return

            self.paths[name] = []
            self.path_combo.addItem(name)
            self.path_combo.setCurrentText(name)

        def delete_path(self) -> None:
            name = self.path_combo.currentText()
            if not name:
                return
            answer = QtWidgets.QMessageBox.question(
                self,
                'Delete Path',
                f'Delete {name}?',
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

            self.paths.pop(name, None)
            self.load_yaml_from_memory()

        def load_yaml_from_memory(self) -> None:
            self.path_combo.blockSignals(True)
            self.path_combo.clear()
            self.path_combo.addItems(sorted(self.paths.keys()))
            self.path_combo.blockSignals(False)
            self.load_selected_path(self.path_combo.currentText())

        def remove_selected_step(self) -> None:
            rows = sorted(
                {item.row() for item in self.step_table.selectedItems()},
                reverse=True,
            )
            for row in rows:
                self.step_table.removeRow(row)

        def move_selected_step(self, direction: int) -> None:
            rows = sorted(
                {item.row() for item in self.step_table.selectedItems()}
            )
            if len(rows) != 1:
                return
            row = rows[0]
            new_row = row + direction
            if new_row < 0 or new_row >= self.step_table.rowCount():
                return

            steps = self.steps_from_table()
            steps[row], steps[new_row] = steps[new_row], steps[row]
            self.loading_table = True
            self.step_table.setRowCount(0)
            for step in steps:
                self.add_step(step)
            self.loading_table = False
            self.step_table.selectRow(new_row)

        def table_changed(self) -> None:
            if self.loading_table:
                return

        def type_changed(self, *_args) -> None:
            if self.loading_table:
                return
            row = self.sender_row()
            if row is None:
                return
            type_combo = self.step_table.cellWidget(row, 0)
            step_type = type_combo.currentText()
            if step_type == 'move':
                self.step_table.item(row, 1).setText('0.2000')
                self.step_table.item(row, 2).setText('0.0500')
            elif step_type == 'rotate':
                self.step_table.item(row, 1).setText('1.5700')
                self.step_table.item(row, 2).setText('0.2500')
            elif step_type == 'wait':
                self.step_table.item(row, 1).setText('1.0000')
                self.step_table.item(row, 2).setText('')

        def sender_row(self):
            sender = self.sender()
            for row in range(self.step_table.rowCount()):
                if self.step_table.cellWidget(row, 0) is sender:
                    return row
            return None

        def reload_executor(self) -> None:
            self.ros_node.call_trigger(
                self.ros_node.reload_client,
                '/tb3_motion/reload',
            )

        def run_selected(self) -> None:
            self.save_yaml()
            path_name = self.path_combo.currentText()
            if not path_name:
                self.set_status('No path selected.')
                return
            try:
                steps = self.steps_from_table()
            except ValueError as exc:
                self.set_status(f'Cannot run: {exc}')
                return
            self.ros_node.publish_command({
                'command': 'run',
                'path': path_name,
                'steps': steps,
            })

        def stop_motion(self) -> None:
            self.ros_node.call_trigger(
                self.ros_node.stop_client,
                '/tb3_motion/stop',
            )

        def set_status(self, text: str) -> None:
            self.status_edit.setPlainText(text)


def main(args=None):
    if QtWidgets is None:
        print(
            'PyQt5 is required for motion_ui_node but could not be imported: '
            f'{QT_IMPORT_ERROR}'
        )
        return

    rclpy.init(args=args)
    app = QtWidgets.QApplication(sys.argv)
    status_bridge = StatusBridge()
    node = MotionUiNode(status_bridge)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    window = MotionWindow(node, status_bridge)
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
