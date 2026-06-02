import json
import os
import time
import threading

import cv2
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import torch
from ultralytics import YOLO


class YoloV8RosNode(Node):
    def __init__(self):
        super().__init__('yolo_v8_ros_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', '/root/robotpro.pt')

        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_size', 640)
        dynamic_parameter = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('device', 'cpu', descriptor=dynamic_parameter)

        # timer_period=0.2 means YOLO runs about 5 fps.
        self.declare_parameter('timer_period', 0.2)

        self.declare_parameter('display_window', False)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('annotated_image_topic', '/yolo/image')
        self.declare_parameter('classes_topic', '/yolo/classes')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('best_object_topic', '/yolo/best_object')
        self.declare_parameter('available_classes_topic', '/yolo/available_classes')
        self.declare_parameter('publish_best_object', True)
        self.declare_parameter('continuous_detection', True)
        self.declare_parameter('keep_image_subscription', True)
        self.declare_parameter('snapshot_wait_for_image_sec', 2.0)
        self.declare_parameter('snapshot_duration_sec', 3.0)
        self.declare_parameter('snapshot_sample_period_sec', 0.2)
        self.declare_parameter('log_detections_period_sec', 0.0)
        self.declare_parameter('snapshot_service_name', '/yolo/snapshot')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('drive_on_snapshot_detection', True)
        self.declare_parameter('snapshot_forward_linear_x', 0.2)
        self.declare_parameter('snapshot_forward_duration_sec', 2.0)
        self.declare_parameter('snapshot_reverse_linear_x', -0.05)
        self.declare_parameter('snapshot_reverse_duration_sec', 3.0)
        self.declare_parameter(
            'exclude_class_names',
            '',
            descriptor=dynamic_parameter,
        )

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model_path').value

        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.input_size = int(self.get_parameter('input_size').value)
        self.device = self.normalize_device(self.get_parameter('device').value)
        self.timer_period = float(self.get_parameter('timer_period').value)

        self.display_window = bool(self.get_parameter('display_window').value)
        self.publish_annotated_image = bool(
            self.get_parameter('publish_annotated_image').value
        )
        self.annotated_image_topic = self.get_parameter('annotated_image_topic').value
        self.classes_topic = self.get_parameter('classes_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.best_object_topic = self.get_parameter('best_object_topic').value
        self.available_classes_topic = self.get_parameter('available_classes_topic').value
        self.publish_best_object = bool(self.get_parameter('publish_best_object').value)
        self.continuous_detection = bool(
            self.get_parameter('continuous_detection').value
        )
        self.keep_image_subscription = (
            bool(self.get_parameter('keep_image_subscription').value)
            or self.continuous_detection
        )
        self.snapshot_wait_for_image_sec = float(
            self.get_parameter('snapshot_wait_for_image_sec').value
        )
        self.snapshot_duration_sec = float(
            self.get_parameter('snapshot_duration_sec').value
        )
        self.snapshot_sample_period_sec = float(
            self.get_parameter('snapshot_sample_period_sec').value
        )
        self.log_detections_period_sec = float(
            self.get_parameter('log_detections_period_sec').value
        )
        self.snapshot_service_name = self.get_parameter('snapshot_service_name').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.drive_on_snapshot_detection = bool(
            self.get_parameter('drive_on_snapshot_detection').value
        )
        self.snapshot_forward_linear_x = float(
            self.get_parameter('snapshot_forward_linear_x').value
        )
        self.snapshot_forward_duration_sec = float(
            self.get_parameter('snapshot_forward_duration_sec').value
        )
        self.snapshot_reverse_linear_x = float(
            self.get_parameter('snapshot_reverse_linear_x').value
        )
        self.snapshot_reverse_duration_sec = float(
            self.get_parameter('snapshot_reverse_duration_sec').value
        )
        self.exclude_class_names = set(
            self.string_list_parameter('exclude_class_names')
        )

        self.bridge = CvBridge()
        self.latest_msg = None
        self.latest_stamp_ns = None
        self.last_processed_stamp_ns = None
        self.lock = threading.Lock()
        self.image_callback_group = ReentrantCallbackGroup()
        self.service_callback_group = ReentrantCallbackGroup()

        self.last_log_time = time.time()

        if os.path.isabs(self.model_path) and not os.path.exists(self.model_path):
            message = (
                f'YOLO model file not found: {self.model_path}. '
                'Pass yolo_model_path:=/path/to/best.pt or model_path:=/path/to/best.pt.'
            )
            self.get_logger().fatal(message)
            raise RuntimeError(message)

        self.get_logger().info(f'Loading YOLOv8 model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.predict_class_ids = self.class_ids_for_prediction()

        if self.keep_image_subscription:
            self.image_sub = self.create_image_subscription(self.image_callback)
        else:
            self.image_sub = None

        self.detection_pub = self.create_publisher(String, self.detections_topic, 10)
        self.classes_pub = self.create_publisher(String, self.classes_topic, 10)
        self.best_object_pub = self.create_publisher(String, self.best_object_topic, 10)
        self.available_classes_pub = self.create_publisher(
            String,
            self.available_classes_topic,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.snapshot_service = self.create_service(
            Trigger,
            self.snapshot_service_name,
            self.snapshot_callback,
            callback_group=self.service_callback_group,
        )

        if self.publish_annotated_image:
            self.image_pub = self.create_publisher(
                Image,
                self.annotated_image_topic,
                qos_profile_sensor_data
            )
        else:
            self.image_pub = None

        if self.display_window:
            cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLOv8 Detection', 640, 480)

        if self.continuous_detection:
            self.timer = self.create_timer(self.timer_period, self.timer_callback)
        else:
            self.timer = None
        self.available_classes_timer = self.create_timer(1.0, self.publish_available_classes)

        if self.keep_image_subscription:
            self.get_logger().info(f'Subscribing image topic: {self.image_topic}')
        else:
            self.get_logger().info(
                f'Image topic {self.image_topic} will be subscribed only during snapshots.'
            )
        self.get_logger().info(f'Publishing detection result: {self.detections_topic}')
        self.get_logger().info(f'Publishing detected classes: {self.classes_topic}')
        self.get_logger().info(f'Publishing best object: {self.best_object_topic}')
        self.get_logger().info(f'Publishing available classes: {self.available_classes_topic}')
        if self.publish_annotated_image:
            self.get_logger().info(
                f'Publishing annotated image for rqt: {self.annotated_image_topic}'
            )
        else:
            self.get_logger().info('Annotated image publishing is disabled')
        self.get_logger().info(f'Timer period: {self.timer_period} sec')
        self.get_logger().info(f'Continuous detection: {self.continuous_detection}')
        self.get_logger().info(f'Keep image subscription: {self.keep_image_subscription}')
        self.get_logger().info(
            f'Snapshot judgment window: {self.snapshot_duration_sec:.2f} sec'
        )
        self.get_logger().info(f'YOLO snapshot service: {self.snapshot_service_name}')
        self.get_logger().info(f'Snapshot drive topic: {self.cmd_vel_topic}')
        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(
            f'Excluding class names: {sorted(self.exclude_class_names)}'
        )
        self.publish_available_classes()

    def create_image_subscription(self, callback):
        return self.create_subscription(
            Image,
            self.image_topic,
            callback,
            qos_profile_sensor_data,
            callback_group=self.image_callback_group,
        )

    def image_callback(self, msg):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        with self.lock:
            self.latest_msg = msg
            self.latest_stamp_ns = stamp_ns

    def string_list_parameter(self, name):
        value = self.get_parameter(name).value
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def normalize_device(self, value):
        if value is None:
            return 'cpu'

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        device = str(value).strip()
        if not device:
            return 'cpu'

        requested_cuda = device.isdigit() or device.lower().startswith('cuda')
        if requested_cuda and not torch.cuda.is_available():
            self.get_logger().warn(
                f'YOLO device {device!r} requested, but CUDA is not available. Falling back to cpu.'
            )
            return 'cpu'

        return device

    def timer_callback(self):
        with self.lock:
            msg = self.latest_msg
            stamp_ns = self.latest_stamp_ns

        if msg is None:
            return

        if stamp_ns == self.last_processed_stamp_ns:
            return

        self.last_processed_stamp_ns = stamp_ns
        self.process_image_message(msg)

    def snapshot_callback(self, request, response):
        del request
        try:
            result = self.process_snapshot_window()
            if result is None:
                response.success = True
                response.message = 'YOLO snapshot: no image frame available yet'
                return response

            detections = result['detections']
            best_detection = result['best_detection']
            drove_sequence = bool(detections) and self.drive_on_snapshot_detection
            if drove_sequence:
                self.drive_snapshot_sequence()

            response.success = True
            response.message = json.dumps({
                'message': 'YOLO snapshot complete',
                'snapshot_duration_sec': result['duration_sec'],
                'frames_processed': result['frames_processed'],
                'frames_with_detections': result['frames_with_detections'],
                'detection_count': len(detections),
                'detections': detections,
                'detected_classes': result['detected_classes'],
                'best_object': best_detection,
                'raw_detection_count': len(result['raw_detections']),
                'raw_detections_before_exclude': result['raw_detections'],
                'excluded_detections': result['excluded_detections'],
                'conf_threshold': self.conf_threshold,
                'exclude_class_names': sorted(self.exclude_class_names),
                'drove_sequence': drove_sequence,
                'drove_forward': drove_sequence,
                'forward_linear_x': (
                    self.snapshot_forward_linear_x if drove_sequence else 0.0
                ),
                'forward_duration_sec': (
                    self.snapshot_forward_duration_sec if drove_sequence else 0.0
                ),
                'reverse_linear_x': (
                    self.snapshot_reverse_linear_x if drove_sequence else 0.0
                ),
                'reverse_duration_sec': (
                    self.snapshot_reverse_duration_sec if drove_sequence else 0.0
                ),
            }, ensure_ascii=False)
            return response
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'YOLO snapshot error: {exc}')
            response.success = False
            response.message = str(exc)
            return response

    def process_image_message(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return None

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.input_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                classes=self.predict_class_ids,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f'YOLO predict error: {e}')
            return None

        result = results[0]
        annotated_frame = frame.copy()
        raw_detections = []
        excluded_detections = []
        detections = []
        detected_classes = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = map(int, xyxy)
                class_name = self.class_name_for_id(cls_id)
                detection = {
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'bbox_xywh': [x1, y1, x2 - x1, y2 - y1]
                }
                raw_detections.append(detection)
                if class_name in self.exclude_class_names:
                    excluded_detections.append(detection)
                    continue

                detected_classes.append(class_name)
                detections.append(detection)

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f'{class_name} {conf:.2f}'
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        detection_msg = String()
        detection_msg.data = json.dumps(detections, ensure_ascii=False)
        self.detection_pub.publish(detection_msg)

        classes_msg = String()
        classes_msg.data = json.dumps(detected_classes, ensure_ascii=False)
        self.classes_pub.publish(classes_msg)

        best_detection = max(detections, key=lambda item: item['confidence']) if detections else None
        if self.publish_best_object and best_detection is not None:
            best_msg = String()
            best_msg.data = json.dumps({
                'class': best_detection['class_name'],
                'object_class': best_detection['class_name'],
                'label': best_detection['class_name'],
                'confidence': best_detection['confidence'],
                'class_id': best_detection['class_id'],
                'bbox_xyxy': best_detection['bbox_xyxy'],
                'bbox_xywh': best_detection['bbox_xywh'],
            }, ensure_ascii=False)
            self.best_object_pub.publish(best_msg)

        if self.publish_annotated_image and self.image_pub is not None:
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)

        if self.display_window:
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.get_logger().info('q pressed. Closing OpenCV window.')
                cv2.destroyAllWindows()

        self.log_detections_if_needed(len(detections))
        return {
            'detections': detections,
            'detected_classes': detected_classes,
            'best_detection': best_detection,
            'raw_detections': raw_detections,
            'excluded_detections': excluded_detections,
        }

    def process_snapshot_window(self):
        if self.keep_image_subscription:
            return self.process_snapshot_messages(self.get_latest_image_message)

        state_lock = threading.Lock()
        event = threading.Event()
        state = {'msg': None, 'stamp_ns': None}

        def one_image_callback(image_msg):
            stamp = (
                image_msg.header.stamp.sec * 1_000_000_000
                + image_msg.header.stamp.nanosec
            )
            with state_lock:
                state['msg'] = image_msg
                state['stamp_ns'] = stamp
            event.set()

        def get_latest():
            with state_lock:
                return state['msg'], state['stamp_ns']

        subscription = self.create_image_subscription(one_image_callback)
        try:
            if not event.wait(timeout=max(0.1, self.snapshot_wait_for_image_sec)):
                return None
            return self.process_snapshot_messages(get_latest)
        finally:
            self.destroy_subscription(subscription)

    def get_latest_image_message(self):
        with self.lock:
            return self.latest_msg, self.latest_stamp_ns

    def process_snapshot_messages(self, get_latest_message):
        duration_sec = max(0.1, self.snapshot_duration_sec)
        sample_period_sec = max(0.05, self.snapshot_sample_period_sec)
        deadline = time.monotonic() + duration_sec
        next_sample_time = 0.0
        last_stamp_ns = None
        frames_processed = 0
        frames_with_detections = 0
        detections = []
        raw_detections = []
        excluded_detections = []
        detected_classes = []

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_sample_time:
                time.sleep(min(0.02, next_sample_time - now))
                continue

            msg, stamp_ns = get_latest_message()
            if msg is None or stamp_ns == last_stamp_ns:
                time.sleep(min(0.02, max(0.0, deadline - now)))
                continue

            result = self.process_image_message(msg)
            last_stamp_ns = stamp_ns
            self.last_processed_stamp_ns = stamp_ns
            next_sample_time = time.monotonic() + sample_period_sec
            if result is None:
                continue

            frames_processed += 1
            if result['detections']:
                frames_with_detections += 1

            for detection in result['detections']:
                detections.append(self.copy_detection_with_sample(detection, frames_processed))
            for detection in result['raw_detections']:
                raw_detections.append(self.copy_detection_with_sample(detection, frames_processed))
            for detection in result['excluded_detections']:
                excluded_detections.append(
                    self.copy_detection_with_sample(detection, frames_processed)
                )
            for class_name in result['detected_classes']:
                if class_name not in detected_classes:
                    detected_classes.append(class_name)

        if frames_processed == 0:
            return None

        best_detection = (
            max(detections, key=lambda item: item['confidence'])
            if detections
            else None
        )
        return {
            'duration_sec': duration_sec,
            'frames_processed': frames_processed,
            'frames_with_detections': frames_with_detections,
            'detections': detections,
            'detected_classes': detected_classes,
            'best_detection': best_detection,
            'raw_detections': raw_detections,
            'excluded_detections': excluded_detections,
        }

    @staticmethod
    def copy_detection_with_sample(detection, sample_index):
        copied = dict(detection)
        copied['sample_index'] = sample_index
        return copied

    def log_detections_if_needed(self, detection_count):
        if self.log_detections_period_sec <= 0.0:
            return

        now = time.time()
        if now - self.last_log_time >= self.log_detections_period_sec:
            self.get_logger().info(f'Detections: {detection_count}')
            self.last_log_time = now

    def drive_snapshot_sequence(self):
        self.drive_cmd_vel_for(
            self.snapshot_forward_linear_x,
            self.snapshot_forward_duration_sec,
        )
        self.drive_cmd_vel_for(
            self.snapshot_reverse_linear_x,
            self.snapshot_reverse_duration_sec,
        )

    def drive_cmd_vel_for(self, linear_x, duration_sec):
        if duration_sec <= 0.0 or abs(linear_x) < 1e-9:
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        deadline = time.monotonic() + float(duration_sec)

        try:
            while time.monotonic() < deadline:
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.05)
        finally:
            self.cmd_vel_pub.publish(Twist())

    def publish_available_classes(self):
        class_names = []
        names = self.model.names
        class_ids = range(len(names)) if isinstance(names, list) else sorted(names)
        for class_id in class_ids:
            class_name = self.class_name_for_id(class_id)
            if class_name in self.exclude_class_names:
                continue
            class_names.append(class_name)

        msg = String()
        msg.data = json.dumps(class_names, ensure_ascii=False)
        self.available_classes_pub.publish(msg)

    def class_ids_for_prediction(self):
        if not self.exclude_class_names:
            return None

        names = self.model.names
        class_ids = range(len(names)) if isinstance(names, list) else sorted(names)
        return [
            int(class_id)
            for class_id in class_ids
            if self.class_name_for_id(int(class_id)) not in self.exclude_class_names
        ]

    def class_name_for_id(self, class_id):
        names = self.model.names
        if isinstance(names, list):
            if 0 <= class_id < len(names):
                return names[class_id]
            return str(class_id)
        return names.get(class_id, str(class_id))


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = YoloV8RosNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        return 1
    finally:
        if node is not None and node.display_window:
            cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
