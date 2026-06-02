import math
import os
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout


ARUCO_DICTIONARIES = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
}


def create_detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters'):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def detect_markers(gray, dictionary, parameters):
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length):
    try:
        if hasattr(cv2, 'drawFrameAxes'):
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
        elif hasattr(cv2.aruco, 'drawAxis'):
            cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    except cv2.error:
        pass


def estimate_marker_poses(corners, marker_size, camera_matrix, dist_coeffs):
    if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_size,
            camera_matrix,
            dist_coeffs,
        )
        return rvecs, tvecs

    half_size = marker_size * 0.5
    object_points = np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )
    solvepnp_flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    rvecs = []
    tvecs = []
    for marker_corners in corners:
        image_points = marker_corners.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=solvepnp_flag,
        )
        if not ok:
            rvec = np.full((3, 1), np.nan, dtype=np.float64)
            tvec = np.full((3, 1), np.nan, dtype=np.float64)
        rvecs.append(rvec.reshape(1, 3))
        tvecs.append(tvec.reshape(1, 3))
    return np.array(rvecs, dtype=np.float64), np.array(tvecs, dtype=np.float64)


def marker_yaw_from_rvec(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return yaw


class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose')

        self.declare_parameter('image_topic', '/camera/image_decompressed')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('annotated_topic', '/aruco/annotated_image')
        self.declare_parameter('align_error_topic', '/aruco/align_error')
        self.declare_parameter('active_topic', '/aruco_align/active')
        self.declare_parameter('shutdown_topic', '/aruco_align/shutdown')
        self.declare_parameter('dictionary', 'DICT_4X4_50')
        self.declare_parameter('marker_size', 0.015)
        self.declare_parameter('marker_ids', '0,1')
        self.declare_parameter('min_valid_pose_distance', 0.01)
        self.declare_parameter('log_period_sec', 0.5)
        self.declare_parameter('use_fallback_camera_info', True)
        self.declare_parameter('fallback_fx', 554.38271282)
        self.declare_parameter('fallback_fy', 554.38271282)
        self.declare_parameter('fallback_cx', 320.5)
        self.declare_parameter('fallback_cy', 240.5)
        self.declare_parameter('log_only_when_active', True)
        self.declare_parameter('process_only_when_active', True)
        self.declare_parameter('pose_hold_sec', 0.8)
        # Detector tuning is disabled for the delivery path; the original node
        # uses OpenCV's default ArUco parameters directly.
        # self.declare_parameter('detector_upscale', 2.0)
        # self.declare_parameter('equalize_histogram', True)
        # self.declare_parameter('adaptive_thresh_win_size_min', 3)
        # self.declare_parameter('adaptive_thresh_win_size_max', 53)
        # self.declare_parameter('adaptive_thresh_win_size_step', 4)
        # self.declare_parameter('adaptive_thresh_constant', 7.0)
        # self.declare_parameter('min_marker_perimeter_rate', 0.008)
        # self.declare_parameter('max_marker_perimeter_rate', 4.0)
        # self.declare_parameter('polygonal_approx_accuracy_rate', 0.05)
        # self.declare_parameter('min_corner_distance_rate', 0.02)
        # self.declare_parameter('min_distance_to_border', 2)
        # self.declare_parameter('corner_refinement', True)
        # self.declare_parameter('corner_refinement_win_size', 5)
        # self.declare_parameter('corner_refinement_max_iterations', 30)
        # self.declare_parameter('corner_refinement_min_accuracy', 0.01)

        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        annotated_topic = self.get_parameter('annotated_topic').value
        align_error_topic = self.get_parameter('align_error_topic').value
        active_topic = self.get_parameter('active_topic').value
        shutdown_topic = self.get_parameter('shutdown_topic').value
        dictionary_name = self.get_parameter('dictionary').value

        self.marker_size = float(self.get_parameter('marker_size').value)
        self.target_ids = self.parse_marker_ids(self.get_parameter('marker_ids').value)
        self.min_valid_pose_distance = float(self.get_parameter('min_valid_pose_distance').value)
        self.log_period_sec = float(self.get_parameter('log_period_sec').value)
        self.use_fallback_camera_info = bool(
            self.get_parameter('use_fallback_camera_info').value
        )
        self.log_only_when_active = bool(self.get_parameter('log_only_when_active').value)
        self.process_only_when_active = bool(
            self.get_parameter('process_only_when_active').value
        )
        self.pose_hold_sec = float(self.get_parameter('pose_hold_sec').value)
        # self.detector_upscale = float(self.get_parameter('detector_upscale').value)
        # self.equalize_histogram = bool(self.get_parameter('equalize_histogram').value)
        self.fallback_camera_matrix = np.array(
            [
                [float(self.get_parameter('fallback_fx').value), 0.0,
                 float(self.get_parameter('fallback_cx').value)],
                [0.0, float(self.get_parameter('fallback_fy').value),
                 float(self.get_parameter('fallback_cy').value)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.fallback_dist_coeffs = np.zeros(5, dtype=np.float64)
        self.last_log_time = self.get_clock().now()

        if self.marker_size <= 0.0:
            raise ValueError('marker_size must be greater than 0.0 meters.')
        if len(self.target_ids) != 2:
            raise ValueError('marker_ids must contain exactly two marker ids.')

        if dictionary_name not in ARUCO_DICTIONARIES:
            valid = ', '.join(sorted(ARUCO_DICTIONARIES))
            raise ValueError(f'Unsupported dictionary "{dictionary_name}". Valid: {valid}')

        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
        self.detector_parameters = create_detector_parameters()
        # self.configure_detector_parameters()
        self.bridge = CvBridge()
        self.camera_matrix = self.fallback_camera_matrix.copy() if self.use_fallback_camera_info else None
        self.dist_coeffs = self.fallback_dist_coeffs.copy() if self.use_fallback_camera_info else None
        self.using_fallback_camera_info = self.use_fallback_camera_info
        self.alignment_active = False
        self.last_target_poses = {}
        self.last_target_pose_times = {}

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.active_sub = self.create_subscription(
            Bool,
            active_topic,
            self.active_callback,
            10,
        )
        self.shutdown_sub = self.create_subscription(
            Bool,
            shutdown_topic,
            self.shutdown_callback,
            10,
        )
        self.annotated_pub = self.create_publisher(Image, annotated_topic, 10)
        self.align_error_pub = self.create_publisher(Float32MultiArray, align_error_topic, 10)

        self.get_logger().info(f'Subscribed image: {image_topic}')
        self.get_logger().info(f'Subscribed camera info: {camera_info_topic}')
        self.get_logger().info(f'Subscribed active command: {active_topic}')
        self.get_logger().info(f'Subscribed shutdown command: {shutdown_topic}')
        self.get_logger().info(f'Publishing annotated image: {annotated_topic}')
        self.get_logger().info(f'Publishing align error: {align_error_topic}')
        self.get_logger().info(
            f'Aruco dictionary={dictionary_name}, ids={sorted(self.target_ids)}, '
            f'marker_size={self.marker_size:.4f} m'
        )
        self.get_logger().info(f'min_valid_pose_distance={self.min_valid_pose_distance:.2f} m')
        self.get_logger().info(f'log_only_when_active={self.log_only_when_active}')
        self.get_logger().info(f'process_only_when_active={self.process_only_when_active}')
        self.get_logger().info(f'pose_hold_sec={self.pose_hold_sec:.2f}')
        if self.use_fallback_camera_info:
            self.get_logger().warn(
                'Using fallback camera intrinsics until valid /camera/camera_info arrives: '
                f'K={self.fallback_camera_matrix.reshape(-1).tolist()}'
            )

    @staticmethod
    def parse_marker_ids(value):
        if isinstance(value, str):
            return {int(item.strip()) for item in value.split(',') if item.strip()}
        return {int(item) for item in value}

    def active_callback(self, msg):
        was_active = self.alignment_active
        self.alignment_active = bool(msg.data)
        if self.alignment_active != was_active:
            state = 'active' if self.alignment_active else 'inactive'
            self.get_logger().info(f'ArUco pose logging {state}.')
            if not self.alignment_active:
                self.clear_pose_cache()
                self.publish_align_error({})

    def shutdown_callback(self, msg):
        if not bool(msg.data):
            return
        self.get_logger().info('ArUco pose shutdown requested.')
        threading.Timer(0.1, lambda: os._exit(0)).start()
        rclpy.shutdown()

    def camera_info_callback(self, msg):
        camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if not self.is_valid_camera_matrix(camera_matrix):
            if self.camera_matrix is None:
                self.log_when_allowed(
                    self.get_logger().warn,
                    f'Invalid camera_info K matrix: {camera_matrix.reshape(-1).tolist()}',
                    throttle_duration_sec=2.0,
                )
            else:
                source = 'fallback' if self.using_fallback_camera_info else 'last valid'
                self.log_when_allowed(
                    self.get_logger().warn,
                    f'Invalid camera_info K matrix; keeping {source} intrinsics. '
                    f'K={camera_matrix.reshape(-1).tolist()}',
                    throttle_duration_sec=2.0,
                )
            return

        dist_coeffs = np.array(msg.d, dtype=np.float64)
        if dist_coeffs.size == 0:
            dist_coeffs = np.zeros(5, dtype=np.float64)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        if self.using_fallback_camera_info:
            self.log_when_allowed(
                self.get_logger().info,
                f'Received valid camera_info. Switching from fallback to K={camera_matrix.reshape(-1).tolist()}'
            )
        self.using_fallback_camera_info = False

    @staticmethod
    def is_valid_camera_matrix(camera_matrix):
        return bool(
            camera_matrix.shape == (3, 3)
            and np.all(np.isfinite(camera_matrix))
            and camera_matrix[0, 0] > 0.0
            and camera_matrix[1, 1] > 0.0
            and camera_matrix[2, 2] != 0.0
        )

    def image_callback(self, msg):
        if self.process_only_when_active and not self.alignment_active:
            return

        try:
            self.process_image(msg)
        except Exception as exc:  # noqa: BLE001 - keep perception alive during debugging.
            self.get_logger().error(
                f'ArUco image processing failed: {exc}',
                throttle_duration_sec=1.0,
            )

    def process_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001 - keep the node alive on bad camera frames.
            self.get_logger().error(f'cv_bridge image conversion failed: {exc}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detect_markers(gray, self.dictionary, self.detector_parameters)

        if ids is None or len(ids) == 0:
            self.publish_annotated(frame, msg.header)
            poses = self.poses_with_hold({})
            self.publish_align_error(poses)
            if poses:
                self.log_pose_summary(poses, prefix='No fresh ArUco markers; using held pose')
            else:
                self.log_periodic('No ArUco markers detected.')
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if self.camera_matrix is None or self.dist_coeffs is None:
            self.publish_annotated(frame, msg.header)
            self.publish_align_error({})
            self.log_when_allowed(
                self.get_logger().warn,
                'ArUco markers detected, but valid camera_info is missing. '
                'Pose alignment is disabled until /camera/camera_info has a non-zero K matrix '
                'or use_fallback_camera_info is true.',
                throttle_duration_sec=2.0,
            )
            return

        rvecs, tvecs = estimate_marker_poses(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.dist_coeffs,
        )

        poses = {}
        for index, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            rvec = rvecs[index].reshape(3)
            tvec = tvecs[index].reshape(3)
            draw_axes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 0.5)

            if marker_id in self.target_ids:
                if not self.is_valid_tvec(tvec):
                    self.get_logger().warn(
                        f'Ignoring invalid pose for marker {marker_id}: tvec={tvec.tolist()}',
                        throttle_duration_sec=1.0,
                    )
                    continue
                poses[marker_id] = {
                    'tvec': tvec,
                    'yaw': marker_yaw_from_rvec(rvec),
                    'held': False,
                }
                self.draw_marker_text(frame, corners[index], marker_id, tvec)

        self.update_pose_cache(poses)
        poses = self.poses_with_hold(poses)
        self.publish_annotated(frame, msg.header)
        self.publish_align_error(poses)
        self.log_pose_summary(poses)

    def clear_pose_cache(self):
        self.last_target_poses.clear()
        self.last_target_pose_times.clear()

    def update_pose_cache(self, poses):
        now = time.monotonic()
        for marker_id, pose in poses.items():
            if marker_id not in self.target_ids:
                continue
            self.last_target_poses[marker_id] = {
                'tvec': pose['tvec'].copy(),
                'yaw': float(pose['yaw']),
                'held': True,
            }
            self.last_target_pose_times[marker_id] = now

    def poses_with_hold(self, poses):
        if self.pose_hold_sec <= 0.0:
            return poses

        now = time.monotonic()
        merged = dict(poses)
        expired_ids = []
        for marker_id, pose in self.last_target_poses.items():
            age = now - self.last_target_pose_times.get(marker_id, 0.0)
            if age > self.pose_hold_sec:
                expired_ids.append(marker_id)
                continue
            if marker_id not in merged:
                merged[marker_id] = {
                    'tvec': pose['tvec'].copy(),
                    'yaw': float(pose['yaw']),
                    'held': True,
                }

        for marker_id in expired_ids:
            self.last_target_poses.pop(marker_id, None)
            self.last_target_pose_times.pop(marker_id, None)
        return merged

    def draw_marker_text(self, frame, corner, marker_id, tvec):
        points = corner.reshape(4, 2)
        x = int(points[:, 0].mean())
        y = int(points[:, 1].mean())
        text = f'id {marker_id}: x={tvec[0]:.2f} z={tvec[2]:.2f}'
        cv2.putText(
            frame,
            text,
            (max(0, x - 90), max(20, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def is_valid_tvec(self, tvec):
        return bool(np.all(np.isfinite(tvec)) and tvec[2] >= self.min_valid_pose_distance)

    def publish_annotated(self, frame, header):
        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        annotated_msg.header = header
        self.annotated_pub.publish(annotated_msg)

    def publish_align_error(self, poses):
        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(
                    label=(
                        '[center_x_m, center_z_m, z_diff_m, both_markers_visible, '
                        'visible_count, single_marker_id, single_marker_x_m, single_marker_z_m]'
                    ),
                    size=8,
                    stride=8,
                )
            ],
            data_offset=0,
        )

        if not self.target_ids.issubset(poses):
            visible_ids = sorted(marker_id for marker_id in poses if marker_id in self.target_ids)
            if len(visible_ids) == 1:
                marker_id = visible_ids[0]
                tvec = poses[marker_id]['tvec']
                msg.data = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    float(marker_id),
                    float(tvec[0]),
                    float(tvec[2]),
                ]
            else:
                msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
            self.align_error_pub.publish(msg)
            return

        sorted_ids = sorted(self.target_ids)
        first = poses[sorted_ids[0]]['tvec']
        second = poses[sorted_ids[1]]['tvec']
        center = (first + second) * 0.5
        z_diff = second[2] - first[2]

        msg.data = [
            float(center[0]),
            float(center[2]),
            float(z_diff),
            1.0,
            2.0,
            -1.0,
            0.0,
            0.0,
        ]
        self.align_error_pub.publish(msg)

    def log_periodic(self, message):
        if not self.should_log_detail():
            return
        now = self.get_clock().now()
        elapsed = (now - self.last_log_time).nanoseconds / 1e9
        if elapsed < self.log_period_sec:
            return
        self.last_log_time = now
        self.get_logger().info(message)

    def log_pose_summary(self, poses, prefix=''):
        if not poses:
            self.log_periodic('Target markers not visible.')
            return

        lines = []
        for marker_id in sorted(poses):
            tvec = poses[marker_id]['tvec']
            yaw_deg = math.degrees(poses[marker_id]['yaw'])
            held_text = ' held' if poses[marker_id].get('held', False) else ''
            lines.append(
                f'id {marker_id}{held_text}: x={tvec[0]:+.3f}m y={tvec[1]:+.3f}m '
                f'z={tvec[2]:.3f}m yaw={yaw_deg:+.1f}deg'
            )

        if self.target_ids.issubset(poses):
            sorted_ids = sorted(self.target_ids)
            first = poses[sorted_ids[0]]['tvec']
            second = poses[sorted_ids[1]]['tvec']
            center = (first + second) * 0.5
            baseline = second - first
            baseline_yaw = math.degrees(math.atan2(baseline[0], baseline[2]))
            lines.append(
                f'center: x={center[0]:+.3f}m y={center[1]:+.3f}m z={center[2]:.3f}m '
                f'baseline_yaw={baseline_yaw:+.1f}deg'
            )

        message = ' | '.join(lines)
        if prefix:
            message = f'{prefix}: {message}'
        self.log_periodic(message)

    def should_log_detail(self):
        return self.alignment_active or not self.log_only_when_active

    def log_when_allowed(self, log_fn, message, **kwargs):
        if self.should_log_detail():
            log_fn(message, **kwargs)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
