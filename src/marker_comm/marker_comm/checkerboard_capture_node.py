import datetime
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CheckerboardCaptureNode(Node):
    def __init__(self):
        super().__init__('checkerboard_capture')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('save_dir', 'checkerboards')
        self.declare_parameter('show_preview', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.save_dir = Path(self.get_parameter('save_dir').value).expanduser()
        self.show_preview = bool(self.get_parameter('show_preview').value)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.saved_count = 0

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        if self.show_preview:
            self.timer = self.create_timer(0.03, self.preview_loop)

        self.get_logger().info(f'Subscribed to {self.image_topic}')
        self.get_logger().info(f'Saving checkerboard images to {self.save_dir}')
        if self.show_preview:
            self.get_logger().info('Preview keys: s/a = save, q = quit')

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def preview_loop(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        cv2.putText(
            frame,
            f's/a: save  q: quit  saved: {self.saved_count}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow('checkerboard_capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('a')):
            self.save_frame()
        elif key == ord('q'):
            rclpy.shutdown()

    def save_frame(self):
        if self.latest_frame is None:
            self.get_logger().warn('No image received yet.')
            return

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = self.save_dir / f'capture_{timestamp}.png'
        cv2.imwrite(str(filename), self.latest_frame)
        self.saved_count += 1
        self.get_logger().info(f'Saved {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = CheckerboardCaptureNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
