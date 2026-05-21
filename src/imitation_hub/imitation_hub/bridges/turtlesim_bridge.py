import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from turtlesim.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np


class TurtlesimBridge(Node):
    def __init__(self):
        super().__init__("turtlesim_bridge")
        self.bridge = CvBridge()
        self._canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Listen to the Turtlesim game
        self.create_subscription(Pose, "/turtle1/pose", self.pose_cb, 10)
        self.create_subscription(Twist, "/turtle1/cmd_vel", self.cmd_cb, 10)

        # Publish standard topics to feed your Data Collector
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.teleop_pub = self.create_publisher(Twist, "/cmd_vel_teleop", 10)

        self.latest_pose = Pose()
        self.latest_cmd = Twist()

        # Run at 30Hz (Standard camera framerate)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_cb)
        self.get_logger().info(
            "Turtlesim Bridge running! Translating game data for the AI."
        )

    def pose_cb(self, msg):
        self.latest_pose = msg

    def cmd_cb(self, msg):
        self.latest_cmd = msg

    def timer_cb(self):
        # 1. VISUALIZATION: Create an image with a green dot where the turtle is!
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        self._canvas[:] = 0

        # OPTIMIZATION: Clamp coordinates so it never crashes
        x_px = int(np.clip((self.latest_pose.x / 11.0) * 640, 0, 639))
        y_px = int(np.clip((self.latest_pose.y / 11.0) * 480, 0, 479))

        cv2.circle(self._canvas, (x_px, 480 - y_px), 20, (0, 255, 0), -1)

        img_msg = self.bridge.cv2_to_imgmsg(self._canvas, encoding="bgr8")
        img_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )  # Required for message_filters!

        # 2. Map Turtlesim Pose to standard ROS2 Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = img_msg.header.stamp  # Sync timestamps perfectly
        odom_msg.twist.twist.linear.x = self.latest_pose.linear_velocity
        odom_msg.twist.twist.angular.z = self.latest_pose.angular_velocity

        # 3. Publish to the Imitation Hub
        self.img_pub.publish(img_msg)
        self.odom_pub.publish(odom_msg)
        self.teleop_pub.publish(self.latest_cmd)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TurtlesimBridge())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
