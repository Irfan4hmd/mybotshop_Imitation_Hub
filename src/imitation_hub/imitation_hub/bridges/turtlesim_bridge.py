import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from turtlesim.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class TurtlesimBridge(Node):
    """
    CRITICAL ARCHITECTURE NODE:
    Provides three perfectly synchronized, stamped streams for the Data Collector:

      1. Fake camera image (/camera/image_raw)
         Demonstrates handling of high-bandwidth computer vision streams.

      2. Timestamped robot state (/odom)
         Turtlesim's native 'Pose' lacks a header. This node wraps it into
         standard Odometry so the Synchronizer can perfectly match it.

      3. Timestamped teleop commands (/cmd_vel_teleop)
         Acts as a Zero-Order Hold filter, republishing sparse human
         commands at a continuous 30Hz with a matching timestamp.
    """

    def __init__(self):
        super().__init__("turtlesim_bridge")
        self.bridge = CvBridge()
        self._canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_pose = Pose()
        self.latest_cmd = Twist()

        # Inputs — listen to raw turtlesim data (no headers)
        self.create_subscription(Pose, "/turtle1/pose", self.pose_cb, 10)
        self.create_subscription(Twist, "/turtle1/cmd_vel", self.cmd_cb, 10)

        # Outputs — publish production-ready, perfectly stamped data
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.teleop_pub = self.create_publisher(TwistStamped, "/cmd_vel_teleop", 10)

        # 30Hz timer — matches standard camera framerate
        self.timer = self.create_timer(1.0 / 30.0, self.timer_cb)
        self.get_logger().info(
            "Turtlesim Bridge running: Injecting perfect timestamps at 30Hz."
        )

    def pose_cb(self, msg):
        self.latest_pose = msg

    def cmd_cb(self, msg):
        self.latest_cmd = msg

    def timer_cb(self):
        # Draw turtle as a green dot on a black canvas
        self._canvas[:] = 0
        x_px = int(np.clip((self.latest_pose.x / 11.0) * 640, 0, 639))
        y_px = int(np.clip((self.latest_pose.y / 11.0) * 480, 0, 479))
        cv2.circle(self._canvas, (x_px, 480 - y_px), 20, (0, 255, 0), -1)

        # --- 1. Generate ONE Unified Timestamp ---
        # Critical for ApproximateTimeSynchronizer to match all 3 streams perfectly!
        stamp = self.get_clock().now().to_msg()

        # --- 2. Package Image ---
        img_msg = self.bridge.cv2_to_imgmsg(self._canvas, encoding="bgr8")
        img_msg.header.stamp = stamp

        # --- 3. Package State (Odometry) ---
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.pose.pose.position.x = self.latest_pose.x
        odom_msg.pose.pose.position.y = self.latest_pose.y

        odom_msg.pose.pose.orientation.w = math.cos(self.latest_pose.theta / 2.0)
        odom_msg.pose.pose.orientation.z = math.sin(self.latest_pose.theta / 2.0)

        # --- 4. Package Action (TwistStamped) ---
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = stamp
        twist_stamped.twist = self.latest_cmd

        # --- 5. Publish Synchronously ---
        self.img_pub.publish(img_msg)
        self.odom_pub.publish(odom_msg)
        self.teleop_pub.publish(twist_stamped)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TurtlesimBridge())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
