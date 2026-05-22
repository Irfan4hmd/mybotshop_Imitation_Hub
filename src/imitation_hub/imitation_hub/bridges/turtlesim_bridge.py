import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from turtlesim.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np


class TurtlesimBridge(Node):
    """
    Provides two things the data collector needs during teleoperation:

      1. Fake camera image (/camera/image_raw)
         A black canvas with a green dot at the turtle's position.
         Published at 30Hz with a synchronized timestamp so
         message_filters can pair it with other topics if needed.

      2. Timestamped teleop commands (/cmd_vel_teleop)
         Forwards /turtle1/cmd_vel with a fresh timestamp so
         ApproximateTimeSynchronizer can match it to the pose.

    State (pose) is NOT republished here — data_collector_node and
    inference_node subscribe to /turtle1/pose directly.

    NOTE: This bridge is only needed during DATA COLLECTION.
          The inference node does not use it.
    """

    def __init__(self):
        super().__init__("turtlesim_bridge")
        self.bridge = CvBridge()
        self._canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_pose = Pose()
        self.latest_cmd = Twist()

        # Inputs — listen to turtlesim
        self.create_subscription(Pose, "/turtle1/pose", self.pose_cb, 10)
        self.create_subscription(Twist, "/turtle1/cmd_vel", self.cmd_cb, 10)

        # Outputs — feed the data collector
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.teleop_pub = self.create_publisher(Twist, "/cmd_vel_teleop", 10)

        # 30Hz timer — matches standard camera framerate
        self.timer = self.create_timer(1.0 / 30.0, self.timer_cb)
        self.get_logger().info("Turtlesim Bridge running.")

    def pose_cb(self, msg):
        self.latest_pose = msg

    def cmd_cb(self, msg):
        self.latest_cmd = msg

    def timer_cb(self):
        # Draw turtle as a green dot on a black canvas
        # Turtlesim world is 11x11 units — map to 640x480 pixels
        self._canvas[:] = 0
        x_px = int(np.clip((self.latest_pose.x / 11.0) * 640, 0, 639))
        y_px = int(np.clip((self.latest_pose.y / 11.0) * 480, 0, 479))
        cv2.circle(self._canvas, (x_px, 480 - y_px), 20, (0, 255, 0), -1)

        # Single timestamp shared by both messages — critical for
        # ApproximateTimeSynchronizer to match them correctly
        stamp = self.get_clock().now().to_msg()

        img_msg = self.bridge.cv2_to_imgmsg(self._canvas, encoding="bgr8")
        img_msg.header.stamp = stamp

        self.img_pub.publish(img_msg)
        self.teleop_pub.publish(self.latest_cmd)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TurtlesimBridge())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
