import cv2
import rclpy
from imitation_hub.base_node import ImitationBaseNode
import message_filters
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import torch
import numpy as np
import os

from imitation_hub.train_bc import VisuomotorPolicy


class InferenceNode(ImitationBaseNode):
    def __init__(self):
        super().__init__("il_inference_node")
        self.bridge = CvBridge()
        self._setup_robot_interface()

        self.get_logger().info(f"Loaded config: Type={self.robot_type}...")
        # OPTIMIZATION: Parameterize weights path
        self.declare_parameter("weights_path", "bc_model_weights.pth")
        weights_path = (
            self.get_parameter("weights_path").get_parameter_value().string_value
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisuomotorPolicy(
            joint_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 2. Hardware-Agnostic Subscriptions & Publishers
        self.image_sub = message_filters.Subscriber(self, Image, self.camera_topic)

        self.cmd_pub = self.create_publisher(self.CommandMsg, self.command_topic, 10)
        self.state_sub = message_filters.Subscriber(
            self, self.StateMsg, self.state_topic
        )

        # Synchronize live camera and robot state
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.state_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(
            "Inference Node running. Robot is now in AUTONOMOUS mode."
        )

    def sync_callback(self, img_msg, state_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                img_msg, desired_encoding="passthrough"
            )
            # CRITICAL FIX: Match training resolution
            cv_image = cv2.resize(cv_image, (160, 120), interpolation=cv2.INTER_AREA)
            img_array = np.transpose(cv_image, (2, 0, 1)).astype(np.float32) / 255.0

            # OPTIMIZATION: Zero-copy tensor creation
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(self.device)

            state_array = self.parse_state(state_msg)

            state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)

            # OPTIMIZATION: Strict inference mode
            with torch.inference_mode():
                predicted_action = self.model(img_tensor, state_tensor)

            raw_action = predicted_action.squeeze().cpu().numpy()

            self.previous_action = raw_action  # Save for the next frame
            cmd_msg = self.create_command(raw_action)
            self.cmd_pub.publish(cmd_msg)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
