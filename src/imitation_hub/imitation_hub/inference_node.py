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
        super().__init__('il_inference_node')
        self.bridge = CvBridge()
        
        # 1. Load the PyTorch Model using dynamic dimensions from YAML
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VisuomotorPolicy(joint_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        
        weights_path = "bc_model_weights.pth"
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.get_logger().info(f"Successfully loaded model weights from {weights_path}")
        else:
            self.get_logger().error(f"Weights file not found at {weights_path}! Autonomy cannot start.")
            return

        self.model.eval()

        # 2. Hardware-Agnostic Subscriptions & Publishers
        self.image_sub = message_filters.Subscriber(self, Image, self.camera_topic)
        
        if self.robot_type == "turtlebot":
            self.cmd_pub = self.create_publisher(Twist, self.command_topic, 10)
            self.state_sub = message_filters.Subscriber(self, Odometry, self.state_topic)
        else:
            self.cmd_pub = self.create_publisher(JointTrajectory, self.command_topic, 10)
            self.state_sub = message_filters.Subscriber(self, JointState, self.state_topic)

        # Synchronize live camera and robot state
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.state_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("Inference Node running. Robot is now in AUTONOMOUS mode.")

    def sync_callback(self, img_msg, state_msg):
        try:
            # --- 1. Pre-process Image ---
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            img_array = np.transpose(cv_image, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_array).unsqueeze(0).to(self.device)
            
            # --- 2. Pre-process State dynamically ---
            if self.robot_type == "turtlebot":
                state_array = np.array([state_msg.twist.twist.linear.x, state_msg.twist.twist.angular.z], dtype=np.float32)
            else:
                state_array = np.array(state_msg.position, dtype=np.float32)
                
            state_tensor = torch.tensor(state_array).unsqueeze(0).to(self.device)

            # --- 3. Neural Network Inference ---
            with torch.no_grad():
                predicted_action = self.model(img_tensor, state_tensor)
            
            target_action = predicted_action.squeeze().cpu().numpy()

            # --- 4. Execute Movement ---
            self.publish_movement(target_action)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    def publish_movement(self, target_action):
        """ Dynamically publishes either Twist or JointTrajectory """
        if self.robot_type == "turtlebot":
            twist_msg = Twist()
            twist_msg.linear.x = float(target_action[0])
            twist_msg.angular.z = float(target_action[1])
            self.cmd_pub.publish(twist_msg)
        else:
            traj_msg = JointTrajectory()
            # For demonstration, we assume standard joint names
            traj_msg.joint_names = [f'joint_{i+1}' for i in range(self.action_dim)]
            point = JointTrajectoryPoint()
            point.positions = target_action.tolist()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 100000000 # 0.1 seconds
            traj_msg.points.append(point)
            self.cmd_pub.publish(traj_msg)

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

if __name__ == '__main__':
    main()