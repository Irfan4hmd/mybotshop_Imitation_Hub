import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import torch
import numpy as np
import os

from imitation_hub.train_bc import VisuomotorPolicy

class InferenceNode(Node):
    def __init__(self):
        super().__init__('il_inference_node')
        self.bridge = CvBridge()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VisuomotorPolicy(joint_dim=6, action_dim=6).to(self.device)
        weights_path = "bc_model_weights.pth"
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.get_logger().info(f"Successfully loaded model weights from {weights_path}")
        else:
            self.get_logger().error(f"Weights file not found at {weights_path}! Autonomy cannot start.")
            return

        # VERY IMPORTANT: Set model to evaluation mode (disables dropout, batchnorm updates)
        self.model.eval()
        
        
        # 2. ROS2 Publishers & Subscribers
        # We publish to the trajectory controller so ros2_control can execute the movement smoothly
        self.cmd_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.joint_sub = message_filters.Subscriber(self, JointState, '/joint_states')

        # Synchronize live camera and joint states
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("Inference Node running. Robot is now in AUTONOMOUS mode.")

    def sync_callback(self, img_msg, joint_msg):
        """
        Takes live ROS2 data, passes it through PyTorch, and sends commands back to ROS2.
        """
        try:
            # --- 1. Pre-process ROS Data for PyTorch ---
            # Image: Convert to CV2 -> CHW format -> Normalize -> Tensor
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            img_array = np.transpose(cv_image, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_array).unsqueeze(0).to(self.device) # Add batch dimension [1, C, H, W]
            
            # Joints: Convert to Tensor
            joints_array = np.array(joint_msg.position, dtype=np.float32)
            joints_tensor = torch.tensor(joints_array).unsqueeze(0).to(self.device) # Add batch dimension [1, 6]

            # --- 2. Neural Network Inference ---
            # Disable gradient calculation for faster, memory-efficient inference
            with torch.no_grad():
                predicted_action = self.model(img_tensor, joints_tensor)
            
            # Convert back to numpy array on CPU
            target_joints = predicted_action.squeeze().cpu().numpy()

            # --- 3. Post-process PyTorch output to ROS Trajectory ---
            self.publish_trajectory(target_joints, joint_msg.name)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    def publish_trajectory(self, target_joints, joint_names):
        """
        Constructs a JointTrajectory message to send to ros2_control.
        """
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        point.positions = target_joints.tolist()
        # In a real scenario, we would calculate velocities based on control frequency, 
        # but for this prototype, we assign a fixed time from start.
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
