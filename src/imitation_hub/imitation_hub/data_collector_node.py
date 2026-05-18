import rclpy
from rclpy.node import Node
from imitation_hub.base_node import ImitationBaseNode
import message_filters
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import h5py
import numpy as np
import os
from datetime import datetime

class DataCollectorNode(ImitationBaseNode):
    def __init__(self):
        super().__init__('data_collector_node')
        
        self.bridge = CvBridge()
        
        # Define HDF5 file to save the dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # timestamp = "dummy"
        self.dataset_path = f"demo_dataset_{timestamp}.h5"
        self.h5_file = h5py.File(self.dataset_path, 'w')
        
        # Create dynamically resizable HDF5 datasets
        # Notice we now use self.state_dim and self.action_dim from the YAML config!
        self.img_dset = self.h5_file.create_dataset('camera_images', shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype='uint8')
        self.state_dset = self.h5_file.create_dataset('joint_states', shape=(0, self.state_dim), maxshape=(None, self.state_dim), dtype='float32')
        self.action_dset = self.h5_file.create_dataset('actions', shape=(0, self.action_dim), maxshape=(None, self.action_dim), dtype='float32')
        
        self.dataset_size = 0

        # Keep the latest human command in memory (Fixes the missing timestamp issue)
        self.latest_action = np.zeros(self.action_dim, dtype=np.float32)

        # ---------------------------------------------------------
        # Hardware-Agnostic Subscriptions
        # ---------------------------------------------------------
        self.image_sub = message_filters.Subscriber(self, Image, self.camera_topic)
        
        if self.robot_type == "turtlebot":
            # Turtlebot setup
            self.state_sub = message_filters.Subscriber(self, Odometry, self.state_topic)
            self.create_subscription(Twist, self.teleop_topic, self.teleop_cb, 10)
        else:
            # Robotic Arm setup
            self.state_sub = message_filters.Subscriber(self, JointState, self.state_topic)
            self.create_subscription(Float64MultiArray, self.teleop_topic, self.teleop_cb, 10)

        # ApproximateTimeSynchronizer matches Image and State (Sensors with timestamps)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.state_sub],
            queue_size=10,
            slop=0.05 # Allow 50ms difference
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f"Data Collector started for {self.robot_type}. Recording to {self.dataset_path}")

    def teleop_cb(self, msg):
        """ Asynchronously updates the latest human command in memory """
        if self.robot_type == "turtlebot":
            self.latest_action = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)
        else:
            self.latest_action = np.array(msg.data, dtype=np.float32)

    def sync_callback(self, img_msg, state_msg):
        """
        Fires when an Image and Robot State arrive at the exact same time.
        We pair these with the most recent human action.
        """
        try:
            # 1. Convert ROS Image to Numpy Array
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            
            # 2. Extract state dynamically based on robot type
            if self.robot_type == "turtlebot":
                current_state = np.array([state_msg.twist.twist.linear.x, state_msg.twist.twist.angular.z], dtype=np.float32)
            else:
                current_state = np.array(state_msg.position, dtype=np.float32)
            
            # 3. Save to HDF5 Dataset (Use .copy() to ensure we don't accidentally link memory)
            self._append_to_h5(cv_image, current_state, self.latest_action.copy())
            
        except Exception as e:
            self.get_logger().error(f"Failed to process synchronized messages: {e}")

    def _append_to_h5(self, image, state, action):
        """ Dynamically resize HDF5 datasets and append new synchronized frame """
        self.dataset_size += 1
        
        self.img_dset.resize(self.dataset_size, axis=0)
        self.img_dset[-1] = image
        
        self.state_dset.resize(self.dataset_size, axis=0)
        self.state_dset[-1] = state
        
        self.action_dset.resize(self.dataset_size, axis=0)
        self.action_dset[-1] = action
        
        if self.dataset_size % 100 == 0:
            self.get_logger().info(f"Recorded {self.dataset_size} synchronized frames...")

    def destroy_node(self):
        # Ensure the HDF5 file is properly closed when node shuts down safely
        self.h5_file.close()
        self.get_logger().info(f"Dataset successfully saved with {self.dataset_size} frames.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()