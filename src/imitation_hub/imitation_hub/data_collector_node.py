import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import h5py
import numpy as np
import os
from datetime import datetime

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        self.bridge = CvBridge()
        
        # Define HDF5 file to save the dataset (fast loading for PyTorch)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # taking current timestamp to generate unique filname
        self.dataset_path = f"demo_dataset_{timestamp}.h5"
        self.h5_file = h5py.File(self.dataset_path, 'w')
        
        # Create dynamically resizable HDF5 datasets
        self.max_shape = (None,)
        # 3 empty folders for camera_images, joint states and actions
        # file grows dynamically becuase of maxshape set to none
        self.img_dset = self.h5_file.create_dataset('camera_images', shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype='uint8')
        self.joint_dset = self.h5_file.create_dataset('joint_states', shape=(0, 6), maxshape=(None, 6), dtype='float32')
        self.action_dset = self.h5_file.create_dataset('actions', shape=(0, 6), maxshape=(None, 6), dtype='float32')
        
        self.dataset_size = 0

        # Subscriptions using message_filters for synchronization
        # We don't use standard subscribers because camera (30hz) and joints (100hz) publish at different rates.
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.joint_sub = message_filters.Subscriber(self, JointState, '/joint_states')
        self.action_sub = message_filters.Subscriber(self, Float64MultiArray, '/teleop_cmd') # Human commands

        # ApproximateTimeSynchronizer matches messages that arrive around the same time
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub, self.action_sub],
            queue_size=10,
            slop=0.05 # Allow 50ms difference between messages
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f"Data Collector Node started. Recording to {self.dataset_path}")

    def sync_callback(self, img_msg, joint_msg, action_msg):
        """
        This callback only fires when an image, joint state, and action command 
        arrive at approximately the same timestamp.
        """
        try:
            # 1. Convert ROS Image to Numpy Array 
            # passthrough means keep the colours as they are
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            
            # 2. Extract joint positions (current state) and teleop commands (human action)
            current_joints = np.array(joint_msg.position, dtype=np.float32)
            human_action = np.array(action_msg.data, dtype=np.float32)
            
            # 3. Save to HDF5 Dataset
            self._append_to_h5(cv_image, current_joints, human_action)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process synchronized messages: {e}")

    def _append_to_h5(self, image, joints, action):
        """ Dynamically resize HDF5 datasets and append new synchronized frame """
        self.dataset_size += 1
        
        self.img_dset.resize(self.dataset_size, axis=0)
        self.img_dset[-1] = image
        
        self.joint_dset.resize(self.dataset_size, axis=0)
        self.joint_dset[-1] = joints
        
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