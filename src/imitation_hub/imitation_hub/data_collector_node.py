import rclpy
from imitation_hub.base_node import ImitationBaseNode
import message_filters
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import h5py
import numpy as np
import os, cv2
from datetime import datetime


class DataCollectorNode(ImitationBaseNode):
    def __init__(self):
        super().__init__("data_collector_node")
        self.bridge = CvBridge()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_path = f"demo_dataset_{timestamp}.h5"
        self.h5_file = h5py.File(self.dataset_path, "w")

        # OPTIMIZATION: Chunking and Compression added
        self.img_dset = self.h5_file.create_dataset(
            "camera_images",
            shape=(0, 120, 160, 3),
            maxshape=(None, 120, 160, 3),
            dtype="uint8",
            chunks=(50, 120, 160, 3),
            compression="gzip",
        )
        self.state_dset = self.h5_file.create_dataset(
            "joint_states",
            shape=(0, self.state_dim),
            maxshape=(None, self.state_dim),
            dtype="float32",
            chunks=(50, self.state_dim),
            compression="gzip",
        )
        self.action_dset = self.h5_file.create_dataset(
            "actions",
            shape=(0, self.action_dim),
            maxshape=(None, self.action_dim),
            dtype="float32",
            chunks=(50, self.action_dim),
            compression="gzip",
        )

        self.dataset_size = 0

        # OPTIMIZATION: RAM Buffers to prevent I/O bottlenecks
        self.BUFFER_SIZE = 50
        self.img_buffer, self.state_buffer, self.action_buffer = [], [], []

        self.latest_action = np.zeros(self.action_dim, dtype=np.float32)

        self.image_sub = message_filters.Subscriber(self, Image, self.camera_topic)

        self.image_sub = message_filters.Subscriber(self, Image, self.camera_topic)
        self.state_sub = message_filters.Subscriber(
            self, self.StateMsg, self.state_topic
        )
        self.create_subscription(self.TeleopMsg, self.teleop_topic, self.teleop_cb, 10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.state_sub], queue_size=50, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

    def teleop_cb(self, msg):
        self.latest_action = self.parse_teleop(msg)

    def sync_callback(self, img_msg, state_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                img_msg, desired_encoding="passthrough"
            )
            # OPTIMIZATION: Resize image BEFORE saving to save 9x disk space
            cv_image = cv2.resize(cv_image, (160, 120), interpolation=cv2.INTER_AREA)

            current_state = self.parse_state(state_msg)

            self.img_buffer.append(cv_image)
            self.state_buffer.append(current_state)
            self.action_buffer.append(self.latest_action.copy())

            # OPTIMIZATION: Bulk flush to disk
            if len(self.img_buffer) >= self.BUFFER_SIZE:
                self._flush_buffers()

        except Exception as e:
            self.get_logger().error(f"Sync error: {e}")

    def _flush_buffers(self):
        s = self.dataset_size
        n = len(self.img_buffer)

        self.img_dset.resize(s + n, axis=0)
        self.state_dset.resize(s + n, axis=0)
        self.action_dset.resize(s + n, axis=0)

        self.img_dset[s : s + n] = np.stack(self.img_buffer)
        self.state_dset[s : s + n] = np.stack(self.state_buffer)
        self.action_dset[s : s + n] = np.stack(self.action_buffer)

        self.dataset_size += n
        self.img_buffer.clear()
        self.state_buffer.clear()
        self.action_buffer.clear()

        # OPTIMIZATION: Safety flush
        if self.dataset_size % 500 == 0:
            self.h5_file.flush()

    def destroy_node(self):
        if len(self.img_buffer) > 0:
            self._flush_buffers()
        self.h5_file.close()
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


if __name__ == "__main__":
    main()
