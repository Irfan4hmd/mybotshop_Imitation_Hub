import rclpy
from imitation_hub.base_node import ImitationBaseNode
import message_filters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import h5py
import numpy as np
from datetime import datetime


class DataCollectorNode(ImitationBaseNode):
    """
    Records synchronized (state, action) pairs to HDF5 during teleoperation.

    Turtlesim topic flow (handled by turtlesim_bridge.py):
      /turtle1/pose      -> state  [x, y, theta]
      /cmd_vel_teleop    -> action [linear.x, angular.z]
                           (bridge forwards /turtle1/cmd_vel here with
                            a synchronized timestamp for message_filters)

    Real arm topic flow:
      /joint_states      -> state  [6 joint positions]
      /teleop_cmd        -> action [6 joint commands]

    To run for a real arm:
        ros2 run imitation_hub data_collector_node --ros-args
            -p robot_type:=arm
            -p state_topic:=/joint_states
            -p teleop_topic:=/teleop_cmd
            -p state_dim:=6
            -p action_dim:=6
    """

    def __init__(self):
        super().__init__("data_collector_node")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_path = f"demo_dataset_{timestamp}.h5"
        self.h5_file = h5py.File(self.dataset_path, "w")

        # HDF5 datasets grow dynamically — maxshape=None means unlimited rows
        self.state_dset = self.h5_file.create_dataset(
            "joint_states",
            shape=(0, self.state_dim),
            maxshape=(None, self.state_dim),
            dtype="float32",
        )
        self.action_dset = self.h5_file.create_dataset(
            "actions",
            shape=(0, self.action_dim),
            maxshape=(None, self.action_dim),
            dtype="float32",
        )
        self.dataset_size = 0

        if self.robot_type == "turtlesim":
            self._setup_turtlesim()
        elif self.robot_type == "arm":
            self._setup_arm()
        else:
            self.get_logger().error(
                f"Unknown robot_type: '{self.robot_type}'. Use 'turtlesim' or 'arm'."
            )
            return

        self.get_logger().info(
            f"Data Collector started. Recording to {self.dataset_path}"
        )

    # ------------------------------------------------------------------
    # Robot-specific setup
    # ------------------------------------------------------------------

    def _setup_turtlesim(self):
        self.state_sub = message_filters.Subscriber(self, Pose, self.state_topic)
        self.action_sub = message_filters.Subscriber(self, Twist, self.teleop_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.state_sub, self.action_sub],
            queue_size=20,
            slop=0.05,
            allow_headerless=True,  # <-- add this
        )
        self.ts.registerCallback(self._turtlesim_callback)

    def _setup_arm(self):
        """
        Subscribes to:
          - /joint_states  (JointState) — current joint positions
          - /teleop_cmd    (JointState) — operator joint commands
        """
        self.state_sub = message_filters.Subscriber(self, JointState, self.state_topic)
        self.action_sub = message_filters.Subscriber(
            self, JointState, self.teleop_topic
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.state_sub, self.action_sub], queue_size=20, slop=0.05
        )
        self.ts.registerCallback(self._arm_callback)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _turtlesim_callback(self, pose_msg, twist_msg):
        try:
            state = np.array([pose_msg.x, pose_msg.y, pose_msg.theta], dtype=np.float32)
            action = np.array(
                [twist_msg.linear.x, twist_msg.angular.z], dtype=np.float32
            )
            self._append_to_h5(state, action)
        except Exception as e:
            self.get_logger().error(f"Turtlesim callback error: {e}")

    def _arm_callback(self, state_msg, action_msg):
        try:
            state = np.array(state_msg.position, dtype=np.float32)
            action = np.array(action_msg.position, dtype=np.float32)
            self._append_to_h5(state, action)
        except Exception as e:
            self.get_logger().error(f"Arm callback error: {e}")

    # ------------------------------------------------------------------
    # HDF5 writing
    # ------------------------------------------------------------------

    def _append_to_h5(self, state, action):
        self.dataset_size += 1

        self.state_dset.resize(self.dataset_size, axis=0)
        self.state_dset[-1] = state

        self.action_dset.resize(self.dataset_size, axis=0)
        self.action_dset[-1] = action

        if self.dataset_size % 100 == 0:
            self.get_logger().info(f"Recorded {self.dataset_size} frames...")

    def destroy_node(self):
        self.h5_file.close()
        self.get_logger().info(
            f"Dataset saved: {self.dataset_path} ({self.dataset_size} frames)"
        )
        super().destroy_node()  # remove rclpy.shutdown() from here if present


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
