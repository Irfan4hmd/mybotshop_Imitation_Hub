import rclpy
from rclpy.node import Node
import numpy as np

# Import all possible message types here so child nodes don't have to!
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class ImitationBaseNode(Node):
    """
    Base ROS2 node that supports multiple robot profiles
    from a single YAML config file, and dynamically adapts
    message types and parsing logic.
    """

    # ---------------------------------------------------------
    # Robot Profiles (fallback defaults)
    # ---------------------------------------------------------
    ROBOT_CONFIGS = {
        "manipulator": {
            "robot_type": "manipulator",
            "camera_topic": "/camera/image_raw",
            "state_topic": "/joint_states",
            "teleop_topic": "/teleop_cmd",
            "command_topic": "/joint_trajectory_controller/joint_trajectory",
            "state_dim": 6,
            "action_dim": 6,
        },
        "turtlebot": {
            "robot_type": "turtlebot",
            "camera_topic": "/camera/image_raw",
            "state_topic": "/odom",
            "teleop_topic": "/cmd_vel_teleop",
            "command_topic": "/turtle1/cmd_vel",
            "state_dim": 2,
            "action_dim": 2,
        },
    }

    def __init__(self, node_name):
        super().__init__(node_name)

        # ---------------------------------------------------------
        # Select active robot profile
        # ---------------------------------------------------------
        self.declare_parameter("robot_profile", "turtlebot")
        self.robot_profile = (
            self.get_parameter("robot_profile").get_parameter_value().string_value
        )

        # ---------------------------------------------------------
        # Validate profile
        # ---------------------------------------------------------
        if self.robot_profile not in self.ROBOT_CONFIGS:
            raise ValueError(
                f"Unknown robot profile: {self.robot_profile}. "
                f"Available profiles: {list(self.ROBOT_CONFIGS.keys())}"
            )

        selected_config = self.ROBOT_CONFIGS[self.robot_profile]

        # ---------------------------------------------------------
        # Declare + Load parameters dynamically
        # ---------------------------------------------------------
        for param_name, default_val in selected_config.items():
            self.declare_parameter(param_name, default_val)
            param_value = self.get_parameter(param_name).value
            setattr(self, param_name, param_value)

        # ---------------------------------------------------------
        # Setup the Hardware Adapter (Message types & Parsing)
        # ---------------------------------------------------------
        self._setup_robot_interface()

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        self.get_logger().info(f"""
            ==============================
            Loaded Robot Profile: {self.robot_profile}
            Robot Type         : {self.robot_type}
            Camera Topic       : {self.camera_topic}
            State Topic        : {self.state_topic}
            Teleop Topic       : {self.teleop_topic}
            Command Topic      : {self.command_topic}
            State Dimension    : {self.state_dim}
            Action Dimension   : {self.action_dim}
            ==============================
            """)

    def _setup_robot_interface(self):
        """
        The Strategy/Adapter Pattern: Maps the correct ROS2 message types
        and parsing logic based on the robot_type, keeping child nodes 100% DRY.
        """
        if self.robot_type == "turtlebot":
            self.StateMsg = Odometry
            self.TeleopMsg = Twist
            self.CommandMsg = Twist

            # Lambda functions to instantly parse messages into numpy arrays
            self.parse_state = lambda msg: np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.angular.z], dtype=np.float32
            )
            self.parse_teleop = lambda msg: np.array(
                [msg.linear.x, msg.angular.z], dtype=np.float32
            )

            # Function to package numpy arrays back into ROS2 messages
            def create_cmd(action_array):
                msg = Twist()
                msg.linear.x = float(action_array[0])
                msg.angular.z = float(action_array[1])
                return msg

            self.create_command = create_cmd

        else:  # manipulator
            self.StateMsg = JointState
            self.TeleopMsg = Float64MultiArray
            self.CommandMsg = JointTrajectory

            self.parse_state = lambda msg: np.array(msg.position, dtype=np.float32)
            self.parse_teleop = lambda msg: np.array(msg.data, dtype=np.float32)

            def create_cmd(action_array):
                msg = JointTrajectory()
                msg.joint_names = [f"joint_{i+1}" for i in range(len(action_array))]
                point = JointTrajectoryPoint()
                point.positions = action_array.tolist()
                # 0.1 seconds interpolation time
                point.time_from_start.nanosec = 100000000
                msg.points.append(point)
                return msg

            self.create_command = create_cmd
