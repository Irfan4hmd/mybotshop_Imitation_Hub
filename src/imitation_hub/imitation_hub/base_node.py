import rclpy
from rclpy.node import Node


class ImitationBaseNode(Node):
    """
    Base ROS2 node that supports multiple robot profiles
    from a single YAML config file.
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
        }
    }

    def __init__(self, node_name):
        super().__init__(node_name)

        # ---------------------------------------------------------
        # Step 1: Select active robot profile
        # ---------------------------------------------------------
        self.declare_parameter("robot_profile", "turtlebot")

        self.robot_profile = (
            self.get_parameter("robot_profile")
            .get_parameter_value()
            .string_value
        )

        # ---------------------------------------------------------
        # Step 2: Validate profile
        # ---------------------------------------------------------
        if self.robot_profile not in self.ROBOT_CONFIGS:
            raise ValueError(
                f"Unknown robot profile: {self.robot_profile}. "
                f"Available profiles: {list(self.ROBOT_CONFIGS.keys())}"
            )

        selected_config = self.ROBOT_CONFIGS[self.robot_profile]

        # ---------------------------------------------------------
        # Step 3: Declare + Load parameters dynamically
        # ---------------------------------------------------------
        for param_name, default_val in selected_config.items():

            self.declare_parameter(param_name, default_val)

            param_value = self.get_parameter(param_name).value

            setattr(self, param_name, param_value)

        # ---------------------------------------------------------
        # Step 4: Logging
        # ---------------------------------------------------------
        self.get_logger().info(
            f"""
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
            """
                    )