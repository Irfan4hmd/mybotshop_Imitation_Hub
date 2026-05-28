from rclpy.node import Node


class ImitationBaseNode(Node):
    """
    A base ROS2 node that automatically loads all standard robot parameters
    using a Profile Selector.
    """

    # ---------------------------------------------------------
    # 1. Define all supported robots here (The "Single Source of Truth")
    # ---------------------------------------------------------
    ROBOT_CONFIGS = {
        "arm": {
            "robot_type": "arm",
            "state_topic": "/joint_states",
            "teleop_topic": "/teleop_cmd",
            "command_topic": "/joint_trajectory_controller/joint_trajectory",
            "state_dim": 6,
            "action_dim": 6,
        },
        "turtlebot": {
            "robot_type": "turtlesim",
            "state_topic": "/odom",
            "teleop_topic": "/cmd_vel_teleop",
            "command_topic": "/turtle1/cmd_vel",
            "state_dim": 3,  # x, y, theta
            "action_dim": 2,  # linear.x, angular.z
        },
    }

    def __init__(self, node_name):
        super().__init__(node_name)

        # ---------------------------------------------------------
        # 2. Read the YAML file to see which profile the user wants
        # ---------------------------------------------------------
        self.declare_parameter("robot_profile", "turtlebot")  # Default fallback
        self.robot_profile = (
            self.get_parameter("robot_profile").get_parameter_value().string_value
        )

        if self.robot_profile not in self.ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot profile: {self.robot_profile}")

        # ---------------------------------------------------------
        # 3. Apply the settings for that specific robot
        # ---------------------------------------------------------
        selected_config = self.ROBOT_CONFIGS[self.robot_profile]

        for param_name, param_val in selected_config.items():
            # We don't need to declare parameters here, just set them as class attributes!
            setattr(self, param_name, param_val)

        # ---------------------------------------------------------
        # 4. Log the configuration
        # ---------------------------------------------------------
        self.get_logger().info(
            f"[{node_name}] PROFILE: {self.robot_profile.upper()} | "
            f"state_topic={self.state_topic} | "
            f"state_dim={self.state_dim} action_dim={self.action_dim}"
        )
