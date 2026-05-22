from rclpy.node import Node


class ImitationBaseNode(Node):
    """
    A base ROS2 node that automatically loads all standard robot parameters.
    Other nodes will inherit from this to stay DRY.

    Defaults are set for turtlesim demo.
    For a real arm, override via ROS2 launch file:
        ros2 run imitation_hub <node> --ros-args
            -p robot_type:=arm
            -p state_topic:=/joint_states
            -p teleop_topic:=/teleop_cmd
            -p command_topic:=/joint_trajectory_controller/joint_trajectory
            -p state_dim:=6
            -p action_dim:=6
    """

    def __init__(self, node_name):
        super().__init__(node_name)

        default_params = {
            # Robot type drives which ROS message types to use
            # Supported: 'turtlesim' | 'arm'
            "robot_type": "turtlesim",
            # State: turtlesim publishes Pose directly — no bridge needed
            "state_topic": "/turtle1/pose",
            # Teleop: bridge forwards /turtle1/cmd_vel here so we can
            # timestamp-sync it with the fake camera image
            "teleop_topic": "/cmd_vel_teleop",
            # Command: inference node publishes here to drive the robot
            "command_topic": "/turtle1/cmd_vel",
            # Dimensions
            # turtlesim: state=[x, y, theta], action=[linear.x, angular.z]
            # arm:       state=6 joint positions, action=6 joint commands
            "state_dim": 3,
            "action_dim": 2,
        }

        for param_name, default_val in default_params.items():
            self.declare_parameter(param_name, default_val)
            setattr(self, param_name, self.get_parameter(param_name).value)

        self.get_logger().info(
            f"[{node_name}] robot_type={self.robot_type} | "
            f"state_topic={self.state_topic} | "
            f"state_dim={self.state_dim} action_dim={self.action_dim}"
        )
