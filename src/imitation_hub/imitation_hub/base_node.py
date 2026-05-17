import rclpy
from rclpy.node import Node
class ImitationBaseNode(Node):
    """
    A base ROS2 node that automatically loads all standard robot parameters.
    Other nodes will inherit from this to stay DRY.
    """
    def __init__(self, node_name):
        super().__init__(node_name)
        
        # 1. Single Source of Truth for Default Parameters
        default_params = {
            'camera_topic': '/camera/image_raw',
            'state_topic': '/joint_states',
            'teleop_topic': '/teleop_cmd',
            'command_topic': '/joint_trajectory_controller/joint_trajectory',
            'state_dim': 6,
            'action_dim': 6
        }

        # 2. The DRY Loop
        for param_name, default_val in default_params.items():
            self.declare_parameter(param_name, default_val)
            setattr(self, param_name, self.get_parameter(param_name).value)
            
        self.get_logger().info(f"Loaded config for {node_name}: State Topic = {self.state_topic}, Dims = {self.state_dim}")