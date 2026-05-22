import rclpy
from imitation_hub.base_node import ImitationBaseNode
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np

from imitation_hub.train_bc import BehaviorCloningPolicy


class InferenceNode(ImitationBaseNode):
    """
    Runs the trained behavior cloning policy and publishes commands to the robot.

    Turtlesim topic flow:
      Subscribes: /turtle1/pose   (Pose)
      Publishes:  /turtle1/cmd_vel (Twist)

    Real arm topic flow:
      Subscribes: /joint_states   (JointState)
      Publishes:  /joint_trajectory_controller/joint_trajectory (JointTrajectory)

    To run for a real arm:
        ros2 run imitation_hub inference_node --ros-args
            -p robot_type:=arm
            -p state_topic:=/joint_states
            -p command_topic:=/joint_trajectory_controller/joint_trajectory
            -p state_dim:=6
            -p action_dim:=6
    """

    def __init__(self):
        super().__init__("il_inference_node")

        self.declare_parameter("weights_path", "bc_model_weights.pth")
        weights_path = (
            self.get_parameter("weights_path").get_parameter_value().string_value
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint — weights + normalization stats saved during training
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Normalization stats must match training exactly
        self.state_mean = checkpoint["state_mean"].to(self.device)
        self.state_std = checkpoint["state_std"].to(self.device)
        self.action_mean = checkpoint["action_mean"].to(self.device)
        self.action_std = checkpoint["action_std"].to(self.device)

        # Load model dims from checkpoint — prevents silent mismatch
        # if runtime params differ from what the model was trained on
        state_dim = checkpoint["state_dim"]
        action_dim = checkpoint["action_dim"]

        self.model = BehaviorCloningPolicy(state_dim, action_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.get_logger().info(
            f"Model loaded from {weights_path} | "
            f"state_dim={state_dim} action_dim={action_dim}"
        )

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
            f"Inference Node running in AUTONOMOUS mode [{self.robot_type}]."
        )
        self.safety_trigger_time = None

    # ------------------------------------------------------------------
    # Robot-specific setup
    # ------------------------------------------------------------------

    def _setup_turtlesim(self):
        """
        Subscribes to /turtle1/pose directly — no bridge needed for inference.
        Publishes Twist to /turtle1/cmd_vel to drive the turtle.
        """
        self.cmd_pub = self.create_publisher(Twist, self.command_topic, 10)
        self.state_sub = self.create_subscription(
            Pose, self.state_topic, self._turtlesim_callback, 10
        )

    def _setup_arm(self):
        """
        Subscribes to /joint_states.
        Publishes JointTrajectory so ros2_control executes motion smoothly.
        """
        self._joint_names = None  # cached from first JointState message
        self.cmd_pub = self.create_publisher(JointTrajectory, self.command_topic, 10)
        self.state_sub = self.create_subscription(
            JointState, self.state_topic, self._arm_callback, 10
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _turtlesim_callback(self, pose_msg):
        try:
            now = self.get_clock().now()

            # ---------------------------------------------------------
            # 1. STATE MACHINE: Are we currently in a Safety Maneuver?
            # ---------------------------------------------------------
            if self.safety_trigger_time is not None:
                # Calculate how many seconds we have been in the safety maneuver
                elapsed_sec = (now - self.safety_trigger_time).nanoseconds / 1e9

                cmd = Twist()
                if elapsed_sec < 1.5:
                    # Phase 1: Rotate for 1 seconds
                    cmd.linear.x = 1.0
                    cmd.angular.z = 1.0
                    self.cmd_pub.publish(cmd)
                    return  # Skip the AI entirely

                elif elapsed_sec < 2.5:
                    # Phase 2: Stop for 1.0 seconds
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.cmd_pub.publish(cmd)
                    return  # Skip the AI entirely

                else:
                    # Phase 3: Maneuver is over! Give control back to AI.
                    self.get_logger().info(
                        "✅ Safety maneuver complete. Handing control back to AI."
                    )
                    self.safety_trigger_time = None

            # ---------------------------------------------------------
            # 2. NORMAL OPERATION: Check if we NEED to trigger safety
            # ---------------------------------------------------------
            MARGIN = 1.0
            is_too_close_to_x = (pose_msg.x < MARGIN) or (pose_msg.x > 11.0 - MARGIN)
            is_too_close_to_y = (pose_msg.y < MARGIN) or (pose_msg.y > 11.0 - MARGIN)

            if is_too_close_to_x or is_too_close_to_y:
                self.get_logger().warn(
                    "⚠️ SAFETY TRIGGERED: Turtle near wall. Initiating override sequence!"
                )
                self.safety_trigger_time = now
                return  # Skip AI for this frame, maneuver starts next frame

            # ---------------------------------------------------------
            # 3. AI CONTROL: If we are safe, let the AI drive!
            # ---------------------------------------------------------
            state = np.array([pose_msg.x, pose_msg.y, pose_msg.theta], dtype=np.float32)
            action = self._infer(state)

            cmd = Twist()
            cmd.linear.x = float(action[0])
            cmd.angular.z = float(action[1])
            self.cmd_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    def _arm_callback(self, joint_msg):
        try:
            # Cache joint names from first message
            if self._joint_names is None:
                self._joint_names = list(joint_msg.name)

            state = np.array(joint_msg.position, dtype=np.float32)
            action = self._infer(state)

            # Wrap in JointTrajectory for ros2_control
            traj = JointTrajectory()
            traj.joint_names = self._joint_names

            point = JointTrajectoryPoint()
            point.positions = action.tolist()
            point.time_from_start.nanosec = 100_000_000  # 0.1s per step

            traj.points.append(point)
            self.cmd_pub.publish(traj)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    # ------------------------------------------------------------------
    # Core inference — shared by all robot types
    # ------------------------------------------------------------------

    def _infer(self, state_array: np.ndarray) -> np.ndarray:
        """
        Normalizes state, runs the policy, denormalizes the predicted action.
        Returns numpy array in real (denormalized) action space.
        """
        state_tensor = torch.from_numpy(state_array).to(self.device)

        # Normalize — must match training normalization exactly
        state_tensor = (state_tensor - self.state_mean) / self.state_std

        with torch.inference_mode():
            predicted = self.model(state_tensor.unsqueeze(0)).squeeze()

        # Denormalize — convert back to real velocity / joint position space
        predicted = (predicted * self.action_std) + self.action_mean

        return predicted.cpu().numpy()


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


if __name__ == "__main__":
    main()
