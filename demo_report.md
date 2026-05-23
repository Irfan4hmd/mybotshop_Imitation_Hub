# Demo Report: Imitation Hub Pipeline
**Branch:** `turtlesim_imp`  
**Focus:** Production Optimization, Hardware Agnosticism, and Safety Guardrails.

---

## 1. Executive Summary

The `turtlesim_imp` branch contains a fully functional, end-to-end runnable prototype of the
Imitation Learning pipeline. Beyond the core concept, this branch introduces clean software
architecture (base node inheritance, strategy pattern for robot types), PyTorch training
optimizations, and a deterministic safety state machine for real-time inference.

To validate the pipeline without physical hardware, ROS2's `turtlesim` is used as a proxy
simulator. The robot is taught to wander in open space via teleoperation demonstrations.
A hardcoded safety layer handles wall avoidance — a deliberate design choice explained in
Section 4.

---

## 2. Core Architecture & Hardware Agnosticism

Hardcoded topics and tensor dimensions have been eliminated. The system is parameter-driven
via a single YAML file, allowing robot profiles to be switched without any code changes.

- **Single YAML Configuration (`config/robot_params.yaml`):** All topics, dimensions, and
  robot type are defined here. Switching robots means changing one value: `robot_type`.
- **Base Node Inheritance (`base_node.py`):** `ImitationBaseNode` is a parent class for all
  executable nodes. It loads all parameters from the YAML automatically, keeping each node
  DRY. Based on `robot_type`, each node routes to the correct ROS2 message types and topics
  via `_setup_turtlesim()` or `_setup_arm()` methods.
- **Runtime Profile Switching:** `base_node.py` exposes a `switch_profile()` method and a
  ROS2 service (`/switch_robot_profile`) so the webserver backend can switch robot profiles
  at runtime without restarting nodes.

---

## 3. Data Collection & PyTorch Training

The ROS2-to-HDF5 logging pipeline and PyTorch training scripts were refactored for numerical
stability and multi-dataset support.

- **Headerless Synchronization:** Resolved a ROS2 limitation where `message_filters` drops
  teleop commands due to missing `header.stamp` metadata by using `allow_headerless=True`,
  ensuring correct state-action pairing during recording.
- **Boundary Filtering:** Frames recorded near the turtlesim walls are discarded during
  collection. Near-boundary frames contain inconsistent operator behavior (corrections,
  stops) that would pollute the dataset with contradictory demonstrations.
- **Global Z-Score Normalization:** Mean and standard deviation are computed across the
  entire aggregated dataset before training. This prevents gradient instability caused by
  raw physical measurements at different scales.
- **Self-Contained Checkpoints:** Model weights, network dimensions, and normalization
  statistics are saved together in a single `.pth` dictionary. This guarantees the inference
  node always uses the exact same scaling as training.
- **Validation Split & Best Checkpointing:** An 80/20 train/validation `random_split` is
  used. The checkpoint is only overwritten when validation loss improves, preventing the
  policy from overfitting to the training set.

---

## 4. Inference & Safety Guardrails

Deploying a learned policy onto a robot requires a clear separation between what the model
controls and what safety code controls.

- **Learned Behavior (Open-Field Wandering):** The behavior cloning model handles locomotion
  in open space. It takes the current pose `[x, y, theta]` as input and outputs velocity
  commands `[linear.x, angular.z]`. This behavior was learned from teleoperation
  demonstrations.
- **Deterministic Safety State Machine (Wall Avoidance):** Wall avoidance is intentionally
  NOT left to the learned model. When the turtle comes within 0.5 units of any wall, the
  safety layer overrides the AI and executes a timed recovery maneuver (rotate → stop →
  hand back to AI). This is a deliberate design choice — safety-critical behavior should
  remain deterministic and auditable, not learned. This mirrors how real robot systems work,
  where learned policies operate within hardcoded safety envelopes.
- **Zero-Copy Tensor Inference:** `torch.from_numpy()` with `torch.inference_mode()` is
  used for low-latency inference, disabling autograd tracking entirely. Predicted actions
  are denormalized back to real units `(predicted * std) + mean` before publishing.

---

## 5. Next Steps / Future Roadmap

1. **DAgger (Dataset Aggregation):** Implement human-in-the-loop corrections during active
   inference to combat covariate shift — the core weakness of pure behavior cloning.
2. **Action Chunking:** Modify the network to predict a window of future actions rather than
   single-step commands, improving trajectory smoothness via temporal ensembling.
3. **Visuomotor Policy:** For a real arm with a camera, replace the state-only MLP with a
   ResNet18 + MLP architecture that fuses visual features with joint state observations.
4. **Physical Hardware Deployment:** Validate the YAML profile switch on a real robotic
   manipulator using `/joint_states` and `/joint_trajectory_controller`.

---

# How to Run

### Step 0: Build & Source
Run this in the root of your workspace to ensure all the latest changes are registered:
```bash
colcon build --packages-select imitation_hub
source install/setup.bash
```

---

### Phase 1: Data Collection (Recording)
You will need 3 terminals open to play the game and record data.

**Terminal 1 (Start the Simulator):**
```bash
ros2 run turtlesim turtlesim_node
```

**Terminal 2 (Start Teleop - Keep this active to drive!):**
```bash
ros2 run turtlesim turtle_teleop_key
```

**Terminal 3 (Start the Data Collector using your YAML):**
```bash
ros2 run imitation_hub data_collector_node --ros-args --params-file config/robot.yaml
```
*(Drive the turtle around using Terminal 2 for about 30–60 seconds, then press `Ctrl+C` in Terminal 3 to save the `.h5` dataset).*

---

### Phase 2: Training the Brain
You only need 1 terminal for this. 

**Terminal 4 (Train the AI):**
```bash
ros2 run imitation_hub train_bc
```

---

### Phase 3: Autonomy (Inference)
Make sure Turtlesim (Terminal 1) is still open. Reset the turtle so it starts in the center:
```bash
ros2 service call /reset std_srvs/srv/Empty
```

**Terminal 5 (Unleash the AI using your YAML):**
```bash
ros2 run imitation_hub inference_node --ros-args --params-file config/robot.yaml
```

As soon as you run this command, the `inference_node` will read the `robot.yaml` file, see `"turtlebot"`, load the `state_dim: 3` and `action_dim: 2` settings, load the `.pth` weights, and the turtle will start driving autonomously!

***

### What if MYBOTSHOP wants to test the Arm?

```bash
# It uses the exact same command, but acts completely differently!
only the config/robot.yaml needs to be changed.
ros2 run imitation_hub inference_node --ros-args --params-file config/robot.yaml
```