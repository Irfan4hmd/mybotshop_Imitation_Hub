
# Demo Report: Imitation Hub Pipeline
**Branch:** `turtlesim_imp`  
**Focus:** Production Optimization, Hardware Agnosticism, and Safety Guardrails.

## 1. Executive Summary
The `turtlesim_imp` branch contains a fully optimized, end-to-end runnable structural prototype of the Imitation Learning pipeline. Beyond the core concept, this branch introduces enterprise-grade software architecture (Strategy/Adapter patterns), rigorous memory/IO optimizations for PyTorch, and a deterministic safety state-machine for real-time inference.

To prove the pipeline works end-to-end without physical hardware, it uses ROS2's `turtlesim` as a proxy, demonstrating real-time Visuomotor Policy control via synthetic vision and odometry.

---

## 2. Core Architecture & Hardware Agnosticism
Hardcoded topics and tensor dimensions have been completely eliminated. The system is now 100% parameter-driven, allowing the webserver backend to control the robot profile securely.

* **Single YAML Configuration (`config/robot_params.yaml`):** The webserver only needs to pass a single parameter (`robot_profile: "turtlebot"` or `"arm"`).
* **The Strategy/Adapter Pattern (`base_node.py`):** An `ImitationBaseNode` acts as a parent class for all executable nodes. Based on the selected profile, it dynamically routes ROS2 topics, sets PyTorch input/output dimensions (`state_dim`, `action_dim`).

---

## 3. Data Collection & PyTorch Training
The ROS2-to-HDF5 logging pipeline and PyTorch training scripts were heavily refactored for numerical stability and multi-dataset support.

* **Headerless Synchronization:** Resolved a core ROS2 limitation where `message_filters` drop human teleop commands due to missing `header.stamp` metadata by utilizing `allow_headerless=True`, ensuring perfect state-action pairing.
* **Global Z-Score Normalization:** The script computes the mean and standard deviation across the entire aggregated dataset. This prevents massive gradient spikes during backpropagation caused by raw physical measurements (like unscaled XY coordinates).
* **Self-Contained Checkpoints:** The script packages the model weights, network dimensions, and normalization statistics into a single `.pth` dictionary. This guarantees the inference node always uses the exact same scaling math.
* **Validation Split & Best-Checkpointing:** Implemented an 80/20 train/validation `random_split`. The pipeline evaluates the model at the end of each epoch and only saves the `.pth` file when the validation loss improves, mathematically preventing the policy from overfitting.

---

## 4. Inference & Safety Guardrails
Deploying AI onto physical hardware requires strict safety measures. The inference node now features a protective wrapper around the Neural Network.

* **Deterministic Safety State Machine (Virtual Bumper):** The AI policy is wrapped in a classical programmatic safety filter. If the AI hallucinates or drifts outside a safe boundary (e.g., gets within 1 meter of a wall), the software instantly intercepts the tensor and overrides the AI. It executes a non-blocking, multi-stage timed recovery maneuver using the ROS2 clock.
* **Zero-Copy Tensor Deployment:** Upgraded inference conversion to use `torch.from_numpy()` alongside `torch.inference_mode()`, disabling autograd tracking and ensuring ultra-low latency execution. Furthermore, predicted actions are accurately denormalized back into physical units (`(predicted * std) + mean`) before being published.

---

### Next Steps / Future Roadmap
1. **Dataset Aggregation (DAgger):** Fully exploit the architecture's decoupled UI/Inference design to implement human-in-the-loop corrections during active inference runs to combat covariate shift.
2. **Action Chunking:** Modify the network to predict future action windows (Temporal Ensembling) rather than single-step actions to further improve trajectory smoothness.
3. **Physical Hardware Deployment:** Test the YAML profile on an active MYBOTSHOP robotic manipulator (via `/joint_states` and `/joint_trajectory_controller`).

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
