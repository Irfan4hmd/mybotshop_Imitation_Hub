# Progress Report: Imitation Hub Prototype

## 1. What Has Been Completed
An end-to-end, runnable structural prototype for the MYBOTSHOP Imitation Hub has been implemented. The core focus was building a robust software architecture that seamlessly bridges ROS2 hardware controllers with PyTorch Deep Learning environments.

**Key Achievements:**
* **True Hardware Agnosticism:** The entire pipeline is driven by a single ROS2 YAML configuration file. By changing the `robot_type` parameter, the system automatically switches topics and neural network tensor dimensions from a 6-DOF manipulator (`JointState` / `JointTrajectory`) to a differential drive mobile base (`Odometry` / `Twist`), requiring zero code changes.
* **Synchronized Data Collection:** A custom ROS2 node captures live sensor data, pairs it with human teleoperation commands, and saves it to an HDF5 dataset for high-speed PyTorch I/O.
* **Dynamic Training Pipeline:** A PyTorch `train_bc.py` script automatically detects dataset dimensions, initializes the correct Visuomotor Policy (CNN + MLP), and trains the model via Behavior Cloning.
* **Real-time Inference:** The inference node dynamically loads the `.pth` weights, captures live ROS2 data, processes it through the neural network using `torch.no_grad()` for memory efficiency, and autonomously publishes hardware commands.

## 2. Technical Challenges & Solutions
During the integration of ROS2 and PyTorch, several system-level challenges were addressed:

**Challenge 1: Asynchronous Sensor Synchronization**
* *Issue:* Standard `message_filters.ApproximateTimeSynchronizer` drops `Twist` teleoperation messages because they lack a `header.stamp`.
* *Solution:* Built an asynchronous subscriber for the human input that keeps the latest command in memory, while strictly synchronizing the camera feeds and robot states.

**Challenge 2: Dynamic Tensor Dimensionality & Resolution**
* *Issue:* Changing cameras or robot types causes PyTorch matrix multiplication crashes (`mat1 and mat2 shapes cannot be multiplied`).
* *Solution:* Designed the PyTorch Dataset class to dynamically detect state and action dimensions directly from the HDF5 file shape. Additionally, introduced an `AdaptiveAvgPool2d` layer in the CNN to ensure the network is resolution-agnostic (e.g., automatically handling both 480p and 1080p cameras).

**Challenge 3: I/O Bottlenecks vs. RAM Limits**
* *Issue:* Training on uncompressed images caused heavy I/O bottlenecks. However, attempting to load the entire HDF5 file into RAM resulted in a system crash (Exit 247). Multiprocessing (`num_workers`) also caused HDF5 dataloader corruption.
* *Solution:* Kept data reads on the disk but optimized the pipeline by applying `cv2.resize` to shrink the image frame-by-frame before passing it to the CNN, resulting in exponentially faster GPU training without memory overflow.

---

## 3. Current Status & Next Steps
To provide a visual, runnable demonstration of the pipeline, I built a `turtlesim_bridge` node. This node translates the 2D ROS2 game into synthetic camera images (a green dot) and odometry, acting as a proxy for a mobile robot.

**Current State:** 
The software pipeline works flawlessly end-to-end. It records data, trains the model, and safely returns control to the inference node. However, the AI model's physical behavior currently looks "stupid" (e.g., spinning in place or driving into walls). 
So I dindt merge the code to main branch instead I isolated the turtlesim prototype implemetation in a turtlesim_imp branch for reference.

**The Machine Learning Reality:**
This is not a software bug, but a classic Imitation Learning phenomenon known as *Covariate Shift* or *Initial State Mismatch*. Because the model only learned what to do in the center of the room, it panics and guesses incorrectly when it drifts toward the edge of the screen.

**Next Steps:**
To resolve this, my next step is to record **Recovery Data**. I will record specific demonstrations of the turtle starting near a wall and steering away from it, then concatenate this data with the normal driving dataset. Once trained on this combined dataset for the proper number of epochs, the policy will learn how to correct its own mistakes and drive flawlessly.

**Note:**
I would also like to add that I had written the code of small 3-layer CNN just as a structural prototype for demo but for deployment on real MYBOTSHOP hardware dealing with complex lighting and real-world textures, the 3-layer CNN block can be easily swapped for a pre-trained ResNet-18/50 backbone (via torchvision) to leverage robust spatial feature extraction without changing the downstream MLP or ROS2 pipeline.

## How to Test

**Switch to turtlesim_imp branch**

### 1. Build the Workspace

```bash
colcon build
source install/setup.bash
```

### 2. Start the Simulator

```bash
ros2 run turtlesim turtlesim_node
```

Open another terminal and run:

```bash
ros2 run imitation_hub turtlesim_bridge
```

### 3. Collect Training Data

```bash
ros2 run imitation_hub data_collector_node --ros-args --params-file config/robot_params.yaml
```

Drive the robot manually using:

```bash
ros2 run turtlesim turtle_teleop_key
```

### 4. Train the Behavior Cloning Model

```bash
ros2 run imitation_hub train_bc
```

### 5. Deploy the Trained Policy

```bash
ros2 run imitation_hub inference_node --ros-args --params-file config/robot_params.yaml
```
