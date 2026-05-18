# mybotshop_Imitation_Hub
The **MYBOTSHOP Imitation Hub** is a proposed extension to the MYBOTSHOP robotic webserver platform. It provides an end-to-end pipeline where users can interact with their hardware via a clean UI, demonstrate tasks using teleoperation, and train the robot to perform those tasks autonomously.

### Key Features
* **Seamless Data Collection:** Synchronizes camera feeds and joint states during Web UI teleoperation.
* **Training Pipeline:** Converts recorded data into Behavior Cloning models upon triggering.
* **One-Click Autonomy:** Deploys trained neural networks back to ROS2 for real-time autonomous execution.
* **YAML-Driven Hardware Agnosticism:** Seamlessly switches between 6-DOF manipulators and mobile bases without altering Python source code.

# The Architecture of the Imitation Hub pipeline

```mermaid
graph TD
    %% Web Interface Level
    subgraph Web_Server_Platform ["MYBOTSHOP Webserver (UI & API)"]
        UI_Teleop[Teleoperation UI]
        UI_Train[Trigger Training UI]
        UI_Auto[Autonomy UI]
    end

    %% ROS2 Environment Level
    subgraph ROS2_Environment ["ROS2 Workspace (Robot Agnostic)"]
        Hardware[(Robot Hardware / Sim)]
        Controller_Manager[ros2_control / Controller Manager]
        
        Teleop_Node[Teleop Relay Node]
        Data_Collector[Data Collector Node]
        Inference_Node[IL Inference Node]
        
        %% Topics & Controllers
        Topic_Teleop([/teleop_cmd <br> e.g. Forward Position])
        Topic_Sensors([/camera/image_raw <br> /joint_states])
        Topic_Control([/joint_trajectory_controller])
    end

    %% Machine Learning Level
    subgraph ML_Pipeline ["Imitation Learning Pipeline"]
        Dataset[(Dataset: HDF5 / Zarr)]
        PyTorch_Trainer[PyTorch Training Script]
        Model_Weights[(Model Weights: .pth)]
    end

    %% Flow: Data Collection
    UI_Teleop -- "1. User Commands" --> Teleop_Node
    Teleop_Node -- "Streams to" --> Topic_Teleop
    Topic_Teleop -- "Drives" --> Controller_Manager
    Controller_Manager -- "Actuates" --> Hardware
    Hardware -- "Publishes Live Data" --> Topic_Sensors
    
    Topic_Teleop --> Data_Collector
    Topic_Sensors --> Data_Collector
    Data_Collector -- "Synchronizes & Saves" --> Dataset

    %% Flow: Training
    UI_Train -- "2. Start API Call" --> PyTorch_Trainer
    Dataset -- "Loads Data" --> PyTorch_Trainer
    PyTorch_Trainer -- "Saves Model" --> Model_Weights

    %% Flow: Inference (Autonomy)
    UI_Auto -- "3. Start API Call" --> Inference_Node
    Model_Weights -- "Loads Model" --> Inference_Node
    Topic_Sensors -- "Live Observations" --> Inference_Node
    Inference_Node -- "Predicts Trajectory" --> Topic_Control
    Topic_Control -- "Drives" --> Controller_Manager

    %% Styling
    classDef web fill:#f9f,stroke:#333,stroke-width:2px;
    classDef ros fill:#bbf,stroke:#333,stroke-width:2px;
    classDef ml fill:#bfb,stroke:#333,stroke-width:2px;
```

# The Architecture overview:
The architecture of pipeline is designed to be highly modular and hardware-agnostic. By relying on standard ROS2 interfaces (sensor_msgs, trajectory_msgs), the pipeline can be attached to any robot.

We have 3 distinct phases:

**1. Data Collection:**

   * Trigger: The user initiates a recording session while teleoperating the robot via the Web UI. The UI sends actions to the robot's controller manager, and upon               execution, the /joint_states are updated.
   * Process: A custom ROS2 data_collector_node subscribes to the human's teleop commands, the robot's /joint_states, and the /camera/image_raw.
   * Storage: Using message_filters.ApproximateTimeSynchronizer, the node synchronizes the images with the joint states and saves them directly into an HDF5 or Zarr              format. (I would choose Zarr/HDF5 over standard rosbag2 because it allows for lightning-fast, parallelized dataloading in PyTorch).
    
    
**2. Training the Brain(Model):**

   * Trigger: The user clicks "Train Model" on the webserver, which sends an API call to the backend.
   * Process: A PyTorch training pipeline reads the HDF5 dataset. The task is framed as Behavior Cloning: the model takes an observation (Image + Current Joints) and learns      to predict the human's action (Target Joints).
   * Output: The script outputs weights file (.pth) obtained after training.

**3. Autonomous Execution:**

   * Trigger: The user selects the trained model and starts autonomy via the Web UI.
   * Process: The ROS2 inference_node is spun up. It loads the PyTorch model, subscribes to the live /camera/image_raw and /joint_states, passes them through the neural          network, and continuously publishes the predicted actions to the robot's /joint_trajectory_controller.

### Design Choices & Trade-offs
**Why a Manual Training Trigger instead of Automated Cron Jobs?**

While continuous integration and scheduled training (e.g., via cron jobs) are standard in traditional ML, Imitation Learning requires strict data quality control. Teleoperation data is inherently noisy (users make mistakes, drop objects, or pause). Providing a manual **"Trigger Training"** endpoint allows for Human-in-the-Loop validation—ensuring the user verifies the quality of the recorded demonstrations before initiating a compute-heavy GPU training process. It also provides immediate feedback, allowing users to train and test a specific task instantly rather than waiting for scheduled batch jobs.

**How's this architecture Hardware-agnostic?**

You will notice the architecture routes both `/teleop_cmd` and `/joint_trajectory_controller` through the `ros2_control` Manager. Teleoperation via the Web UI typically requires a streaming controller (like a Forward Position/Velocity Controller or MoveIt Servo) for responsive human input. Conversely, the Inference Node can predict action chunks and execute them smoothly via the `joint_trajectory_controller`, allowing the system to remain hardware-agnostic while ensuring safe, interpolated movements.To fulfill the requirement that this platform must work with different types of robots (humanoids, arms, mobile bases), hardcoding topic names and AI network dimensions is avoided. The entire pipeline is driven by a central ROS2 Parameter File (config/robot_params.yaml).

**Resolution Independence & I/O Optimization**

The PyTorch VisuomotorPolicy integrates an AdaptiveAvgPool2d layer. This mathematically guarantees a fixed feature vector size entering the MLP, preventing matrix multiplication crashes regardless of the physical camera resolution used (e.g., 480p vs 1080p). Furthermore, cv2.resize is applied frame-by-frame during the PyTorch Dataloader phase to bypass severe I/O bottlenecks and RAM overflow, resulting in exponentially faster GPU training speeds.

**Controller Management**

The architecture routes both /teleop_cmd and /joint_trajectory_controller through the ros2_control Manager. Teleoperation via the Web UI typically requires a streaming controller (like a Forward Position/Velocity Controller) for responsive human input. Conversely, the Inference Node safely interpolates AI action chunks via the joint_trajectory_controller, allowing the system to remain hardware-agnostic while ensuring safe, continuous movements.
