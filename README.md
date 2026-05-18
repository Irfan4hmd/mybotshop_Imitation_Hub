# mybotshop_Imitation_Hub
The **MYBOTSHOP Imitation Hub** is a proposed extension to the MYBOTSHOP robotic webserver platform. It provides an end-to-end pipeline where users can interact with their hardware via a clean UI, demonstrate tasks using teleoperation, and train the robot to perform those tasks autonomously.

### Key Features
* **Seamless Data Collection:** Synchronizes camera feeds and joint states during Web UI teleoperation.
* **Training Pipeline:** Converts recorded data into Behavior Cloning models upon triggering.
* **One-Click Autonomy:** Deploys trained neural networks back to ROS2 for real-time autonomous execution.
* **YAML-Driven Hardware Agnosticism:** Seamlessly switches between 6-DOF manipulators and mobile bases without altering Python source code.

## Technology Stack
To ensure a robust, production-ready system, the following technologies were selected:
* **ROS2 (Middleware):** Provides the hardware-agnostic communication layer (`sensor_msgs`, `trajectory_msgs`, `geometry_msgs`) and hardware control via `ros2_control`.
* **PyTorch (Machine Learning):** Chosen for its dynamic computation graph, making it the industry standard for robotics Visuomotor policies and real-time inference (`torch.no_grad()`).
* **HDF5 / Zarr (Data Storage):** Chosen over standard `.db3` ROS bags because PyTorch dataloaders require lightning-fast, parallelized I/O access to matrix data, which standard ROS bags handle poorly.
* **OpenCV & cv_bridge:** Used for real-time manipulation, resizing, and normalization of visual data between ROS2 sensor feeds and PyTorch CNN inputs.

## How the Problem is Structured
The challenge of teaching a robot a physical task is framed as a **Supervised Visuomotor Regression** problem (Behavior Cloning). The system structure is broken down into three decoupled phases:

**1. Data Generation (Demonstration Phase):**
* The user teleoperates the robot via the Web UI. 
* A custom ROS2 `data_collector_node` acts as an observer. It asynchronously captures the human's teleop commands while strictly synchronizing the robot's `/camera/image_raw` and `/joint_states` (or `/odom`) using `message_filters.ApproximateTimeSynchronizer`.

**2. Policy Optimization (Training Phase):**
* The problem is structured to map an Observation (Image + Current Robot State) to a Target Action (Human Command).
* The dataset auto-detects hardware dimensions (e.g., 6-DOF vs 2-DOF) to dynamically build a **Visuomotor Policy**. A CNN extracts spatial features from the image, concatenates them with the physical robot state, and passes them through an MLP to minimize Mean Squared Error (MSE) against the human's demonstrated actions.

**3. Autonomous Deployment (Inference Phase):**
* A ROS2 `inference_node` is spun up. It loads the `.pth` weights and processes the live camera/state feeds through the network. It translates the raw PyTorch tensor outputs back into standard ROS2 messages and publishes them to the hardware controller.

## Connecting the Webserver, ROS2, and the ML Pipeline
To integrate this pipeline with the MYBOTSHOP robotic webserver, a decoupled REST API bridge is proposed. (See `api_proposal.json` for detailed endpoint structures).

1. **Triggering Data Collection:** When the user clicks "Record" on the Web UI, the web backend sends a POST request to the API. This triggers a subprocess that spins up the ROS2 `data_collector_node` in the background, listening to the Web UI's teleop topic.
2. **Triggering the ML Pipeline:** When the user clicks "Train", the webserver invokes the `train_bc.py` PyTorch script. The script directly reads the HDF5 file from disk, trains the model on the GPU/CPU, and saves the resulting `.pth` weights. The API can return a `task_id` so the Web UI can poll for training progress (e.g., current loss/epoch).
3. **Triggering Autonomy:** Upon clicking "Run Autonomy", the webserver sends an API call that gracefully kills the active teleoperation nodes and spins up the `inference_node`. The node connects the PyTorch model's outputs directly to the `ros2_control` manager. An Emergency Stop (E-Stop) API endpoint is also exposed to instantly kill this node if the AI behaves unsafely.


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
