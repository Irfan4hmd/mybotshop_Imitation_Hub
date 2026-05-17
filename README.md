# mybotshop_Imitation_Hub
MYBOTSHOP Imitation Hub: A conceptual ROS2-to-PyTorch pipeline that allows users to teleoperate robots via a web UI, collect vision-based data, and train the robot to perform tasks autonomously using Imitation Learning.


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
        Teleop_Node[Teleop Relay Node]
        Data_Collector[Data Collector Node]
        Inference_Node[IL Inference Node]
        
        %% Topics
        Topic_Teleop([/teleop_cmd])
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
    Teleop_Node -- "Publishes" --> Topic_Teleop
    Topic_Teleop -- "Moves" --> Hardware
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
    Inference_Node -- "Predicts Actions" --> Topic_Control
    Topic_Control -- "Executes" --> Hardware

    %% Styling
    classDef web fill:#f9f,stroke:#333,stroke-width:2px;
    classDef ros fill:#bbf,stroke:#333,stroke-width:2px;
    classDef ml fill:#bfb,stroke:#333,stroke-width:2px;
