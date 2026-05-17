import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

# ---------------------------------------------------------
# 1. PyTorch Dataset for loading HDF5 (Fast I/O)
# ---------------------------------------------------------
class RobotDemonstrationDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.file = h5py.File(h5_file_path, 'r')
        
        self.images = self.file['camera_images']
        self.joints = self.file['joint_states']
        self.actions = self.file['actions']
        
        self.length = self.images.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Read data
        img = self.images[idx]
        joints = self.joints[idx]
        action = self.actions[idx]

        # Convert Image from HWC (OpenCV/ROS) to CHW (PyTorch standard)
        img = np.transpose(img, (2, 0, 1))
        
        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0

        return (
            torch.tensor(img, dtype=torch.float32), 
            torch.tensor(joints, dtype=torch.float32), 
            torch.tensor(action, dtype=torch.float32)
        )

# ---------------------------------------------------------
# 2. Visuomotor Neural Network Architecture
# ---------------------------------------------------------
class VisuomotorPolicy(nn.Module):
    def __init__(self, joint_dim=6, action_dim=6):
        super(VisuomotorPolicy, self).__init__()
        
        # Image Feature Extractor (Simple CNN for demonstration)
        # In production, we might use a pre-trained ResNet18 here.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56 * 76, 128), # Assuming 480x640 input, flattened
            nn.ReLU()
        )
        
        # Action Predictor (Combines Image Features + Current Joints)
        self.mlp = nn.Sequential(
            nn.Linear(128 + joint_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs the predicted joint command
        )

    def forward(self, img, joints):
        # Extract features from the camera image
        img_features = self.cnn(img)
        
        # Concatenate image features with the robot's current joint states
        combined_features = torch.cat((img_features, joints), dim=1)
        
        # Predict the next action
        action_pred = self.mlp(combined_features)
        return action_pred

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def train(dataset_path, epochs=10, batch_size=32):
    print(f"Loading dataset from {dataset_path}...")
    dataset = RobotDemonstrationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = VisuomotorPolicy(joint_dim=6, action_dim=6).to(device)
    
    # MSE Loss is standard for Behavior Cloning (Regression)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, joints, actions in dataloader:
            # Move data to GPU if available
            imgs, joints, actions = imgs.to(device), joints.to(device), actions.to(device)
            # to forget the math from last batch
            optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = model(imgs, joints)
            
            # Compute Loss
            loss = criterion(predicted_actions, actions)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")

    # Save the trained model weights
    save_path = "bc_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model weights saved to {save_path}")

def main():
    # In a real scenario, the webserver API would pass the specific dataset path here.
    # We will just look for the most recently created dummy dataset.
    dataset_file = "demo_dataset_dummy.h5" 
    
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found. Please run the data collector first.")
        # For demonstration purposes in a structural prototype, we won't crash.
        return
        
    train(dataset_file)

if __name__ == "__main__":
    main()