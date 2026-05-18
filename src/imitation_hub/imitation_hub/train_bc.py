import glob
import os
import cv2
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

        self.state_dim = self.joints.shape[1]
        self.action_dim = self.actions.shape[1]

    def __len__(self):
        # return self.length
        return min(self.length, 2000)


    def __getitem__(self, idx):
        # Read data
        img = self.images[idx]
        joints = self.joints[idx]
        action = self.actions[idx]
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
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
        
        # Image Feature Extractor (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            
            # --- THE MAGIC FIX ---
            # This forces the output of the CNN to ALWAYS be 5x5 pixels, 
            # regardless of whether the camera is 480p, 720p, or 1080p!
            nn.AdaptiveAvgPool2d((5, 5)), 
            nn.Flatten(),
            
            # Now we know the flattened size is exactly 64 channels * 5 * 5 = 1600
            nn.Linear(64 * 5 * 5, 128), 
            nn.ReLU()
        )
        
        # Action Predictor (Combines Image Features + Current Joints)
        self.mlp = nn.Sequential(
            nn.Linear(128 + joint_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs the predicted commands
        )

    def forward(self, img, joints):
        # Extract features from the camera image
        img_features = self.cnn(img)
        
        # Concatenate image features with the robot's current states
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # --- NEW CODE: Use the dynamically detected dimensions! ---
    print(f"Auto-detected architecture: State Dim = {dataset.state_dim}, Action Dim = {dataset.action_dim}")
    model = VisuomotorPolicy(joint_dim=dataset.state_dim, action_dim=dataset.action_dim).to(device)
    
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
    
    list_of_files = glob.glob('demo_dataset_*.h5')
    
    if not list_of_files:
        print("Error: No datasets found. Please run the data collector first.")
        return
        
    # Magically select the one that was created most recently!
    latest_dataset = max(list_of_files, key=os.path.getctime)
    
    print(f"Auto-selected the newest dataset: {latest_dataset}")
    train(latest_dataset,epochs=30)

if __name__ == "__main__":
    main()