import os, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from torch.cuda.amp import GradScaler, autocast  # OPTIMIZATION: Mixed Precision


class RobotDemonstrationDataset(Dataset):
    def __init__(self, dataset_pattern="demo_dataset_*.h5"):
        self.file_paths = glob.glob(dataset_pattern)
        self.file_handles = {}  # Lazy loading prevents multiprocessing crashes

        # Get dimensions from the first file safely
        with h5py.File(self.file_paths[0], "r") as f:
            self.state_dim = f["joint_states"].shape[1]
            self.action_dim = f["actions"].shape[1]

        # --- CRITICAL FIX: Build a Global-to-Local Index Map ---
        # This maps a global index (e.g. 7864) to a specific file and local index
        self.index_map = []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                num_frames = f["camera_images"].shape[0]
                for local_idx in range(num_frames):
                    self.index_map.append((path, local_idx))

        self.total_length = len(self.index_map)
        print(
            f"Dataset initialized with {self.total_length} total frames across {len(self.file_paths)} files."
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Dynamically find the correct file and local index!
        path, local_idx = self.index_map[idx]

        # Lazy open file per-worker
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "r")

        f = self.file_handles[path]

        # Fetch using the LOCAL index
        img = f["camera_images"][local_idx]
        joints = f["joint_states"][local_idx]
        action = f["actions"][local_idx]

        # Convert Image from HWC to CHW and normalize
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

        return torch.from_numpy(img), torch.from_numpy(joints), torch.from_numpy(action)


class VisuomotorPolicy(nn.Module):
    def __init__(self, joint_dim=6, action_dim=6):
        super(VisuomotorPolicy, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128 + joint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, img, joints):
        img_features = self.cnn(img)
        combined_features = torch.cat((img_features, joints), dim=1)
        return self.mlp(combined_features)


def train(dataset_pattern, epochs=30, batch_size=64):
    dataset = RobotDemonstrationDataset(dataset_pattern)
    # OPTIMIZATION: Parallel I/O workers and pinned memory!
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisuomotorPolicy(
        joint_dim=dataset.state_dim, action_dim=dataset.action_dim
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # OPTIMIZATION: Learning Rate Scheduler & AMP Scaler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, joints, actions in dataloader:
            imgs, joints, actions = (
                imgs.to(device),
                joints.to(device),
                actions.to(device),
            )
            optimizer.zero_grad()

            # OPTIMIZATION: Automatic Mixed Precision (AMP)
            with torch.amp.autocast("cuda"):
                predicted_actions = model(imgs, joints)
                loss = criterion(predicted_actions, actions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "bc_model_weights.pth")
    print("Training complete.")


def main():
    if not glob.glob("demo_dataset_*.h5"):
        return
    """ since there is no backend integration I just gave the pattern to all available datasets but in practice we can have 
    mapping of datasets with respect to sessions and then the pattern is sent by backend server when requested """
    train("demo_dataset_*.h5", epochs=30)


if __name__ == "__main__":
    main()
