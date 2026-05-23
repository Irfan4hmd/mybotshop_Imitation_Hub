import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np


# ---------------------------------------------------------
# 1. Dataset — works for any robot, no images required
#    HDF5 schema: joint_states (N, state_dim), actions (N, action_dim)
# ---------------------------------------------------------
class RobotDemonstrationDataset(Dataset):
    """
    Loads (state, action) pairs from one or more HDF5 files.
    No camera images — state-only policy.

    For turtlesim: state_dim=3 (x, y, theta), action_dim=2 (linear, angular)
    For a real arm: state_dim=6, action_dim=6
    """

    def __init__(self, dataset_pattern="demo_dataset_*.h5"):
        self.file_paths = glob.glob(dataset_pattern)
        if not self.file_paths:
            raise FileNotFoundError(f"No files found matching: {dataset_pattern}")

        self.file_handles = {}  # Lazy open — safe for multiprocessing DataLoader

        with h5py.File(self.file_paths[0], "r") as f:
            self.state_dim = f["joint_states"].shape[1]
            self.action_dim = f["actions"].shape[1]

        # Build a flat global index map: global_idx -> (file_path, local_idx)
        self.index_map = []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                for i in range(f["joint_states"].shape[0]):
                    self.index_map.append((path, i))

        print(
            f"Dataset: {len(self.index_map)} frames across {len(self.file_paths)} file(s) "
            f"| state_dim={self.state_dim} action_dim={self.action_dim}"
        )

        # Compute normalization statistics from entire dataset
        print("Computing normalization stats...")
        all_states, all_actions = [], []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                all_states.append(f["joint_states"][:])
                all_actions.append(f["actions"][:])

        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        self.state_mean = torch.tensor(all_states.mean(axis=0), dtype=torch.float32)
        self.state_std = torch.tensor(
            all_states.std(axis=0) + 1e-8, dtype=torch.float32
        )
        self.action_mean = torch.tensor(all_actions.mean(axis=0), dtype=torch.float32)
        self.action_std = torch.tensor(
            all_actions.std(axis=0) + 1e-8, dtype=torch.float32
        )

        print(f"State  mean={self.state_mean.numpy()}  std={self.state_std.numpy()}")
        print(f"Action mean={self.action_mean.numpy()} std={self.action_std.numpy()}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        path, local_idx = self.index_map[idx]

        # Open file lazily — each DataLoader worker opens its own handle
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "r")

        f = self.file_handles[path]
        state = torch.from_numpy(f["joint_states"][local_idx])
        action = torch.from_numpy(f["actions"][local_idx])

        # Normalize to zero mean, unit variance
        state = (state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std

        return state, action


# ---------------------------------------------------------
# 2. Policy — generic MLP, works for any state/action dims
#
#    For turtlesim: state_dim=3, action_dim=2
#    For a real arm: state_dim=6, action_dim=6
#
#    NOTE: For a real arm with a camera, replace this with
#    VisuomotorPolicy (ResNet18 + MLP). State-only is used
#    here because turtlesim has no camera.
# ---------------------------------------------------------
class BehaviorCloningPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BehaviorCloningPolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        return self.net(state)


# ---------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------
def train(dataset_pattern="demo_dataset_*.h5", epochs=100, batch_size=64):
    dataset = RobotDemonstrationDataset(dataset_pattern)

    # Train / val split — 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {train_size} train / {val_size} val")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Sanity check — baseline MSE if model just predicts the mean
    for states, actions in train_loader:
        baseline = (actions - actions.mean(dim=0)).pow(2).mean()
        print(f"Baseline MSE (predict mean): {baseline.item():.4f}")
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = BehaviorCloningPolicy(dataset.state_dim, dataset.action_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            loss = criterion(model(states), actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                val_loss += criterion(model(states), actions).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # Save best checkpoint — includes normalization stats for inference node
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": dataset.state_dim,
                    "action_dim": dataset.action_dim,
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "action_mean": dataset.action_mean,
                    "action_std": dataset.action_std,
                },
                "bc_model_weights.pth",
            )
            saved = " <- best, saved"
        else:
            saved = ""

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | Train: {avg_train:.4f} | Val: {avg_val:.4f}{saved}"
        )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------
def main():
    if not glob.glob("demo_dataset_*.h5"):
        print("No dataset files found. Run data_collector_node first.")
        return
    train("demo_dataset_*.h5", epochs=100)


if __name__ == "__main__":
    main()
