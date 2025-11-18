import torch
from torch.utils.data import DataLoader, Subset, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AISDataloader(Dataset):

  def __init__(self, trajectories: torch.Tensor):
    """
    trajectories: torch.Tensor of shape (N, T, D)
    """
    self.trajectories = trajectories

  def __len__(self):
    """ returns number of sequences """
    return self.trajectories.shape[0]

  def __getitem__(self, idx):
    """ returns (T, D) """
    return self.trajectories[idx]

  @staticmethod
  def get_dataloaders(trajectories, batch_size=4, num_workers=2, return_stats=False):
    """Create train, validation, and test dataloaders with 60/20/20 split"""

    trajectories_tensor = torch.as_tensor(trajectories, dtype=torch.float32)   # (N, T, D)
    N = trajectories_tensor.shape[0]

    # TRAIN ONLY subset for normalization
    train_trajectories_tensor = trajectories_tensor[:train_end]     # (N_train, T, D)

    # Compute normalization on train only
    mean = train_trajectories_tensor.mean(dim=(0, 1), keepdim=True)  # (1, 1, D)
    std = train_trajectories_tensor.std(dim=(0, 1), keepdim=True)
    std = std.clamp_min(1e-8)                      # avoid division by zero

    # Normalize all data using train stats
    x_norm = (trajectories_tensor - mean) / std

    # Build Dataset from normalized tensor
    dataset = AISDataloader(x_norm)

    train_end = int(0.6 * N)
    val_end = int(0.8 * N)

    train_indices = list(range(0, train_end))
    val_indices   = list(range(train_end, val_end))
    test_indices  = list(range(val_end, N))

    train_dataset, val_dataset, test_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

    train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    if return_stats:
        return train_loader, val_loader, test_loader, mean, std
    return train_loader, val_loader, test_loader


