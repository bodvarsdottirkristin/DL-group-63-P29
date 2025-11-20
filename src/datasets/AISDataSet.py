import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split

class AISDataSet(Dataset):

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

  
def get_dataloaders(trajectories, batch_size=4, num_workers=2, seed=0, return_stats=True):
    """Create train, validation, and test dataloaders with 60/20/20 split

    Normalization is FIT on TRAIN ONLY, then applied to all splits.
    Optionally returns mean/std so results can be interpreted in original units.
    """

    # (N, T, D)
    trajectories_tensor = torch.as_tensor(trajectories, dtype=torch.float32)
    total_size = trajectories_tensor.shape[0]

    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # random split on unnormalized data to get indices
    generator = torch.Generator().manual_seed(seed)

    # temporary dataset just to be able to call random_split
    base_dataset = AISDataSet(trajectories_tensor)

    train_dataset, val_dataset, test_dataset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    # random_split returns Subset objects; grab their indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # compute mean/std on train 
    train_trajectories_tensor = trajectories_tensor[train_indices]  # (N_train, T, D)

    mean = train_trajectories_tensor.mean(dim=(0, 1), keepdim=True)  # (1, 1, D)
    std = train_trajectories_tensor.std(dim=(0, 1), keepdim=True)
    std = std.clamp_min(1e-8)  # avoid division by zero

    # normalize ALL data using train stats
    x_norm = (trajectories_tensor - mean) / std

    # rebuild dataset on normalized data
    dataset = AISDataSet(x_norm)

    # and re-create the subsets using the same indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, mean, std
