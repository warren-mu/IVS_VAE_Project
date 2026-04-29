import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class IVSDataset(Dataset):
    """
    Custom Dataset for Implied Volatility Surfaces (IVS).
    Loads .npy files and applies random masks to simulate sparsity.
    """
    def __init__(self, surface_path, param_path, sparsity_range=(0.5, 0.95)):
        # Load the generated datasets [cite: 20]
        self.surfaces = np.load(surface_path)
        self.params = np.load(param_path)
        self.sparsity_range = sparsity_range

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, idx):
        # 1. Get ground truth surface (10, 20)
        surface = torch.from_numpy(self.surfaces[idx]).float()
        
        # 2. Generate a random binary mask to simulate data missingness 
        sparsity = np.random.uniform(*self.sparsity_range)
        mask = torch.bernoulli(torch.full(surface.shape, 1 - sparsity))
        
        # 3. Apply mask (input for the model during inference/LSO)
        sparse_surface = surface * mask
        
        return {
            'target': surface,          # Complete surface for Loss calculation
            'input': sparse_surface,    # Sparse surface
            'mask': mask,               # Mask location
            'params': torch.from_numpy(self.params[idx]).float()
        }

def get_dataloader(surface_path, param_path, batch_size=32, shuffle=True):
    dataset = IVSDataset(surface_path, param_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)