import torch
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from vae_model import VAE
from dataset import IVSDataset

def run_lso():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    latent_dim = 10
    model = VAE(input_dim=200, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "data/vae_model.pth")))
    model.eval() # Freeze layers
    
    # 2. Load a test sample
    dataset = IVSDataset(
        os.path.join(BASE_DIR, "data/processed/heston_surfaces.npy"),
        os.path.join(BASE_DIR, "data/processed/heston_params.npy"),
        sparsity_range=(0.9, 0.9) # Test with 90% sparsity
    )
    sample = dataset[np.random.randint(len(dataset))]
    target = sample['target'].to(device).view(1, 10, 20)
    sparse_input = sample['input'].to(device).view(1, 10, 20)
    mask = sample['mask'].to(device).view(1, 10, 20)
    
    # 3. LSO: Optimize z
    # Initialize z randomly in the latent space
    z = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optimizer = optim.Adam([z], lr=0.01)
    
    print("Starting Latent Space Optimization...")
    for i in range(500):
        optimizer.zero_grad()
        # Decode z into a full surface
        gen_surface = model.decoder(z).view(1, 10, 20)
        
        # Loss is only calculated on OBSERVED points (mask == 1)
        recon_loss = torch.sum(((gen_surface - target) * mask)**2)
        
        recon_loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f"Step [{i+1}/500], Recon Loss: {recon_loss.item():.6f}")

    # 4. Final Reconstruction
    with torch.no_grad():
        completed_surface = model.decoder(z).view(10, 20).cpu().numpy()
    
    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Ground Truth', 'Sparse Input (90% Missing)', 'LSO Reconstructed']
    data_list = [target.squeeze().cpu().numpy(), sparse_input.squeeze().cpu().numpy(), completed_surface]
    
    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lso()