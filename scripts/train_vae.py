import torch
import os
import sys
import numpy as np
from torch.optim import Adam

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from dataset import get_dataloader
from vae_model import VAE
from loss_functions import vae_loss_function

# ==============================================================================
# 🎛️ Ablation Study Control Panel for Final Report
# ==============================================================================
# Task 1: Set to False and run. Trains "vae_model_no_arb.pth" (Control Group)
# Task 2: Set to True and run. Trains "vae_model_full.pth" (Experimental Group)
USE_ARBITRAGE_PENALTY = True  

if USE_ARBITRAGE_PENALTY:
    LAMBDA_ARB = 1.0  # Enable arbitrage penalty weight
    SAVE_NAME = "vae_model_full.pth"
    print(">>> 🚀 Training: [Ultimate Model - Physics-Informed VAE with HJM Arbitrage-Free Constraints] <<<")
else:
    LAMBDA_ARB = 0.0  # Set weight to 0 to completely disable physics constraints
    SAVE_NAME = "vae_model_no_arb.pth"
    print(">>> 🧪 Training: [Control Model - Purely Data-Driven VAE without Arbitrage Constraints] <<<")
# ==============================================================================

def train():
    # 1. Hardware Setup (MPS for Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Hyperparameters
    epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    latent_dim = 10
    
    # Path to the 50,000 high-quality generated surfaces
    surface_path = os.path.join(BASE_DIR, "data/processed/heston_surfaces.npy")
    param_path = os.path.join(BASE_DIR, "data/processed/heston_params.npy")

    # 3. Data & Model Initialization
    dataloader = get_dataloader(surface_path, param_path, batch_size=batch_size)
    model = VAE(input_dim=200, latent_dim=latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        for batch in dataloader:
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(target)
            
            # --- Core Logic: Pass arb_weight matching the interface in loss_functions.py ---
            loss, mse, kld, arb = vae_loss_function(
                recon_batch, 
                target, 
                mu, 
                logvar, 
                batch['mask'].to(device),
                arb_weight=LAMBDA_ARB 
            )
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_epoch_loss/len(dataloader):.6f}")

    # 5. Save Model Weights
    model_path = os.path.join(BASE_DIR, f"data/{SAVE_NAME}")
    torch.save(model.state_dict(), model_path)
    print(f"🎉 Training Complete! Model saved to {model_path}")

if __name__ == "__main__":
    train()