import torch
import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Add src to environment path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from vae_model import VAE

# ================= Experimental Hyperparameters =================
NUM_TEST_SAMPLES = 100  # Test set size
MISSING_RATE = 0.90     # Missing rate 90%
LSO_STEPS = 500         # Latent Space Optimization iteration steps
LSO_LR = 0.05           # LSO learning rate
LATENT_DIM = 10         
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Grid settings (Consistent with your data generation)
TAU_GRID = np.linspace(0.1, 2.0, 10)
K_GRID = np.linspace(80, 120, 20)
K_MESH, TAU_MESH = np.meshgrid(K_GRID, TAU_GRID)

def load_data():
    """Load test set data"""
    surface_path = os.path.join(BASE_DIR, "data/processed/heston_surfaces.npy")
    data = np.load(surface_path)
    # Take the last NUM_TEST_SAMPLES surfaces as the unseen test set
    test_data = data[-NUM_TEST_SAMPLES:]
    return torch.tensor(test_data, dtype=torch.float32)

def generate_mask():
    """Generate 90% missing mask (1 represents observed, 0 represents missing)"""
    mask = np.random.rand(10, 20) > MISSING_RATE
    # Ensure at least a few points exist, otherwise interpolation will throw an error
    if mask.sum() < 5:
        mask[0, 0] = mask[-1, -1] = mask[0, -1] = mask[-1, 0] = True
    return torch.tensor(mask, dtype=torch.float32)

def cubic_spline_imputation(surface, mask):
    """Baseline 1: Traditional SciPy Cubic Spline Interpolation"""
    surface_np = surface.numpy()
    mask_np = mask.numpy()
    
    # Extract observed points
    points = np.array([K_MESH[mask_np == 1], TAU_MESH[mask_np == 1]]).T
    values = surface_np[mask_np == 1]
    
    # Interpolate to the full grid
    grid_points = np.array([K_MESH.ravel(), TAU_MESH.ravel()]).T
    imputed_flat = griddata(points, values, grid_points, method='cubic')
    
    # Since 90% missingness will inevitably cause cubic convex hull interpolation to fail at edges, producing NaNs
    # We use 'nearest' as a fallback to fix edge NaNs
    if np.isnan(imputed_flat).any():
        imputed_nearest = griddata(points, values, grid_points, method='nearest')
        imputed_flat[np.isnan(imputed_flat)] = imputed_nearest[np.isnan(imputed_flat)]
        
    return torch.tensor(imputed_flat.reshape(10, 20), dtype=torch.float32)

def latent_space_optimization(model, target_surface, mask):
    """Baselines 2 & 3: VAE-based Latent Space Optimization (LSO)"""
    model.eval()
    target_surface = target_surface.to(DEVICE)
    mask = mask.to(DEVICE)
    
    # Initialize latent variable z (enable gradients)
    z = nn.Parameter(torch.randn(1, LATENT_DIM, device=DEVICE))
    optimizer = torch.optim.Adam([z], lr=LSO_LR)
    
    for step in range(LSO_STEPS):
        optimizer.zero_grad()
        recon_surface = model.decoder(z).view(10, 20)
        
        # Calculate MSE only on observed points (mask == 1)
        loss = torch.sum(((recon_surface - target_surface) * mask)**2) / torch.sum(mask)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_recon = model.decoder(z).view(10, 20)
    return final_recon.cpu()

def calculate_arbitrage_violations(surface):
    """Calculate arbitrage violation rates (strictly following the finite difference logic in the report)"""
    w = (surface ** 2) * torch.tensor(TAU_GRID).view(10, 1)
    
    # 1. Calendar Spread (dw/dtau >= 0)
    dw_dtau = w[1:, :] - w[:-1, :]
    cal_violations = torch.sum(dw_dtau < 0).item()
    cal_total = dw_dtau.numel()
    
    # 2. Butterfly Arbitrage (Convexity: d2w/dk2 >= 0)
    dw_dk = w[:, 1:] - w[:, :-1]
    d2w_dk2 = dw_dk[:, 1:] - dw_dk[:, :-1]
    but_violations = torch.sum(d2w_dk2 < 0).item()
    but_total = d2w_dk2.numel()
    
    return cal_violations, cal_total, but_violations, but_total

def main():
    print(f"🚀 Starting ultimate evaluation: {NUM_TEST_SAMPLES} test surfaces, missing rate {MISSING_RATE*100}%")
    
    test_data = load_data()
    
    # Load the two models
    model_no_arb = VAE(input_dim=200, latent_dim=LATENT_DIM).to(DEVICE)
    model_no_arb.load_state_dict(torch.load(os.path.join(BASE_DIR, "data/vae_model_no_arb.pth"), map_location=DEVICE))
    
    model_full = VAE(input_dim=200, latent_dim=LATENT_DIM).to(DEVICE)
    model_full.load_state_dict(torch.load(os.path.join(BASE_DIR, "data/vae_model_full.pth"), map_location=DEVICE))
    
    # Statistical metrics logger
    results = {
        "Spline": {"rmse": 0, "cal_v": 0, "but_v": 0},
        "VAE_No_Arb": {"rmse": 0, "cal_v": 0, "but_v": 0},
        "VAE_Full": {"rmse": 0, "cal_v": 0, "but_v": 0}
    }
    
    # We save the first surface for plotting (optional, logic kept intact)
    plot_data = {}

    for i in range(NUM_TEST_SAMPLES):
        target = test_data[i]
        mask = generate_mask()
        
        # --- 1. Spline Imputation ---
        recon_spline = cubic_spline_imputation(target, mask)
        
        # --- 2. VAE (No Constraints) Imputation ---
        recon_no_arb = latent_space_optimization(model_no_arb, target, mask)
        
        # --- 3. VAE (Physics-Informed) Imputation ---
        recon_full = latent_space_optimization(model_full, target, mask)
        
        # Record plotting data
        if i == 0:
            plot_data = {'target': target, 'mask': mask, 'spline': recon_spline, 'no_arb': recon_no_arb, 'full': recon_full}
        
        # Calculate RMSE for the missing regions (mask == 0)
        inv_mask = 1.0 - mask
        for name, recon in zip(["Spline", "VAE_No_Arb", "VAE_Full"], [recon_spline, recon_no_arb, recon_full]):
            rmse = torch.sqrt(torch.sum(((recon - target) * inv_mask)**2) / torch.sum(inv_mask)).item()
            c_v, c_t, b_v, b_t = calculate_arbitrage_violations(recon)
            
            results[name]["rmse"] += rmse
            results[name]["cal_v"] += c_v
            results[name]["but_v"] += b_v
            
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{NUM_TEST_SAMPLES}] Evaluating...")

    # Calculate averages
    total_cal_checks = NUM_TEST_SAMPLES * c_t
    total_but_checks = NUM_TEST_SAMPLES * b_t
    
    print("\n" + "="*50)
    print("📊 Final Empirical Quantitative Report (under 90% missing data)")
    print("="*50)
    print(f"{'Method':<15} | {'Missing RMSE':<12} | {'Calendar Violations':<20} | {'Butterfly Violations':<20}")
    print("-" * 75)
    for name in results.keys():
        avg_rmse = results[name]["rmse"] / NUM_TEST_SAMPLES
        cal_rate = (results[name]["cal_v"] / total_cal_checks) * 100
        but_rate = (results[name]["but_v"] / total_but_checks) * 100
        print(f"{name:<15} | {avg_rmse:<12.5f} | {cal_rate:>6.2f}% ({results[name]['cal_v']}/{total_cal_checks}) | {but_rate:>6.2f}% ({results[name]['but_v']}/{total_but_checks})")
    print("="*50)

if __name__ == "__main__":
    main()