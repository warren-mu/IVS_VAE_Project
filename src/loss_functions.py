import torch
import torch.nn.functional as F

def vae_loss_function(recon_x, x, mu, logvar, mask, kl_weight=1.0, arb_weight=1.0):
    """
    Standard VAE loss (MSE + KLD) + Arbitrage Penalties
    """
    # 1. Reconstruction Loss (Only calculate on observed points if mask is provided)
    # For training, we usually reconstruct the full ground truth
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')

    # 2. KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # 3. Arbitrage Penalties
    # Total Variance w = sigma^2 * tau
    # We define maturities (tau) and strikes (k) grids matching the generation settings
    batch_size = x.size(0)
    tau = torch.linspace(0.1, 2.0, 10).to(x.device)
    k = torch.linspace(80, 120, 20).to(x.device)
    
    # Reshape recon_x back to (Batch, 10, 20) if it was flattened
    w = (recon_x ** 2) * tau.view(1, 10, 1) 

    # --- Calendar Spread Penalty (dw/dtau >= 0) ---
    dw_dtau = w[:, 1:, :] - w[:, :-1, :]
    calendar_penalty = torch.mean(F.relu(-dw_dtau)**2)

    # --- Butterfly Arbitrage Penalty (Numerical second derivative check) ---
    # Simplified check: convexity in strike direction
    dw_dk = w[:, :, 1:] - w[:, :, :-1]
    d2w_dk2 = dw_dk[:, :, 1:] - dw_dk[:, :, :-1]
    butterfly_penalty = torch.mean(F.relu(-d2w_dk2)**2)

    total_loss = mse_loss + kl_weight * kld_loss + arb_weight * (calendar_penalty + butterfly_penalty)
    
    return total_loss, mse_loss, kld_loss, calendar_penalty + butterfly_penalty