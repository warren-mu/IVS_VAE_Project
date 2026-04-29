import numpy as np
import os
import sys
from tqdm import tqdm
import warnings
import scipy.integrate as integrate

# Suppress harmless integration warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

# Dynamically determine the project root directory
# This ensures data is saved correctly regardless of where the script is executed from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_DIR = os.path.join(BASE_DIR, "data", "processed")

# Add 'src' directory to sys.path to allow importing the Heston pricer module
sys.path.append(os.path.join(BASE_DIR, 'src'))
from heston_pricer import HestonPricer, implied_volatility

def generate_dataset(num_samples=5000, save_dir=DEFAULT_SAVE_DIR):
    """
    Generate a large-scale dataset of Heston Implied Volatility Surfaces (IVS).
    """
    # Create the target directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Market environment and grid settings
    S0 = 100.0  # Initial stock price
    r = 0.02    # Risk-free interest rate (2%)
    
    # Define the 20x10 grid: 20 strikes and 10 maturities
    strikes = np.linspace(80, 120, 20)
    maturities = np.linspace(0.1, 2.0, 10)
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    
    valid_surfaces = []
    valid_params = []
    reject_count = 0  # Track the number of discarded parameter sets
    
    print(f"Starting generation of {num_samples} volatility surfaces...")
    pbar = tqdm(total=num_samples)
    
    while len(valid_surfaces) < num_samples:
        # Sample Heston parameters from realistic economic ranges
        kappa = np.random.uniform(1.0, 4.0)   # Mean-reversion speed
        theta = np.random.uniform(0.02, 0.15) # Long-term variance
        sigma = np.random.uniform(0.1, 0.6)  # Volatility of volatility
        rho = np.random.uniform(-0.9, -0.3)  # Asset-volatility correlation (negative skew)
        v0 = np.random.uniform(0.02, 0.15)    # Initial variance
        
        # Feller Condition check: 2 * kappa * theta > sigma^2
        # This prevents the variance process from hitting zero too often,
        # significantly improving the numerical stability of the Fourier integral.
        if 2 * kappa * theta <= sigma**2:
            reject_count += 1
            pbar.set_postfix({"Rejects": reject_count})
            continue
            
        pricer = HestonPricer(kappa, theta, sigma, rho, v0, r)
        iv_surface = np.zeros_like(K_grid)
        is_valid = True
        
        # Iterate through the grid to calculate Implied Volatility for every point
        for i in range(len(maturities)):
            for j in range(len(strikes)):
                # Step 1: Calculate European Call Option Price using Heston Model
                price = pricer.price_european_call(S0, K_grid[i,j], T_grid[i,j])
                
                # Step 2: Solve for Implied Volatility via Black-Scholes inversion
                iv = implied_volatility(price, S0, K_grid[i,j], T_grid[i,j], r)
                
                # If numerical solver fails (returns NaN or negative), discard the surface
                if np.isnan(iv) or iv <= 0:
                    is_valid = False
                    break
                iv_surface[i,j] = iv  # Assign the successful IV to the surface grid
                
            if not is_valid:
                break
                
        # Only store surfaces where the entire grid was solved successfully
        if is_valid:
            valid_surfaces.append(iv_surface)
            valid_params.append([kappa, theta, sigma, rho, v0])
            pbar.update(1)
        else:
            reject_count += 1
            
        pbar.set_postfix({"Rejects": reject_count})
            
    pbar.close()
    
    # Convert lists to NumPy arrays with float32 precision
    # This aligns with PyTorch's default tensor type and saves storage space
    surfaces_array = np.array(valid_surfaces, dtype=np.float32)
    params_array = np.array(valid_params, dtype=np.float32)
    
    # Save datasets locally
    np.save(os.path.join(save_dir, "heston_surfaces.npy"), surfaces_array)
    np.save(os.path.join(save_dir, "heston_params.npy"), params_array)
    
    print(f"Successfully saved {surfaces_array.shape[0]} surfaces to {save_dir}")
    print(f"Surfaces Dataset Shape: {surfaces_array.shape} -> (Samples, Maturities, Strikes)")
    print(f"Params Dataset Shape: {params_array.shape}")

if __name__ == "__main__":
    # Standard training size: 5000 samples. 
    # This might take 15-30 minutes depending on your CPU.
    generate_dataset(num_samples=5000)