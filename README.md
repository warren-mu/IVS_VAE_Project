# IVS-VAE: Physics-Informed Deep Generative Models for Arbitrage-Free Implied Volatility Surface Completion

This repository contains the official implementation of our research on Implied Volatility Surface (IVS) completion using **Physics-Informed Variational Autoencoders (VAE)** and **Latent Space Optimization (LSO)**. 

The core research objective is to bridge quantitative finance theory with deep generative modeling, enforcing structural financial constraints within a purely data-driven neural network architecture to reconstruct highly sparse and noisy market data.

## Authors
- Deqing Mu, Yushi Mao, Jiayi Gao (Johns Hopkins University)

## Core Research Methodologies & Skills
This project highlights an interdisciplinary research approach, combining rigorous financial mathematics with advanced machine learning engineering:
- **Physics-Informed Neural Networks:** Designed custom loss functions integrating discrete soft penalties for first-order (Calendar Spread) and second-order (Butterfly Spread) arbitrage violations.
- **Latent Manifold Calibration (LSO):** Formulated an inference pipeline that freezes decoder weights and uses gradient descent (Adam) to optimize the latent representation $z$ against sparse observations, framing surface completion as an inverse problem.
- **Large-Scale Data Engineering:** Programmed a robust Heston pricing engine utilizing Fourier transforms (Lewis form) to synthesize a stable, arbitrage-free dataset of 50,000 volatility surfaces, strictly filtered by the Feller condition ($2\kappa\theta > \xi^2$).
- **Empirical Benchmarking:** Conducted rigorous ablation studies comparing pure data-driven approaches against physics-informed models and classical financial engineering baselines (Cubic Splines) under extreme data sparsity.

## Project Structure
```text
IVS_VAE_Project/
├── data/
│   ├── processed/               # Generated Heston datasets (50,000 surfaces)
│   ├── vae_model_no_arb.pth     # Control group weights (Pure VAE)
│   └── vae_model_full.pth       # Experimental group weights (Physics-Informed VAE)
├── notebooks/
│   └── 01_heston_eda.ipynb      # Exploratory Data Analysis & Manifold Visualization
├── scripts/
│   ├── generate_data.py         # Large-scale Heston data generation pipeline
│   ├── train_vae.py             # Model training script with Ablation Control Panel
│   ├── evaluate_models.py       # Out-of-sample quantitative evaluation & metric logging
│   └── sim_to_real_spx.py       # Sim-to-Real application with controlled market noise
├── src/
│   ├── heston_pricer.py         # Numerical pricing & IV solving logic
│   ├── dataset.py               # PyTorch Dataset class with random masking
│   ├── vae_model.py             # VAE neural network architecture
│   └── loss_functions.py        # Custom Loss (MSE + KLD + Arbitrage Penalties)
└── requirements.txt             # Project dependencies
```

## Empirical Results & Ablation Study

To validate the generative capacity and financial consistency of our model, we conducted a rigorous out-of-sample evaluation under extreme liquidity drought scenarios (**90% data missingness**, leaving only ~20 discrete quotes per surface).

We benchmarked our Physics-Informed VAE against both a pure data-driven VAE (to isolate the effect of the arbitrage penalty) and classical Cubic Spline Interpolation. 

### Quantitative Performance (Out-of-Sample)
| Method | Missing RMSE | Calendar Violations ($\Delta_\tau w < 0$) | Butterfly Violations ($\Delta_m^2 w < 0$) |
| :--- | :---: | :---: | :---: |
| Cubic Spline | 0.14826 | 8.48% | 37.46% |
| VAE (No Arb) | 0.02609 | 1.83% | 35.82% |
| **VAE (Full, Ours)** | **0.02228** | **1.19%** | 36.72% |

### Critical Reflections
1. **Superiority of Latent Prior:** Under 90% sparsity, traditional Spline interpolation collapses (RMSE 0.148) due to the compromised convex hull. The VAE successfully projects sparse points onto a learned Heston manifold, reducing error by over 85%.
2. **First-Order Constraints:** The soft penalty effectively suppressed Calendar Arbitrage, dropping the violation rate from 8.48% to a near-negligible 1.19%.
3. **Limitations on High-Order Convexity:** The persistence of Butterfly violations (around 36\%) across models exposes a fundamental limitation of soft penalties in deep learning. In coarse discrete grids, the gradients of the reconstruction MSE heavily dominate the delicate second-order convexity gradients ($\Delta_m^2 w$). This opens future research directions toward integrating Neural SDEs or hard PDE constraints.

## Installation & Usage
```bash
# 1. Environment Setup
conda create -n ivs_vae python=3.10
conda activate ivs_vae
pip install -r requirements.txt

# 2. Data Generation (Produces 50,000 surfaces)
python scripts/generate_data.py

# 3. Model Training (Ablation Study)
# Toggle `USE_ARBITRAGE_PENALTY` in the script to train both models
python scripts/train_vae.py

# 4. Quantitative Evaluation
python scripts/evaluate_models.py

# 5. Real Market Application (Sim-to-Real)
# Demonstrates model's denoising and extrapolation capabilities on simulated noisy market data
python scripts/sim_to_real_spx.py
```

## Acknowledgments
This research was developed as part of the **EN.553.640 Machine Learning in Finance** course at Johns Hopkins University.
```
