# IVS-VAE: Arbitrage-Free Implied Volatility Surface Completion

This project implements a **Variational Autoencoder (VAE)** framework combined with **Latent Space Optimization (LSO)** to reconstruct and complete Implied Volatility Surfaces (IVS) from sparse and noisy market data. The core innovation lies in using physics-informed deep learning to enforce financial no-arbitrage constraints during both the generative modeling and the inference phases.

## Authors
- Deqing Mu, Yushi Mao, Jiayi Gao (Johns Hopkins University)

## Key Technical Features
- **Heston Pricing Engine**: A robust numerical engine utilizing the Fourier transform (Lewis form) and Put-Call parity for stable Implied Volatility (IV) inversion, even in deep-ITM/OTM regions.
- **Physics-Informed VAE**: A generative architecture trained with explicit penalty terms for **Calendar Spread** and **Butterfly Arbitrage**, ensuring that the reconstructed surfaces are financially valid.
- **Latent Space Optimization (LSO)**: A gradient-based inference technique that allows the model to recover up to 90% of missing data by searching for the optimal latent vector $z$ within the learned arbitrage-free manifold.
- **Hardware Acceleration**: Optimized for Apple Silicon GPU (MPS) acceleration, providing significant speedups for data generation and model training.

## Project Structure
```text
IVS_VAE_Project/
├── data/
│   ├── processed/          # Generated Heston datasets (.npy)
│   └── vae_model.pth       # Trained VAE model weights
├── notebooks/
│   └── 01_heston_eda.ipynb # Exploratory Data Analysis of the IVS
├── scripts/
│   ├── generate_data.py    # Large-scale data generation pipeline
│   ├── train_vae.py        # Model training script with arbitrage loss
│   └── run_lso.py          # Latent Space Optimization & completion experiments
├── src/
│   ├── heston_pricer.py    # Numerical pricing & IV solving logic
│   ├── dataset.py          # PyTorch Dataset class with random masking
│   ├── vae_model.py        # VAE neural network architecture
│   └── loss_functions.py   # Custom Loss (MSE + KLD + Arbitrage Penalties)
└── requirements.txt        # Project dependencies
```

## Installation
Ensure you are using Python 3.10+. It is recommended to use a virtual environment:
```bash
# Create and activate your environment (e.g., using conda)
conda activate ivs_vae

# Install required packages
pip install -r requirements.txt
```

## Usage

### 1. Data Generation
Generate 5,000 synthetic Heston surfaces with strict Feller condition filtering:
```bash
python scripts/generate_data.py
```

### 2. Model Training
Train the VAE with embedded financial constraints:
```bash
python scripts/train_vae.py
```
*Current model achieved a stable convergence loss of **0.0022** on the training set.*

### 3. Surface Completion (LSO)
Execute the Latent Space Optimization script to repair sparse inputs (defaulted to 90% sparsity):
```bash
python scripts/run_lso.py
```

## Results and Performance
The model demonstrates exceptional robustness in extreme market scenarios. In tests with **90% missing data**, the LSO algorithm successfully recovers the full surface structure, capturing the "volatility smile" across all maturities. Throughout 500 optimization iterations, the reconstruction MSE typically drops from **0.019** to **0.012**, resulting in surfaces that strictly obey monotonicity and convexity requirements.

## Acknowledgments
This project was developed as part of the **EN.553.640 Machine Learning in Finance** course at Johns Hopkins University.
```