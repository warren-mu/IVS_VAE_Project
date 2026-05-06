#!/bin/bash

# IVS-VAE Project: Full Execution Pipeline
# Purpose: Automate data generation, training (ablation study), and evaluation.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--------------------------------------------------------"
echo "IVS-VAE Project: Starting Full Execution Pipeline"
echo "--------------------------------------------------------"

# 1. Install Dependencies
echo "[1/5] Installing project dependencies..."
pip install -r requirements.txt

# 2. Data Generation (Note: This takes time for 50,000 surfaces)
echo "[2/5] Generating synthetic Heston data (50,000 surfaces)..."
python scripts/generate_data.py

# 3. Model Training - Ablation Study
echo "[3/5] Starting Model Training (Ablation Study)..."

echo "Training Model A (Control Group: No Arbitrage Penalty)..."
python scripts/train_vae.py --mode no_arb

echo "Training Model B (Experimental Group: Full Physics-Informed VAE)..."
python scripts/train_vae.py --mode full

# 4. Quantitative Evaluation
echo "[4/5] Running Out-of-Sample Evaluation (90% Sparsity)..."
python scripts/evaluate_models.py

# 5. Sim-to-Real Application
echo "[5/5] Running Sim-to-Real Application (SPX Simulation)..."
python scripts/sim_to_real_spx.py

echo "--------------------------------------------------------"
echo "Success! All results and models are saved in the data/ directory."
echo "--------------------------------------------------------"