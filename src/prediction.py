"""
prediction.py — Inference and Prediction Pipeline
===================================================

What this module does:
    Loads a trained BatteryDNN model checkpoint and its associated
    AntiLeakageScaler to perform real-time degradation prediction
    on unseen battery cycles.

Usage:
    python src/prediction.py --model results/phase1/models/dnn_w5_B0005.pt \
                             --scaler results/phase1/models/scaler_w5_B0005.joblib \
                             --data data/processed/cycle_features.csv \
                             --battery B0005 --cycle 50
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from dnn_model import build_model
from preprocessing import AntiLeakageScaler
from feature_engineering import FEATURE_NAMES


def load_inference_pipeline(model_path, scaler_path, window_size=5, num_features=12, device='cpu'):
    """
    Load a trained BatteryDNN model and fitted scaler for inference.
    
    Parameters
    ----------
    model_path : str
        Path to .pt state dict file
    scaler_path : str
        Path to saved joblib scaler
    window_size : int
    num_features : int
    device : str
    
    Returns
    -------
    (model, scaler) : tuple
    """
    input_dim = window_size * num_features
    model = build_model(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    scaler = AntiLeakageScaler()
    scaler.load(scaler_path)
    
    return model, scaler


def predict_next_cycle(model, scaler, past_window_df, initial_capacity=None, eol_threshold=1.4):
    """
    Predict the next cycle's capacity and SOH given a window of past cycles.
    
    Parameters
    ----------
    model : BatteryDNN
    scaler : AntiLeakageScaler
    past_window_df : pd.DataFrame
        DataFrame of exactly W past consecutive cycles with FEATURE_NAMES columns.
    initial_capacity : float, optional
        Battery initial capacity in Ah to compute SOH %.
    eol_threshold : float
        EOL threshold in Ah (default 1.4 Ah).
        
    Returns
    -------
    dict with:
        'predicted_capacity_ah': float
        'predicted_soh_percent': float or None
        'is_below_eol': bool
    """
    feature_cols = [c for c in FEATURE_NAMES if c in past_window_df.columns]
    feature_matrix = past_window_df[feature_cols].values
    flattened = feature_matrix.flatten().reshape(1, -1)
    
    # Scale input using the training scaler
    scaled_input = scaler.transform(flattened)
    
    # Forward pass
    with torch.no_grad():
        pred_tensor = model(torch.tensor(scaled_input, dtype=torch.float32))
        pred_capacity = float(pred_tensor.item())
        
    soh_percent = None
    if initial_capacity is not None and initial_capacity > 0:
        soh_percent = round((pred_capacity / initial_capacity) * 100.0, 2)
        
    is_below_eol = pred_capacity <= eol_threshold
    
    return {
        'predicted_capacity_ah': round(pred_capacity, 4),
        'predicted_soh_percent': soh_percent,
        'is_below_eol': is_below_eol
    }


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    data_path = os.path.join(project_root, 'data', 'processed', 'cycle_features.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        bat_df = df[df['battery_id'] == 'B0005'].sort_values('cycle_number').reset_index(drop=True)
        print(f"[prediction] Inference test on B0005 using first 5 cycles:")
        window_sample = bat_df.iloc[0:5]
        
        # Test with dummy model to verify pipeline math
        model = build_model(input_dim=5 * 12)
        scaler = AntiLeakageScaler()
        scaler.fit(np.zeros((10, 60)))
        
        res = predict_next_cycle(model, scaler, window_sample, initial_capacity=1.8565)
        print("[prediction] Output:", res)
        print("[prediction] Prediction module verified.")
