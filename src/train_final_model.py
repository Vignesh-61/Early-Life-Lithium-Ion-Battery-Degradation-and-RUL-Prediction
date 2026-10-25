"""
train_final_model.py — Final Model Training and Serialization
=============================================================
Trains the validated final Adaptive Memory-Window DNN (W=10, 120 inputs)
and serializes:
  - models/final/final_model.pt (PyTorch state dict)
  - models/final/scaler.joblib (AntiLeakageScaler)
  - models/final/config.json (Model hyperparameters, feature names, EOL parameters)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dnn_model import build_model
from preprocessing import compute_soh, compute_rul, AntiLeakageScaler, DEFAULT_EOL_THRESHOLD_AH
from memory_windows import create_memory_window_dataset
from phase2_experiments import train_dnn, set_seed

def train_and_save_final_model():
    set_seed(42)
    
    # 1. Load data
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cycle_features.csv')
    df = pd.read_csv(data_path)
    df = compute_soh(df)
    df = compute_rul(df)
    
    # Selected configuration: W = 10
    selected_window = 10
    ds = create_memory_window_dataset(df, window_size=selected_window)
    X = ds['X']
    y = ds['y']
    meta = ds['metadata']
    feature_names = ds['feature_names']
    
    print(f"[final_model] Training final DNN on {len(X)} memory window samples (Window size W={selected_window}, input_dim={X.shape[1]})...")
    
    # Scaler
    scaler = AntiLeakageScaler(scaler_type='minmax')
    X_scaled = scaler.fit_transform(X)
    
    # Split train/val for final model (e.g. 85/15 split)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.85 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_tr, y_tr = X_scaled[train_idx], y[train_idx]
    X_val, y_val = X_scaled[val_idx], y[val_idx]
    
    model, tr_losses, val_losses = train_dnn(X_tr, y_tr, X_val, y_val, epochs=150, lr=0.001, patience=25)
    
    # Save directory
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    final_dir = os.path.join(models_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    
    model_path = os.path.join(final_dir, 'final_model.pt')
    scaler_path = os.path.join(final_dir, 'scaler.joblib')
    config_path = os.path.join(final_dir, 'config.json')
    
    # Also save in models root for easy discovery
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), os.path.join(models_dir, 'final_model.pt'))
    scaler.save(scaler_path)
    scaler.save(os.path.join(models_dir, 'scaler.joblib'))
    
    config = {
        'model_name': 'Adaptive_Memory_Window_BatteryDNN',
        'project_title': 'Early Lithium-Ion Battery Degradation and Remaining Useful Life Prediction Using an Adaptive Memory-Window Deep Neural Network',
        'dataset': 'NASA Lithium-Ion Battery Aging Dataset (B0005, B0006, B0007, B0018)',
        'selected_memory_window': selected_window,
        'input_dimension': int(X.shape[1]),
        'num_base_features_per_cycle': 12,
        'base_feature_names': [
            'voltage_mean', 'voltage_min', 'voltage_max', 'voltage_std',
            'current_mean', 'current_std',
            'temp_mean', 'temp_max', 'temp_range',
            'discharge_duration', 'discharge_capacity', 'cycle_number'
        ],
        'flattened_window_feature_names': feature_names,
        'target_variable': 'discharge_capacity_Ah',
        'soh_formula': 'Current Capacity / Initial Capacity * 100',
        'rated_capacity_Ah': 2.0,
        'eol_threshold_Ah': DEFAULT_EOL_THRESHOLD_AH,
        'eol_soh_threshold_percent': 70.0,
        'model_architecture': {
            'type': 'FeedForward_MLP',
            'hidden_layers': [128, 64, 32],
            'activation': 'ReLU',
            'dropout_rate': 0.2,
            'weight_initialization': 'He / Kaiming Normal',
            'optimizer': 'Adam',
            'loss_function': 'MSELoss'
        },
        'normalization': {
            'scaler_type': 'MinMaxScaler',
            'range': [0, 1]
        },
        'validation_metrics_lobo': {
            'mean_MAE_Ah': 0.08037,
            'mean_RMSE_Ah': 0.09960,
            'mean_R2': 0.6512,
            'mean_MAPE_percent': 5.21,
            'mean_RUL_MAE_cycles': 8.45
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(models_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"[final_model] Saved final model checkpoint to {model_path}")
    print(f"[final_model] Saved fitted scaler to {scaler_path}")
    print(f"[final_model] Saved metadata config to {config_path}")

if __name__ == '__main__':
    train_and_save_final_model()
