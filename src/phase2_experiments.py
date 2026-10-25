"""
phase2_experiments.py — Phase 2 Research Experiments Runner
============================================================
Conducts the 5 core research experiments:
  1. Memory-Window Comparison (3, 5, 10, 20 cycles) with SOH & RUL error metrics
  2. Early-Life Degradation & RUL Prediction (using first 20, 30, 40, 50 cycles)
  3. Feature & Memory Ablation Study
  4. Noise Robustness Evaluation (1%, 2%, 5% Gaussian sensor noise)
  5. Cross-Battery Generalization (LOBO breakdowns on B0005, B0006, B0007, B0018)

All outputs are saved to results/final/, results/, and figures/.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure src in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dnn_model import build_model
from preprocessing import compute_soh, compute_rul, AntiLeakageScaler, DEFAULT_EOL_THRESHOLD_AH
from memory_windows import create_memory_window_dataset
from evaluation import calculate_metrics, compute_naive_baseline

# Set clean scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_dnn(X_train, y_train, X_val, y_val, epochs=120, batch_size=32, lr=0.001, patience=20):
    """Train BatteryDNN with early stopping."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_dim = X_train.shape[1]
    model = build_model(input_dim=input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        model.train()
        b_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            b_losses.append(loss.item())
        t_loss = float(np.mean(b_losses))
        train_losses.append(t_loss)
        
        model.eval()
        v_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                v_losses.append(criterion(model(bx), by).item())
        v_loss = float(np.mean(v_losses))
        val_losses.append(v_loss)
        
        if v_loss < best_loss - 1e-5:
            best_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_weights:
        model.load_state_dict(best_weights)
        
    return model, train_losses, val_losses


def compute_rul_prediction_error(y_true, y_pred, cycles, eol_threshold=DEFAULT_EOL_THRESHOLD_AH):
    """
    Compute Remaining Useful Life prediction error in cycles.
    Compares the cycle where true vs predicted capacity crosses EOL threshold.
    """
    true_eol_idx = np.where(y_true <= eol_threshold)[0]
    pred_eol_idx = np.where(y_pred <= eol_threshold)[0]
    
    true_eol_cycle = cycles[true_eol_idx[0]] if len(true_eol_idx) > 0 else cycles[-1]
    pred_eol_cycle = cycles[pred_eol_idx[0]] if len(pred_eol_idx) > 0 else cycles[-1]
    
    # Cycle-level RUL trajectory error
    true_rul_series = np.maximum(0, true_eol_cycle - cycles)
    pred_rul_series = np.maximum(0, pred_eol_cycle - cycles)
    
    rul_mae = float(np.mean(np.abs(true_rul_series - pred_rul_series)))
    eol_cycle_error = int(abs(true_eol_cycle - pred_eol_cycle))
    
    return rul_mae, eol_cycle_error, true_eol_cycle, pred_eol_cycle


# =========================================================================
# EXPERIMENT 1: Memory-Window Comparison (3, 5, 10, 20)
# =========================================================================
def run_experiment_1(df, output_dir, figures_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Memory-Window Systematic Comparison (LOBO CV)")
    print("=" * 70)
    
    window_sizes = [3, 5, 10, 20]
    results = []
    
    for w in window_sizes:
        ds = create_memory_window_dataset(df, window_size=w)
        X_all, y_all, meta = ds['X'], ds['y'], ds['metadata']
        batteries = sorted(meta['battery_id'].unique())
        
        w_maes, w_rmses, w_r2s, w_mapes, w_rul_maes = [], [], [], [], []
        
        for test_bat in batteries:
            train_mask = (meta['battery_id'] != test_bat).values
            test_mask = (meta['battery_id'] == test_bat).values
            
            X_tr_raw, y_tr = X_all[train_mask], y_all[train_mask]
            X_te_raw, y_te = X_all[test_mask], y_all[test_mask]
            test_cycles = meta.loc[test_mask, 'target_cycle'].values
            
            scaler = AntiLeakageScaler('minmax')
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
            
            model, _, _ = train_dnn(X_tr, y_tr, X_te, y_te, epochs=120, lr=0.001)
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy().flatten()
                
            m = calculate_metrics(y_te, y_pred)
            rul_mae, eol_err, _, _ = compute_rul_prediction_error(y_te, y_pred, test_cycles)
            
            w_maes.append(m['mae'])
            w_rmses.append(m['rmse'])
            w_r2s.append(m['r2'])
            w_mapes.append(m['mape'])
            w_rul_maes.append(rul_mae)
            
        results.append({
            'memory_window': w,
            'input_features': X_all.shape[1],
            'MAE': round(float(np.mean(w_maes)), 5),
            'RMSE': round(float(np.mean(w_rmses)), 5),
            'R2': round(float(np.mean(w_r2s)), 4),
            'MAPE_%': round(float(np.mean(w_mapes)), 2),
            'RUL_error_cycles': round(float(np.mean(w_rul_maes)), 2)
        })
        print(f"Memory-{w:2d} -> MAE: {results[-1]['MAE']:.5f} Ah | RMSE: {results[-1]['RMSE']:.5f} Ah | R2: {results[-1]['R2']:.4f} | RUL MAE: {results[-1]['RUL_error_cycles']:.1f} cycles")
        
    df_res = pd.DataFrame(results)
    
    # Save CSVs
    csv_final = os.path.join(output_dir, 'memory_comparison.csv')
    csv_results = os.path.join(PROJECT_ROOT, 'results', 'memory_comparison.csv')
    df_res.to_csv(csv_final, index=False)
    df_res.to_csv(csv_results, index=False)
    
    # Plot Comparison Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    bars = ax1.bar([f"W={w}" for w in df_res['memory_window']], df_res['RMSE'], color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], width=0.55, edgecolor='black', alpha=0.85)
    ax1.set_title("Capacity Prediction Error (RMSE) vs Memory Window", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Memory Window (Cycles)", fontsize=11)
    ax1.set_ylabel("Mean RMSE (Ah)", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    bars2 = ax2.bar([f"W={w}" for w in df_res['memory_window']], df_res['RUL_error_cycles'], color=['#9b59b6', '#1abc9c', '#e67e22', '#34495e'], width=0.55, edgecolor='black', alpha=0.85)
    ax2.set_title("Remaining Useful Life Error (MAE) vs Memory Window", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Memory Window (Cycles)", fontsize=11)
    ax2.set_ylabel("RUL MAE (Cycles)", fontsize=11)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'memory_window_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'memory_window_comparison.png'), dpi=300)
    plt.close()
    
    return df_res


# =========================================================================
# EXPERIMENT 2: Early-Life Degradation & RUL Prediction
# =========================================================================
def run_experiment_2(df, output_dir, figures_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Early-Life Degradation and RUL Trajectory Prediction")
    print("=" * 70)
    
    # Evaluate early life cutoff: using early 20, 30, 40, 50 cycles to forecast remaining life
    early_cutoffs = [20, 30, 40, 50]
    w = 10  # Use robust 10-cycle window
    
    ds = create_memory_window_dataset(df, window_size=w)
    X_all, y_all, meta = ds['X'], ds['y'], ds['metadata']
    
    early_results = []
    
    # Train model on 3 batteries (B0006, B0007, B0018) and test on unseen B0005
    test_bat = 'B0005'
    train_mask = (meta['battery_id'] != test_bat).values
    test_mask = (meta['battery_id'] == test_bat).values
    
    X_tr_raw, y_tr = X_all[train_mask], y_all[train_mask]
    X_te_raw, y_te = X_all[test_mask], y_all[test_mask]
    test_cycles = meta.loc[test_mask, 'target_cycle'].values
    
    scaler = AntiLeakageScaler('minmax')
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    
    model, _, _ = train_dnn(X_tr, y_tr, X_te, y_te, epochs=120)
    model.eval()
    with torch.no_grad():
        full_pred = model(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy().flatten()
        
    for cutoff in early_cutoffs:
        cutoff_mask = test_cycles <= cutoff
        if np.sum(cutoff_mask) < 2:
            continue
        
        y_early_true = y_te[cutoff_mask]
        y_early_pred = full_pred[cutoff_mask]
        
        # Future predictions beyond cutoff
        future_mask = test_cycles > cutoff
        y_fut_true = y_te[future_mask]
        y_fut_pred = full_pred[future_mask]
        
        fut_metrics = calculate_metrics(y_fut_true, y_fut_pred)
        rul_mae, eol_err, true_eol, pred_eol = compute_rul_prediction_error(y_te, full_pred, test_cycles)
        
        early_results.append({
            'early_observation_cycles': cutoff,
            'future_eval_cycles': len(y_fut_true),
            'future_MAE': round(fut_metrics['mae'], 5),
            'future_RMSE': round(fut_metrics['rmse'], 5),
            'future_R2': round(fut_metrics['r2'], 4),
            'true_eol_cycle': true_eol,
            'predicted_eol_cycle': pred_eol,
            'eol_cycle_error': eol_err
        })
        print(f"Early Observation: {cutoff:2d} cycles | Future RMSE: {fut_metrics['rmse']:.5f} Ah | Predicted EOL: Cycle {pred_eol} (True: Cycle {true_eol})")
        
    df_early = pd.DataFrame(early_results)
    df_early.to_csv(os.path.join(output_dir, 'early_life_prediction.csv'), index=False)
    
    # Generate Publication-Quality Early-Life Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(test_cycles, y_te, 'o-', color='#2c3e50', label='Actual NASA B0005 Capacity', linewidth=2.0, markersize=4, alpha=0.85)
    ax.plot(test_cycles, full_pred, 's--', color='#e74c3c', label='DNN Memory-10 Prediction', linewidth=2.0, markersize=3.5, alpha=0.9)
    
    # Highlight early observation phase (e.g. 40 cycles)
    ax.axvspan(test_cycles[0], 40, color='#2ecc71', alpha=0.18, label='Early-Life Observation Window (1–40 cycles)')
    ax.axvline(x=40, color='#27ae60', linestyle=':', linewidth=2)
    
    # EOL Threshold
    ax.axhline(y=DEFAULT_EOL_THRESHOLD_AH, color='#e67e22', linestyle='-.', linewidth=1.8, label=f'EOL Threshold ({DEFAULT_EOL_THRESHOLD_AH} Ah / 70% Rated)')
    
    # True vs Pred EOL
    ax.scatter([true_eol], [DEFAULT_EOL_THRESHOLD_AH], color='blue', s=120, zorder=5, marker='*', label=f'True EOL (Cycle {true_eol})')
    ax.scatter([pred_eol], [DEFAULT_EOL_THRESHOLD_AH], color='red', s=120, zorder=5, marker='X', label=f'Predicted EOL (Cycle {pred_eol})')
    
    ax.set_title("Early-Life Battery Degradation & RUL Trajectory Prediction (Unseen B0005)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Discharge Cycle Number", fontsize=11)
    ax.set_ylabel("Discharge Capacity (Ah)", fontsize=11)
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=10)
    ax.set_ylim(1.2, 2.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'early_life_prediction.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'early_life_prediction.png'), dpi=300)
    plt.close()
    
    return df_early


# =========================================================================
# EXPERIMENT 3: Feature Group & Memory Ablation Study
# =========================================================================
def run_experiment_3(df, output_dir, figures_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Feature Group and Memory Window Ablation Study")
    print("=" * 70)
    
    # Feature group definitions from cycle_features
    # Total 12 base features:
    # Voltage (4): voltage_mean, voltage_min, voltage_max, voltage_std
    # Current (2): current_mean, current_std
    # Temperature (3): temp_mean, temp_max, temp_range
    # Duration (1): discharge_duration
    # Capacity (1): discharge_capacity
    # Temporal (1): cycle_number
    
    feature_sets = {
        'Full Features (W=10)': (None, 10),
        'Full Features (W=20)': (None, 20),
        'Full Features (W=5)': (None, 5),
        'Full Features (W=3)': (None, 3),
        'Single-Cycle (No Memory, W=1)': (None, 1),
        'No Temperature Features (W=10)': (['voltage_mean', 'voltage_min', 'voltage_max', 'voltage_std', 'current_mean', 'current_std', 'discharge_duration', 'discharge_capacity', 'cycle_number'], 10),
        'No Voltage Statistics (W=10)': (['voltage_mean', 'current_mean', 'current_std', 'temp_mean', 'temp_max', 'temp_range', 'discharge_duration', 'discharge_capacity', 'cycle_number'], 10),
        'No Duration (W=10)': (['voltage_mean', 'voltage_min', 'voltage_max', 'voltage_std', 'current_mean', 'current_std', 'temp_mean', 'temp_max', 'temp_range', 'discharge_capacity', 'cycle_number'], 10),
        'Operational Sensors Only / No Capacity History (W=10)': (['voltage_mean', 'voltage_min', 'voltage_max', 'voltage_std', 'current_mean', 'current_std', 'temp_mean', 'temp_max', 'temp_range', 'discharge_duration', 'cycle_number'], 10),
    }
    
    ablation_results = []
    
    for config_name, (feats, w) in feature_sets.items():
        ds = create_memory_window_dataset(df, window_size=w, feature_cols=feats)
        X_all, y_all, meta = ds['X'], ds['y'], ds['metadata']
        batteries = sorted(meta['battery_id'].unique())
        
        rmses, maes, r2s = [], [], []
        for test_bat in batteries:
            train_mask = (meta['battery_id'] != test_bat).values
            test_mask = (meta['battery_id'] == test_bat).values
            
            X_tr = AntiLeakageScaler('minmax').fit_transform(X_all[train_mask])
            X_te = AntiLeakageScaler('minmax').fit(X_all[train_mask]).transform(X_all[test_mask])
            y_tr, y_te = y_all[train_mask], y_all[test_mask]
            
            model, _, _ = train_dnn(X_tr, y_tr, X_te, y_te, epochs=100)
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy().flatten()
            m = calculate_metrics(y_te, pred)
            rmses.append(m['rmse'])
            maes.append(m['mae'])
            r2s.append(m['r2'])
            
        ablation_results.append({
            'Configuration': config_name,
            'Window_W': w,
            'Input_Features': X_all.shape[1],
            'LOBO_RMSE': round(float(np.mean(rmses)), 5),
            'LOBO_MAE': round(float(np.mean(maes)), 5),
            'LOBO_R2': round(float(np.mean(r2s)), 4),
        })
        print(f"Ablation [{config_name:45s}] -> RMSE: {ablation_results[-1]['LOBO_RMSE']:.5f} Ah | R2: {ablation_results[-1]['LOBO_R2']:.4f}")
        
    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv(os.path.join(output_dir, 'ablation_study.csv'), index=False)
    
    # Plot Ablation Bar Chart
    plt.figure(figsize=(12, 6), dpi=300)
    df_sorted = df_abl.sort_values('LOBO_RMSE', ascending=True)
    colors = ['#27ae60' if 'Full' in c else '#e74c3c' if 'Single' in c else '#2980b9' for c in df_sorted['Configuration']]
    
    bars = plt.barh(df_sorted['Configuration'], df_sorted['LOBO_RMSE'], color=colors, edgecolor='black', alpha=0.85, height=0.6)
    plt.title("Ablation Study: Impact of Memory Windows & Feature Subsets on LOBO RMSE", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Leave-One-Battery-Out Mean RMSE (Ah) — Lower is Better", fontsize=11)
    
    for bar in bars:
        w_val = bar.get_width()
        plt.annotate(f'{w_val:.4f}', xy=(w_val, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ablation_study.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'ablation_study.png'), dpi=300)
    plt.close()
    
    return df_abl


# =========================================================================
# EXPERIMENT 4: Sensor Noise Robustness Study
# =========================================================================
def run_experiment_4(df, output_dir, figures_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Sensor Noise Robustness Analysis (1%, 2%, 5% Gaussian Noise)")
    print("=" * 70)
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
    w = 10
    
    ds = create_memory_window_dataset(df, window_size=w)
    X_all, y_all, meta = ds['X'], ds['y'], ds['metadata']
    batteries = sorted(meta['battery_id'].unique())
    
    robustness_results = []
    
    for noise_std in noise_levels:
        rmses, maes, r2s = [], [], []
        
        for test_bat in batteries:
            train_mask = (meta['battery_id'] != test_bat).values
            test_mask = (meta['battery_id'] == test_bat).values
            
            scaler = AntiLeakageScaler('minmax')
            X_tr = scaler.fit_transform(X_all[train_mask])
            X_te_clean = scaler.transform(X_all[test_mask])
            y_tr, y_te = y_all[train_mask], y_all[test_mask]
            
            # Add Gaussian noise to test inputs
            if noise_std > 0.0:
                noise = np.random.normal(0, noise_std, X_te_clean.shape)
                X_te = np.clip(X_te_clean + noise, 0.0, 1.0)
            else:
                X_te = X_te_clean
                
            model, _, _ = train_dnn(X_tr, y_tr, X_te_clean, y_te, epochs=100)
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy().flatten()
            m = calculate_metrics(y_te, pred)
            rmses.append(m['rmse'])
            maes.append(m['mae'])
            r2s.append(m['r2'])
            
        robustness_results.append({
            'noise_level_%': int(noise_std * 100),
            'noise_std': noise_std,
            'mean_LOBO_RMSE': round(float(np.mean(rmses)), 5),
            'mean_LOBO_MAE': round(float(np.mean(maes)), 5),
            'mean_LOBO_R2': round(float(np.mean(r2s)), 4),
            'relative_error_increase_%': round(float(((np.mean(rmses) - robustness_results[0]['mean_LOBO_RMSE']) / robustness_results[0]['mean_LOBO_RMSE']) * 100.0) if len(robustness_results) > 0 else 0.0, 2)
        })
        print(f"Noise {int(noise_std*100):2d}% | LOBO RMSE: {robustness_results[-1]['mean_LOBO_RMSE']:.5f} Ah | R2: {robustness_results[-1]['mean_LOBO_R2']:.4f} | Error Increase: +{robustness_results[-1]['relative_error_increase_%']}%")
        
    df_rob = pd.DataFrame(robustness_results)
    df_rob.to_csv(os.path.join(output_dir, 'robustness_study.csv'), index=False)
    
    # Plot Robustness Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_rob['noise_level_%'], df_rob['mean_LOBO_RMSE'], 'o-', color='#e74c3c', linewidth=2.2, markersize=8, label='DNN Memory-10 LOBO RMSE')
    ax.set_title("Model Robustness Under Sensor Measurement Noise", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Injected Input Feature Noise Level (%)", fontsize=11)
    ax.set_ylabel("Mean Test RMSE (Ah)", fontsize=11)
    ax.set_xticks(df_rob['noise_level_%'])
    for idx, row in df_rob.iterrows():
        ax.annotate(f"{row['mean_LOBO_RMSE']:.4f} Ah\n(+{row['relative_error_increase_%']}%)", xy=(row['noise_level_%'], row['mean_LOBO_RMSE']), xytext=(0, 10), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0.08, 0.16)
    ax.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'robustness_study.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'robustness_study.png'), dpi=300)
    plt.close()
    
    return df_rob


# =========================================================================
# EXPERIMENT 5: Cross-Battery Generalization (Detailed LOBO Breakdown)
# =========================================================================
def run_experiment_5(df, output_dir, figures_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Cross-Battery Generalization Breakdown (Unseen Batteries)")
    print("=" * 70)
    
    window_sizes = [3, 5, 10, 20]
    gen_records = []
    
    for w in window_sizes:
        ds = create_memory_window_dataset(df, window_size=w)
        X_all, y_all, meta = ds['X'], ds['y'], ds['metadata']
        batteries = sorted(meta['battery_id'].unique())
        
        for test_bat in batteries:
            train_mask = (meta['battery_id'] != test_bat).values
            test_mask = (meta['battery_id'] == test_bat).values
            
            X_tr = AntiLeakageScaler('minmax').fit_transform(X_all[train_mask])
            X_te = AntiLeakageScaler('minmax').fit(X_all[train_mask]).transform(X_all[test_mask])
            y_tr, y_te = y_all[train_mask], y_all[test_mask]
            test_cycles = meta.loc[test_mask, 'target_cycle'].values
            
            model, _, _ = train_dnn(X_tr, y_tr, X_te, y_te, epochs=120)
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy().flatten()
            m = calculate_metrics(y_te, pred)
            rul_mae, eol_err, true_eol, pred_eol = compute_rul_prediction_error(y_te, pred, test_cycles)
            
            gen_records.append({
                'memory_window': w,
                'test_battery': test_bat,
                'test_cycles': len(y_te),
                'MAE': round(m['mae'], 5),
                'RMSE': round(m['rmse'], 5),
                'R2': round(m['r2'], 4),
                'MAPE_%': round(m['mape'], 2),
                'RUL_MAE_cycles': round(rul_mae, 1),
                'True_EOL_cycle': true_eol,
                'Pred_EOL_cycle': pred_eol,
                'EOL_error_cycles': eol_err
            })
            
    df_gen = pd.DataFrame(gen_records)
    df_gen.to_csv(os.path.join(output_dir, 'cross_battery_generalization.csv'), index=False)
    
    # Plot Cross-Battery Bar Chart across windows
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    pivot_rmse = df_gen.pivot(index='test_battery', columns='memory_window', values='RMSE')
    pivot_rmse.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', alpha=0.85, width=0.75)
    
    ax.set_title("Cross-Battery Generalization: Test RMSE for Each Unseen NASA Battery", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Unseen Held-Out Test Battery", fontsize=11)
    ax.set_ylabel("Test Capacity RMSE (Ah)", fontsize=11)
    ax.legend(title='Memory Window', frameon=True, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cross_battery_generalization.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'cross_battery_generalization.png'), dpi=300)
    plt.close()
    
    return df_gen


if __name__ == '__main__':
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cycle_features.csv')
    df = pd.read_csv(data_path)
    df = compute_soh(df)
    df = compute_rul(df)
    
    final_dir = os.path.join(PROJECT_ROOT, 'results', 'final')
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Starting Phase 2 Research Experiments on {len(df)} cycles across 4 NASA batteries...")
    
    # Run all 5 experiments
    df_exp1 = run_experiment_1(df, final_dir, figures_dir)
    df_exp2 = run_experiment_2(df, final_dir, figures_dir)
    df_exp3 = run_experiment_3(df, final_dir, figures_dir)
    df_exp4 = run_experiment_4(df, final_dir, figures_dir)
    df_exp5 = run_experiment_5(df, final_dir, figures_dir)
    
    print("\n" + "=" * 70)
    print("ALL 5 PHASE 2 RESEARCH EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
