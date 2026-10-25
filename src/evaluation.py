"""
evaluation.py — Evaluation Metrics, Baselines, and Plotting Utilities
======================================================================

What this module does:
    1. Computes academic-standard regression metrics: MAE, RMSE, R², MAPE, Max Error.
    2. Implements the Naive Persistence Baseline (predicting next cycle = current cycle).
    3. Generates high-quality research visualizations:
       - Actual vs Predicted capacity/SOH degradation curves
       - Training vs Validation loss curves
       - Memory Window comparison bar charts (Memory-3 vs 5 vs 10 vs 20)
       - Leave-One-Battery-Out fold comparison tables

Why we need it:
    - In deep learning research, a model's worth is proven through comparative metrics
      and baselines. Beating a naive baseline (yesterday's capacity = today's capacity)
      is a fundamental sanity check.
    - Clear, publication-ready figures communicate the degradation trajectory and
      error distribution effectively.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    """
    Compute key regression evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or list
        Ground-truth values (in physical units, e.g., Ah).
    y_pred : np.ndarray or list
        Model predicted values.
        
    Returns
    -------
    dict with:
        'mae': float (Mean Absolute Error)
        'rmse': float (Root Mean Squared Error)
        'r2': float (Coefficient of Determination)
        'mape': float (Mean Absolute Percentage Error in %)
        'max_error': float (Maximum single-cycle absolute error)
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # R2 Score: 1 - SS_res / SS_tot
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # MAPE (safe division avoiding divide-by-zero)
    safe_y_true = np.where(y_true == 0, 1e-6, y_true)
    mape = float(np.mean(abs_errors / np.abs(safe_y_true)) * 100.0)
    
    max_error = float(np.max(abs_errors))
    
    return {
        'mae': round(mae, 5),
        'rmse': round(rmse, 5),
        'r2': round(r2, 4),
        'mape': round(mape, 3),
        'max_error': round(max_error, 5)
    }


def compute_naive_baseline(X, y, window_size, capacity_feature_idx=10):
    """
    Compute Naive Persistence Baseline:
    Predicts the next cycle capacity equal to the most recent cycle capacity in the window.
    
    Parameters
    ----------
    X : np.ndarray of shape (N, W * num_features)
    y : np.ndarray of shape (N,)
    window_size : int
    capacity_feature_idx : int
        Index of the discharge_capacity feature within each cycle (default 10 out of 12).
        
    Returns
    -------
    dict with metrics for the naive baseline.
    """
    # The most recent cycle in the window is the last cycle: index W - 1
    # Its features start at (W - 1) * num_features
    num_features_per_cycle = X.shape[1] // window_size
    most_recent_cap_idx = (window_size - 1) * num_features_per_cycle + capacity_feature_idx
    y_pred_naive = X[:, most_recent_cap_idx]
    
    return calculate_metrics(y, y_pred_naive), y_pred_naive


def plot_actual_vs_predicted(y_true, y_pred, battery_id, cycles=None, window_size=5, save_path=None):
    """
    Plot actual degradation curve vs DNN predictions for a battery.
    """
    plt.figure(figsize=(10, 5))
    
    x_axis = cycles if cycles is not None else np.arange(len(y_true))
    
    plt.plot(x_axis, y_true, 'b-', label='Actual Capacity (Ah)', linewidth=2.0)
    plt.plot(x_axis, y_pred, 'r--', label=f'DNN (Memory-{window_size}) Predicted', linewidth=1.8)
    
    plt.axhline(y=1.4, color='gray', linestyle=':', label='EOL Threshold (1.4 Ah)')
    
    metrics = calculate_metrics(y_true, y_pred)
    title_text = f"Battery {battery_id} — Actual vs Predicted Capacity (Memory-{window_size})\n" \
                 f"MAE: {metrics['mae']:.4f} Ah | RMSE: {metrics['rmse']:.4f} Ah | R²: {metrics['r2']:.4f}"
    
    plt.title(title_text, fontsize=12, pad=12)
    plt.xlabel("Discharge Cycle Number", fontsize=11)
    plt.ylabel("Discharge Capacity (Ah)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[evaluation] Saved plot to {save_path}")
    plt.close()


def plot_loss_curves(train_losses, val_losses, title="DNN Training Loss Convergence", save_path=None):
    """
    Plot training and validation MSE loss curves over epochs.
    """
    plt.figure(figsize=(8, 4.5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss (MSE)', linewidth=1.8)
    if val_losses and len(val_losses) > 0:
        plt.plot(epochs, val_losses, 'orange', label='Validation Loss (MSE)', linewidth=1.8)
        
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE Loss", fontsize=11)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[evaluation] Saved loss curve to {save_path}")
    plt.close()


def plot_memory_window_comparison(results_df, metric='rmse', save_path=None):
    """
    Plot bar comparison of performance across different memory windows (3, 5, 10, 20).
    """
    plt.figure(figsize=(8, 5))
    
    windows = [f"Memory-{w}" for w in results_df['window_size']]
    values = results_df[metric].values
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = plt.bar(windows, values, color=colors[:len(windows)], width=0.55, edgecolor='black', alpha=0.85)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.02 * max(values)), f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    metric_labels = {'rmse': 'RMSE (Ah)', 'mae': 'MAE (Ah)', 'r2': 'R² Score', 'mape': 'MAPE (%)'}
    ylabel = metric_labels.get(metric, metric.upper())
    
    plt.title(f"Memory Window Comparison — Average LOBO {ylabel}", fontsize=12, pad=12)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[evaluation] Saved comparison plot to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Unit tests for metric calculation
    y_t = np.array([1.8, 1.7, 1.6, 1.5])
    y_p = np.array([1.79, 1.71, 1.58, 1.52])
    m = calculate_metrics(y_t, y_p)
    print("[evaluation] Metrics test:", m)
    assert m['mae'] > 0 and m['r2'] > 0.95
    print("[evaluation] Metrics unit test passed.")
