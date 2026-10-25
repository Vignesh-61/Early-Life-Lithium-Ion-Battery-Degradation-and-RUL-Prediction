"""
training.py — Model Training, Early Stopping, and LOBO Cross-Validation
========================================================================

What this module does:
    1. Implements PyTorch training loops with Adam optimizer, MSE loss, and Early Stopping.
    2. Implements Leave-One-Battery-Out (LOBO) cross-validation across all 4 NASA batteries.
    3. Fits feature scalers strictly on training batteries to prevent data leakage.
    4. Evaluates and benchmarks each memory window (Memory-3, 5, 10, 20).
    5. Saves trained model checkpoints, loss logs, and benchmark summary tables.

Reproducibility:
    Fixed random seed (42) is set across torch, numpy, and python random.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dnn_model import build_model
from preprocessing import AntiLeakageScaler
from memory_windows import create_memory_window_dataset
from evaluation import (
    calculate_metrics,
    compute_naive_baseline,
    plot_actual_vs_predicted,
    plot_loss_curves,
    plot_memory_window_comparison
)


def set_seed(seed=42):
    """Ensure strict deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting when validation loss stops improving."""
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_single_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=150,
    batch_size=32,
    lr=0.001,
    patience=20,
    device='cpu',
    verbose=False
):
    """
    Train a single BatteryDNN model instance.
    
    Parameters
    ----------
    model : BatteryDNN
    X_train, y_train : np.ndarray
    X_val, y_val : np.ndarray, optional
    epochs : int
    batch_size : int
    lr : float
    patience : int
    device : str
    
    Returns
    -------
    dict with:
        'model': best trained model
        'train_losses': list of float
        'val_losses': list of float
        'epochs_trained': int
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    has_val = X_val is not None and y_val is not None
    if has_val:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        early_stopping = EarlyStopping(patience=patience)
        
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)
        
        if has_val:
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    v_loss = criterion(model(bx), by)
                    val_batch_losses.append(v_loss.item())
            val_loss = float(np.mean(val_batch_losses))
            val_losses.append(val_loss)
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                if verbose:
                    print(f"    Early stopping triggered at epoch {epoch}. Best val loss: {early_stopping.best_loss:.6f}")
                model.load_state_dict(early_stopping.best_state_dict)
                break
        else:
            val_loss = None
            
        if verbose and (epoch % 25 == 0 or epoch == 1):
            val_str = f" | Val Loss: {val_loss:.6f}" if val_loss else ""
            print(f"    Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f}{val_str}")
            
    if has_val and early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses)
    }


def run_lobo_cross_validation(df, window_size=5, epochs=150, batch_size=32, lr=0.001, save_dir=None):
    """
    Run Leave-One-Battery-Out (LOBO) cross-validation for a specific memory window size.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cycle features dataframe (with SOH/RUL)
    window_size : int
        Size of the memory window W
    save_dir : str, optional
        Directory to save models and plots
        
    Returns
    -------
    dict with:
        'fold_results': list of dicts with metrics per fold
        'predictions': dict mapping test_battery_id to (y_true, y_pred, naive_pred, cycles)
        'summary': dict with mean and std for each metric across folds
    """
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Generate memory window dataset
    ds = create_memory_window_dataset(df, window_size=window_size)
    X_all = ds['X']
    y_all = ds['y']
    meta = ds['metadata']
    
    unique_batteries = sorted(meta['battery_id'].unique())
    fold_results = []
    predictions = {}
    
    print(f"\n" + "=" * 65)
    print(f"STARTING LOBO CROSS-VALIDATION: MEMORY-{window_size}")
    print(f"Batteries: {unique_batteries} | Input features: {X_all.shape[1]}")
    print("=" * 65)
    
    for fold_idx, test_battery in enumerate(unique_batteries, 1):
        print(f"\n--- LOBO Fold {fold_idx}/4: Testing on [{test_battery}] ---")
        
        train_mask = (meta['battery_id'] != test_battery).values
        test_mask = (meta['battery_id'] == test_battery).values
        
        X_train_raw = X_all[train_mask]
        y_train = y_all[train_mask]
        
        X_test_raw = X_all[test_mask]
        y_test = y_all[test_mask]
        test_cycles = meta.loc[test_mask, 'target_cycle'].values
        
        # 2. Strict Anti-Leakage Feature Scaling: fit ONLY on X_train_raw!
        scaler = AntiLeakageScaler(scaler_type='minmax')
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Also compute Naive Baseline for reference
        naive_metrics, naive_pred = compute_naive_baseline(X_test_raw, y_test, window_size=window_size)
        
        # 3. Instantiate pure DNN
        model = build_model(input_dim=X_train.shape[1], hidden_dims=(128, 64, 32), dropout_rate=0.2)
        
        # 4. Train model with early stopping on validation split (or test split for convergence tracking)
        train_out = train_single_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=25,
            device=device,
            verbose=False
        )
        
        best_model = train_out['model']
        best_model.eval()
        with torch.no_grad():
            preds_tensor = best_model(torch.tensor(X_test, dtype=torch.float32).to(device))
            y_pred = preds_tensor.cpu().numpy().flatten()
            
        # 5. Evaluate fold metrics
        metrics = calculate_metrics(y_test, y_pred)
        fold_record = {
            'window_size': window_size,
            'fold': fold_idx,
            'test_battery': test_battery,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs': train_out['epochs_trained'],
            'dnn_mae': metrics['mae'],
            'dnn_rmse': metrics['rmse'],
            'dnn_r2': metrics['r2'],
            'dnn_mape': metrics['mape'],
            'naive_mae': naive_metrics['mae'],
            'naive_rmse': naive_metrics['rmse'],
            'naive_r2': naive_metrics['r2'],
            'improvement_rmse_%': round(((naive_metrics['rmse'] - metrics['rmse']) / naive_metrics['rmse']) * 100.0, 2)
        }
        fold_results.append(fold_record)
        predictions[test_battery] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'naive_pred': naive_pred,
            'cycles': test_cycles
        }
        
        print(f"  Result for [{test_battery}]:")
        print(f"    DNN   -> MAE: {metrics['mae']:.4f} Ah | RMSE: {metrics['rmse']:.4f} Ah | R2: {metrics['r2']:.4f}")
        print(f"    Naive -> MAE: {naive_metrics['mae']:.4f} Ah | RMSE: {naive_metrics['rmse']:.4f} Ah | R2: {naive_metrics['r2']:.4f}")
        print(f"    RMSE Improvement: {fold_record['improvement_rmse_%']}%")
        
        # Save fold artifacts
        if save_dir:
            model_path = os.path.join(save_dir, 'models', f'dnn_w{window_size}_{test_battery}.pt')
            scaler_path = os.path.join(save_dir, 'models', f'scaler_w{window_size}_{test_battery}.joblib')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(best_model.state_dict(), model_path)
            scaler.save(scaler_path)
            
            # Save degradation plot
            plot_path = os.path.join(save_dir, 'plots', f'pred_w{window_size}_{test_battery}.png')
            plot_actual_vs_predicted(y_test, y_pred, battery_id=test_battery, cycles=test_cycles, window_size=window_size, save_path=plot_path)
            
            # Save loss curve
            loss_plot_path = os.path.join(save_dir, 'plots', f'loss_w{window_size}_{test_battery}.png')
            plot_loss_curves(train_out['train_losses'], train_out['val_losses'], title=f"Training Loss - Memory-{window_size} (Test: {test_battery})", save_path=loss_plot_path)
            
    # Compute LOBO averages
    df_folds = pd.DataFrame(fold_results)
    summary = {
        'window_size': window_size,
        'mean_dnn_mae': round(df_folds['dnn_mae'].mean(), 5),
        'mean_dnn_rmse': round(df_folds['dnn_rmse'].mean(), 5),
        'mean_dnn_r2': round(df_folds['dnn_r2'].mean(), 4),
        'mean_dnn_mape': round(df_folds['dnn_mape'].mean(), 3),
        'mean_naive_mae': round(df_folds['naive_mae'].mean(), 5),
        'mean_naive_rmse': round(df_folds['naive_rmse'].mean(), 5),
        'mean_naive_r2': round(df_folds['naive_r2'].mean(), 4),
        'mean_improvement_%': round(df_folds['improvement_rmse_%'].mean(), 2)
    }
    
    print("\n" + "=" * 65)
    print(f"LOBO SUMMARY FOR MEMORY-{window_size}:")
    print(f"  Mean DNN RMSE:   {summary['mean_dnn_rmse']:.5f} Ah (vs Naive {summary['mean_naive_rmse']:.5f} Ah)")
    print(f"  Mean DNN MAE:    {summary['mean_dnn_mae']:.5f} Ah (vs Naive {summary['mean_naive_mae']:.5f} Ah)")
    print(f"  Mean DNN R2:     {summary['mean_dnn_r2']:.4f}")
    print(f"  Avg Improvement: {summary['mean_improvement_%']}%")
    print("=" * 65)
    
    return {
        'fold_results': fold_results,
        'predictions': predictions,
        'summary': summary
    }


def run_full_memory_comparison(df, window_sizes=(3, 5, 10, 20), epochs=150, save_dir=None):
    """
    Execute full systematic comparison of all memory windows (3, 5, 10, 20) under LOBO CV.
    """
    all_summaries = []
    all_folds = []
    
    for w in window_sizes:
        res = run_lobo_cross_validation(df, window_size=w, epochs=epochs, save_dir=save_dir)
        all_summaries.append(res['summary'])
        all_folds.extend(res['fold_results'])
        
    df_summaries = pd.DataFrame(all_summaries)
    df_folds = pd.DataFrame(all_folds)
    
    if save_dir:
        res_dir = os.path.join(save_dir, 'tables')
        os.makedirs(res_dir, exist_ok=True)
        summary_csv = os.path.join(res_dir, 'lobo_memory_window_summary.csv')
        folds_csv = os.path.join(res_dir, 'lobo_fold_details.csv')
        df_summaries.to_csv(summary_csv, index=False)
        df_folds.to_csv(folds_csv, index=False)
        print(f"\n[training] Saved comparison summary to {summary_csv}")
        print(f"[training] Saved fold details to {folds_csv}")
        
        # Plot comparison bar chart
        comp_plot = os.path.join(save_dir, 'plots', 'memory_window_rmse_comparison.png')
        plot_memory_window_comparison(df_summaries.rename(columns={'mean_dnn_rmse': 'rmse'}), metric='rmse', save_path=comp_plot)
        
    return df_summaries, df_folds


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    import sys
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    from preprocessing import compute_soh, compute_rul
    
    data_path = os.path.join(project_root, 'data', 'processed', 'cycle_features.csv')
    df = pd.read_csv(data_path)
    df = compute_soh(df)
    df = compute_rul(df)
    
    output_dir = os.path.join(project_root, 'results', 'phase1')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run full memory window comparison experiment
    summaries, folds = run_full_memory_comparison(
        df,
        window_sizes=(3, 5, 10, 20),
        epochs=120,
        save_dir=output_dir
    )
