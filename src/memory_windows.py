"""
memory_windows.py — Historical Memory Window Dataset Generator
================================================================

What this module does:
    Transforms sequential battery cycle features into memory-windowed
    datasets suitable for feed-forward Deep Neural Networks (DNNs).
    For a memory window of size W, the input to the DNN at step t consists
    of the flattened features from cycles [t - W + 1, ..., t], and the
    prediction target is the battery capacity (and SOH/RUL) at cycle t + 1.

Why we need it:
    - Pure feed-forward DNNs do not possess recurrence (unlike RNN/LSTM) or
      attention mechanisms (unlike Transformers).
    - To capture temporal degradation trajectories, we explicitly construct
      an "adaptive memory window" of past W cycles as input features.
    - We systematically compare W in {3, 5, 10, 20} to answer the core research
      question: How much historical memory is optimal for battery degradation prediction?

Strict Anti-Leakage Rules:
    1. NEVER construct a window that spans across two different batteries.
    2. Input window includes cycles [t - W + 1, ..., t].
    3. Target is cycle t + 1 (strictly future, no lookahead in input).
"""

import os
import numpy as np
import pandas as pd
from feature_engineering import FEATURE_NAMES


def create_memory_window_dataset(df, window_size=5, feature_cols=None, target_col='discharge_capacity'):
    """
    Create a memory-windowed dataset for a given window size W.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'battery_id', 'cycle_number', target_col, and feature_cols.
    window_size : int
        Number of consecutive historical cycles to include in the input vector (e.g. 3, 5, 10, 20).
    feature_cols : list of str, optional
        List of cycle feature names. Defaults to FEATURE_NAMES.
    target_col : str
        Target column name (default 'discharge_capacity').
        
    Returns
    -------
    dict with:
        'X': np.ndarray of shape (N_samples, window_size * len(feature_cols))
        'y': np.ndarray of shape (N_samples,)
        'metadata': pd.DataFrame with columns ['battery_id', 'target_cycle', 'target_capacity', 'target_soh', 'target_rul']
        'feature_names': list of str describing each flattened column
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
        
    num_features_per_cycle = len(feature_cols)
    print(f"[memory_windows] Generating Memory-{window_size} dataset...")
    print(f"  Window size W={window_size}, features per cycle={num_features_per_cycle}")
    print(f"  Input dimensionality = {window_size} x {num_features_per_cycle} = {window_size * num_features_per_cycle}")
    
    # Generate human-readable feature column names
    # e.g., 'w_minus_4_voltage_mean', ..., 'w_0_voltage_mean'
    window_feature_names = []
    for step in range(window_size):
        offset = -(window_size - 1 - step)
        prefix = f"t{offset:+d}" if offset != 0 else "t0"
        for col in feature_cols:
            window_feature_names.append(f"{prefix}_{col}")
            
    X_list = []
    y_list = []
    meta_list = []
    
    # Process each battery independently to NEVER cross battery boundaries
    for battery_id, bat_df in df.groupby('battery_id', sort=False):
        bat_df = bat_df.sort_values('cycle_number').reset_index(drop=True)
        n_cycles = len(bat_df)
        
        # We need at least window_size past cycles + 1 future cycle
        if n_cycles <= window_size:
            print(f"  [WARN] Battery {battery_id} has {n_cycles} cycles <= window_size {window_size}. Skipping.")
            continue
            
        feature_matrix = bat_df[feature_cols].values
        target_series = bat_df[target_col].values
        
        # Optional metadata columns
        has_soh = 'soh_percent' in bat_df.columns
        has_rul = 'rul_cycles' in bat_df.columns
        
        # Sliding window from cycle index W-1 to n_cycles-2
        # So input is [i - W + 1 : i + 1] and target is at cycle index i + 1
        samples_count = 0
        for i in range(window_size - 1, n_cycles - 1):
            window_data = feature_matrix[i - window_size + 1 : i + 1]
            flattened = window_data.flatten()
            target_val = target_series[i + 1]
            
            target_cycle = bat_df.iloc[i + 1]['cycle_number']
            soh_val = bat_df.iloc[i + 1]['soh_percent'] if has_soh else np.nan
            rul_val = bat_df.iloc[i + 1]['rul_cycles'] if has_rul else np.nan
            
            X_list.append(flattened)
            y_list.append(target_val)
            meta_list.append({
                'battery_id': battery_id,
                'target_cycle': target_cycle,
                'target_capacity': target_val,
                'target_soh': soh_val,
                'target_rul': rul_val
            })
            samples_count += 1
            
        print(f"  Battery {battery_id}: {samples_count} window samples generated from {n_cycles} cycles")
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_list)
    
    print(f"[memory_windows] Memory-{window_size} completed: X shape={X.shape}, y shape={y.shape}")
    
    return {
        'X': X,
        'y': y,
        'metadata': meta_df,
        'feature_names': window_feature_names,
        'window_size': window_size
    }


def generate_all_memory_windows(df, window_sizes=(3, 5, 10, 20), output_dir=None):
    """
    Generate and optionally save memory-window datasets for all target window sizes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned cycle features DataFrame (with SOH and RUL computed).
    window_sizes : tuple of int
        Tuple of window sizes to evaluate (default: 3, 5, 10, 20).
    output_dir : str, optional
        Directory to save processed CSV datasets.
        
    Returns
    -------
    dict : window_size -> dataset dict
    """
    datasets = {}
    
    for w in window_sizes:
        ds = create_memory_window_dataset(df, window_size=w)
        datasets[w] = ds
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Combine into a single readable DataFrame for saving
            export_df = pd.DataFrame(ds['X'], columns=ds['feature_names'])
            export_df['target_capacity'] = ds['y']
            for col in ['battery_id', 'target_cycle', 'target_soh', 'target_rul']:
                if col in ds['metadata'].columns:
                    export_df[col] = ds['metadata'][col]
                    
            save_path = os.path.join(output_dir, f'dataset_memory_{w}.csv')
            export_df.to_csv(save_path, index=False)
            print(f"  Saved Memory-{w} dataset to {save_path} ({export_df.shape})")
            
    return datasets


if __name__ == '__main__':
    # Self-test script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    import sys
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    from preprocessing import compute_soh, compute_rul
    
    data_path = os.path.join(project_root, 'data', 'processed', 'cycle_features.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = compute_soh(df)
        df = compute_rul(df, eol_threshold_ah=1.4)
        
        out_dir = os.path.join(project_root, 'data', 'processed')
        datasets = generate_all_memory_windows(df, window_sizes=(3, 5, 10, 20), output_dir=out_dir)
        
        print("\n" + "=" * 60)
        print("MEMORY WINDOW SUMMARY")
        print("=" * 60)
        for w, ds in datasets.items():
            print(f"Memory-{w:2d}: Samples={ds['X'].shape[0]}, Features={ds['X'].shape[1]}")
        print("=" * 60)
    else:
        print(f"Data file not found at {data_path}")
