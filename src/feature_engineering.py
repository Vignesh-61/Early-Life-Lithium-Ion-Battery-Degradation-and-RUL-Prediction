"""
feature_engineering.py — Extract cycle-level features from raw battery data
============================================================================

What this module does:
    Converts raw time-series measurements (voltage, current, temperature
    arrays with hundreds of data points per cycle) into a fixed-size
    feature vector for each discharge cycle.

Why we need it:
    A fully connected DNN needs a fixed-size input vector. Raw battery
    data has variable-length time series per cycle. We compute summary
    statistics (mean, max, std, etc.) that capture the essential
    information from each cycle in a compact, fixed-size format.

How it works:
    For each discharge cycle, we compute 12 features from the raw
    measurement arrays. These features are chosen because they are:
    1. Actually available in the NASA dataset (verified)
    2. Scientifically meaningful for battery degradation
    3. Free from data leakage concerns (when used properly)

What happens if we don't use it:
    We cannot feed variable-length raw time series directly into
    a fully connected DNN. The model needs consistent input dimensions.
"""

import numpy as np
import pandas as pd


# ---- Feature definitions ----
# Each feature is documented with what it means and why it matters

FEATURE_NAMES = [
    'voltage_mean',       # Average terminal voltage during discharge
    'voltage_min',        # Minimum voltage (near cutoff) — indicates depth of discharge
    'voltage_max',        # Maximum voltage at start of discharge
    'voltage_std',        # Voltage variation — changes as battery ages
    'current_mean',       # Average discharge current (should be ~constant for CC discharge)
    'current_std',        # Current variation — indicates stability
    'temp_mean',          # Average temperature during discharge
    'temp_max',           # Peak temperature — thermal stress indicator
    'temp_range',         # Temperature rise during discharge — heat generation indicator
    'discharge_duration', # Time to complete discharge (seconds) — shorter = degraded
    'discharge_capacity', # Measured capacity (Ah) — DIRECT degradation indicator
    'cycle_number',       # Discharge cycle index — temporal aging indicator
]


def extract_cycle_features(cycle_data, discharge_number):
    """
    Extract a fixed-size feature vector from one discharge cycle.
    
    Parameters
    ----------
    cycle_data : dict
        The 'data' dict from a discharge cycle, containing arrays like
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Time', 'Capacity'
    
    discharge_number : int
        The sequential discharge cycle number (0-indexed)
    
    Returns
    -------
    dict : feature_name -> float value
    
    Raises
    ------
    ValueError : if required fields are missing
    """
    features = {}
    
    # ---- Voltage features ----
    # Voltage_measured is the battery terminal voltage during discharge.
    # As a battery degrades, its voltage curve shape and levels change.
    voltage = cycle_data.get('Voltage_measured')
    if voltage is None or len(voltage) == 0:
        raise ValueError("Missing Voltage_measured in discharge cycle")
    
    features['voltage_mean'] = float(np.mean(voltage))
    features['voltage_min'] = float(np.min(voltage))
    features['voltage_max'] = float(np.max(voltage))
    features['voltage_std'] = float(np.std(voltage))
    
    # ---- Current features ----
    # Current_measured is the discharge current. In theory it's constant
    # (CC discharge at 2A), but small variations can be informative.
    current = cycle_data.get('Current_measured')
    if current is None or len(current) == 0:
        raise ValueError("Missing Current_measured in discharge cycle")
    
    features['current_mean'] = float(np.mean(current))
    features['current_std'] = float(np.std(current))
    
    # ---- Temperature features ----
    # Temperature_measured is the battery surface temperature.
    # As batteries degrade, internal resistance increases, causing
    # more heat generation during discharge.
    temperature = cycle_data.get('Temperature_measured')
    if temperature is None or len(temperature) == 0:
        raise ValueError("Missing Temperature_measured in discharge cycle")
    
    features['temp_mean'] = float(np.mean(temperature))
    features['temp_max'] = float(np.max(temperature))
    features['temp_range'] = float(np.max(temperature) - np.min(temperature))
    
    # ---- Time features ----
    # Time is the elapsed time in seconds during the discharge.
    # As capacity fades, discharge duration decreases because the
    # battery runs out of charge faster.
    time_vec = cycle_data.get('Time')
    if time_vec is not None and len(time_vec) > 1:
        features['discharge_duration'] = float(np.max(time_vec) - np.min(time_vec))
    else:
        features['discharge_duration'] = 0.0
    
    # ---- Capacity ----
    # Capacity is the total charge delivered during discharge (Ah).
    # This is the MOST DIRECT indicator of battery health.
    # IMPORTANT: When used as an INPUT feature, it represents the
    # PAST observed capacity. The TARGET will be a FUTURE capacity.
    capacity = cycle_data.get('Capacity')
    if capacity is not None:
        if isinstance(capacity, np.ndarray):
            features['discharge_capacity'] = float(np.max(capacity))
        else:
            features['discharge_capacity'] = float(capacity)
    else:
        features['discharge_capacity'] = np.nan
    
    # ---- Cycle number ----
    # The sequential discharge cycle number. This serves as an aging
    # indicator — batteries degrade more as cycle count increases.
    features['cycle_number'] = float(discharge_number)
    
    return features


def extract_all_features(battery_data):
    """
    Extract features from ALL discharge cycles of one battery.
    
    Parameters
    ----------
    battery_data : dict
        Output from data_loader.load_battery_mat()
    
    Returns
    -------
    pd.DataFrame : one row per discharge cycle, columns = FEATURE_NAMES
    """
    battery_id = battery_data['battery_id']
    all_features = []
    discharge_num = 0
    skipped = 0
    
    for cycle in battery_data['cycles']:
        if cycle['type'] == 'discharge':
            try:
                features = extract_cycle_features(cycle['data'], discharge_num)
                features['battery_id'] = battery_id
                all_features.append(features)
            except (ValueError, KeyError) as e:
                print(f"  Warning: Skipping discharge cycle {discharge_num} "
                      f"of {battery_id}: {e}")
                skipped += 1
            discharge_num += 1
    
    if not all_features:
        raise ValueError(f"No valid discharge cycles found for {battery_id}")
    
    df = pd.DataFrame(all_features)
    
    # Reorder columns: battery_id first, then features in defined order
    cols = ['battery_id'] + [f for f in FEATURE_NAMES if f in df.columns]
    df = df[cols]
    
    print(f"[features] {battery_id}: Extracted {len(df)} cycles, "
          f"{skipped} skipped, shape={df.shape}")
    
    return df


def extract_features_all_batteries(batteries_dict):
    """
    Extract features from all batteries and combine into one DataFrame.
    
    Parameters
    ----------
    batteries_dict : dict
        battery_id -> battery_data dict (from load_all_batteries)
    
    Returns
    -------
    pd.DataFrame : all batteries combined, with 'battery_id' column
    """
    all_dfs = []
    
    for battery_id in sorted(batteries_dict.keys()):
        df = extract_all_features(batteries_dict[battery_id])
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n[features] Combined dataset: {combined.shape[0]} samples, "
          f"{combined.shape[1]} columns")
    print(f"[features] Batteries: {sorted(combined['battery_id'].unique())}")
    print(f"[features] Feature columns: {FEATURE_NAMES}")
    
    return combined


def check_feature_quality(df):
    """
    Check for data quality issues in the extracted features.
    
    Prints warnings about:
    - Missing values (NaN)
    - Constant features (zero variance)
    - Suspicious outliers
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from extract_all_features or extract_features_all_batteries
    """
    print("\n" + "=" * 60)
    print("FEATURE QUALITY CHECK")
    print("=" * 60)
    
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    
    # Check missing values
    missing = df[feature_cols].isnull().sum()
    if missing.any():
        print("\n[WARN] Missing values found:")
        for col in feature_cols:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} missing "
                      f"({missing[col]/len(df)*100:.1f}%)")
    else:
        print("\n[OK] No missing values")
    
    # Check for constant features
    print("\nFeature statistics:")
    for col in feature_cols:
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  {col:25s}: min={min_val:10.4f}, max={max_val:10.4f}, "
              f"std={std:10.4f}")
        if std == 0:
            print(f"    [WARN] Zero variance -- feature provides no information")
    
    # Check for NaN or Inf
    inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"\n[WARN] Found {inf_count} infinite values!")
    else:
        print("\n[OK] No infinite values")
    
    print("=" * 60)


# ---- Quick test ----
if __name__ == '__main__':
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    from data_loader import load_all_batteries
    
    data_dir = os.path.join(project_root, 'data', 'raw')
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    batteries = load_all_batteries(data_dir)
    
    combined_df = extract_features_all_batteries(batteries)
    check_feature_quality(combined_df)
    
    # Save to CSV
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cycle_features.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved cycle features to {output_path}")
    print(f"Shape: {combined_df.shape}")
