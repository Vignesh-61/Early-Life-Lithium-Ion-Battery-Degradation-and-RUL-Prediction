"""
preprocessing.py — Data Preprocessing, Scaling, and Target Computation
========================================================================

What this module does:
    1. Computes State of Health (SOH %) from discharge capacity.
    2. Computes Remaining Useful Life (RUL in cycles) relative to an End-of-Life (EOL) threshold.
    3. Implements strict anti-leakage feature scaling (fit on train batteries ONLY).
    4. Generates Leave-One-Battery-Out (LOBO) train/test splits.

Why we need it:
    - Deep Neural Networks are sensitive to feature scales. If features have vastly
      different ranges (e.g., discharge duration is ~3000s while voltage is ~3.5V),
      gradients will oscillate and training will be unstable.
    - Data leakage is the #1 mistake in battery prognostics: fitting a scaler on the
      entire dataset leaks information from test batteries to the model.
      We enforce that scalers are fitted EXCLUSIVELY on training batteries.

Target Definitions:
    - Capacity (Ah): Direct physical measurement (e.g. 1.85 Ah -> 1.33 Ah)
    - SOH (%): Capacity / Initial_Capacity * 100
    - RUL (cycles): Cycles remaining until capacity falls below EOL threshold (default 1.40 Ah)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Default NASA Li-ion battery ratings
RATED_CAPACITY_AH = 2.0
DEFAULT_EOL_THRESHOLD_AH = 1.4  # 70% of 2.0 Ah rated capacity


def compute_soh(df, initial_capacity_col=None, rated_capacity=None):
    """
    Compute State of Health (SOH) percentage for each cycle.
    
    Formula:
        If using battery-specific initial capacity:
            SOH (%) = (discharge_capacity / C_initial_battery) * 100
        If using nominal rated capacity:
            SOH (%) = (discharge_capacity / rated_capacity) * 100
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'battery_id' and 'discharge_capacity'
    initial_capacity_col : str, optional
        Column name if already computed, otherwise computed from cycle 0
    rated_capacity : float, optional
        Nominal capacity in Ah (e.g., 2.0). If provided, computes SOH relative to nominal.
        If None, computes SOH relative to each battery's first observed cycle capacity.
        
    Returns
    -------
    pd.DataFrame : df with new column 'soh_percent'
    """
    df = df.copy()
    
    if rated_capacity is not None:
        df['soh_percent'] = (df['discharge_capacity'] / rated_capacity) * 100.0
    else:
        # Group by battery and use each battery's cycle 0 capacity as 100% reference
        initial_caps = df.groupby('battery_id')['discharge_capacity'].transform('first')
        df['soh_percent'] = (df['discharge_capacity'] / initial_caps) * 100.0
        
    return df


def compute_rul(df, eol_threshold_ah=DEFAULT_EOL_THRESHOLD_AH):
    """
    Compute Remaining Useful Life (RUL) in discharge cycles.
    
    Definition:
        For a given battery at cycle t:
        RUL(t) = max(0, cycle_eol - t)
        where cycle_eol is the FIRST cycle index where discharge_capacity <= eol_threshold_ah.
        If a battery never drops below eol_threshold_ah, the last available cycle is used as EOL.
        
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'battery_id', 'cycle_number', and 'discharge_capacity'
    eol_threshold_ah : float
        End of life capacity threshold in Ah (default 1.4 Ah)
        
    Returns
    -------
    pd.DataFrame : df with new column 'rul_cycles'
    """
    df = df.copy()
    rul_list = []
    
    for battery_id, group in df.groupby('battery_id', sort=False):
        group = group.sort_values('cycle_number').reset_index(drop=True)
        
        # Find first cycle below EOL threshold
        below_eol = group[group['discharge_capacity'] <= eol_threshold_ah]
        if len(below_eol) > 0:
            eol_cycle = int(below_eol.iloc[0]['cycle_number'])
        else:
            # Battery did not reach EOL in available cycles; use last recorded cycle
            eol_cycle = int(group.iloc[-1]['cycle_number'])
            
        group['rul_cycles'] = group['cycle_number'].apply(lambda c: max(0, eol_cycle - int(c)))
        rul_list.append(group)
        
    result = pd.concat(rul_list, ignore_index=True)
    return result


class AntiLeakageScaler:
    """
    A robust scaler that guarantees NO data leakage between train and test batteries.
    
    How it works:
        1. fit(X_train): Learns min/max (or mean/std) from training data ONLY.
        2. transform(X): Applies the learned transformation to train, val, or test data.
        3. inverse_transform(X): Converts scaled predictions back to original physical units (Ah).
    """
    def __init__(self, scaler_type='minmax', feature_range=(0, 1)):
        self.scaler_type = scaler_type
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}. Choose 'minmax' or 'standard'.")
        self.is_fitted = False

    def fit(self, X):
        """Fit scaler on training array or DataFrame."""
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform data using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before calling transform().")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        """Fit and transform in one step (ONLY use on training data!)."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Convert scaled values back to original domain."""
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        return self.scaler.inverse_transform(X)

    def save(self, filepath):
        """Save fitted scaler to disk."""
        joblib.dump({'scaler': self.scaler, 'type': self.scaler_type, 'fitted': self.is_fitted}, filepath)

    def load(self, filepath):
        """Load fitted scaler from disk."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.scaler_type = data['type']
        self.is_fitted = data['fitted']
        return self


def get_lobo_splits(df):
    """
    Generate Leave-One-Battery-Out (LOBO) splits.
    
    For N batteries, generates N folds. In each fold:
      - Train set: All samples from (N - 1) batteries
      - Test set: All samples from the 1 held-out battery
      
    Parameters
    ----------
    df : pd.DataFrame
        Must have 'battery_id' column
        
    Yields
    ------
    (train_df, test_df, test_battery_id) : tuple
    """
    unique_batteries = sorted(df['battery_id'].unique())
    print(f"[preprocessing] Generating LOBO splits for {len(unique_batteries)} batteries: {unique_batteries}")
    
    for test_battery in unique_batteries:
        train_df = df[df['battery_id'] != test_battery].copy().reset_index(drop=True)
        test_df = df[df['battery_id'] == test_battery].copy().reset_index(drop=True)
        yield train_df, test_df, test_battery


if __name__ == '__main__':
    # Self-test script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_path = os.path.join(project_root, 'data', 'processed', 'cycle_features.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"[preprocessing] Loaded {len(df)} cycles from {data_path}")
        
        # Test SOH calculation
        df_soh = compute_soh(df)
        print(f"[preprocessing] SOH computed: min={df_soh['soh_percent'].min():.2f}%, max={df_soh['soh_percent'].max():.2f}%")
        
        # Test RUL calculation
        df_rul = compute_rul(df_soh, eol_threshold_ah=1.4)
        for bat, grp in df_rul.groupby('battery_id'):
            print(f"  Battery {bat}: Initial Capacity={grp['discharge_capacity'].iloc[0]:.4f} Ah, Final={grp['discharge_capacity'].iloc[-1]:.4f} Ah, Max RUL={grp['rul_cycles'].max()} cycles")
            
        # Test LOBO splits
        for train_df, test_df, test_bat in get_lobo_splits(df_rul):
            print(f"  LOBO Fold [Test: {test_bat}]: Train={len(train_df)} samples, Test={len(test_df)} samples")
            
        # Test anti-leakage scaler
        scaler = AntiLeakageScaler(scaler_type='minmax')
        train_df, test_df, test_bat = next(get_lobo_splits(df_rul))
        cols = ['voltage_mean', 'discharge_capacity', 'cycle_number']
        X_train = scaler.fit_transform(train_df[cols].values)
        X_test = scaler.transform(test_df[cols].values)
        print(f"[preprocessing] Scaler test passed. X_train min={X_train.min():.2f}, max={X_train.max():.2f}")
    else:
        print(f"[preprocessing] Data file {data_path} not found. Run feature_engineering.py first.")
