"""
test_pipeline.py — Unit and Integration Tests for Phase 1 Battery DNN Pipeline
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

# Add src to sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_battery_mat, extract_discharge_cycles
from feature_engineering import extract_cycle_features, FEATURE_NAMES
from preprocessing import compute_soh, compute_rul, AntiLeakageScaler, get_lobo_splits
from memory_windows import create_memory_window_dataset
from dnn_model import build_model
from evaluation import calculate_metrics, compute_naive_baseline

DATA_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
DATA_PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))


def test_data_loader_b0005():
    """Verify loading B0005.mat structure and cycle identification."""
    mat_path = os.path.join(DATA_RAW_DIR, 'B0005.mat')
    assert os.path.exists(mat_path), f"B0005.mat missing from {DATA_RAW_DIR}"
    
    bat_data = load_battery_mat(mat_path)
    assert bat_data['battery_id'] == 'B0005'
    assert len(bat_data['cycles']) == 616
    
    discharge_cycles = extract_discharge_cycles(bat_data)
    assert len(discharge_cycles) == 168


def test_feature_engineering_structure():
    """Verify cycle feature extraction yields 12 valid features with no NaNs."""
    dummy_cycle_data = {
        'Voltage_measured': np.linspace(4.2, 2.7, 300),
        'Current_measured': np.full(300, -2.0),
        'Temperature_measured': np.linspace(24.0, 36.0, 300),
        'Time': np.linspace(0, 3000, 300),
        'Capacity': 1.85
    }
    features = extract_cycle_features(dummy_cycle_data, discharge_number=0)
    assert len(features) == 12
    for fname in FEATURE_NAMES:
        assert fname in features
        assert not np.isnan(features[fname])
    assert features['voltage_max'] == pytest.approx(4.2, rel=1e-3)
    assert features['voltage_min'] == pytest.approx(2.7, rel=1e-3)
    assert features['discharge_capacity'] == pytest.approx(1.85, rel=1e-3)


def test_preprocessing_soh_rul():
    """Verify SOH and RUL formulas."""
    sample_df = pd.DataFrame({
        'battery_id': ['B1', 'B1', 'B1'],
        'cycle_number': [0, 1, 2],
        'discharge_capacity': [2.0, 1.6, 1.2]
    })
    df_soh = compute_soh(sample_df)
    assert df_soh['soh_percent'].iloc[0] == 100.0
    assert df_soh['soh_percent'].iloc[1] == 80.0
    assert df_soh['soh_percent'].iloc[2] == 60.0
    
    df_rul = compute_rul(df_soh, eol_threshold_ah=1.4)
    # EOL reached at cycle 2 (1.2 Ah <= 1.4 Ah)
    assert df_rul['rul_cycles'].iloc[0] == 2
    assert df_rul['rul_cycles'].iloc[1] == 1
    assert df_rul['rul_cycles'].iloc[2] == 0


def test_anti_leakage_scaler():
    """Verify scaler fits strictly on train and transforms test without leakage."""
    X_train = np.array([[10.0, 100.0], [20.0, 200.0]])
    X_test = np.array([[15.0, 150.0]])
    
    scaler = AntiLeakageScaler(scaler_type='minmax')
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    
    assert X_tr_scaled.min() == 0.0
    assert X_tr_scaled.max() == 1.0
    assert X_te_scaled[0, 0] == pytest.approx(0.5)


def test_memory_windows_dimension():
    """Verify memory window dimension and samples count."""
    csv_path = os.path.join(DATA_PROC_DIR, 'cycle_features.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for w in [3, 5]:
            ds = create_memory_window_dataset(df, window_size=w)
            assert ds['X'].shape[1] == w * 12
            assert len(ds['X']) == len(ds['y'])


def test_dnn_forward_pass():
    """Verify PyTorch BatteryDNN forward pass and parameter counts."""
    model = build_model(input_dim=60)
    dummy_x = torch.randn(16, 60)
    out = model(dummy_x)
    assert out.shape == (16, 1)
    assert model.count_parameters() == 18177


def test_evaluation_metrics():
    """Verify MAE, RMSE, R2, MAPE calculations."""
    y_true = np.array([2.0, 1.8, 1.6, 1.4])
    y_pred = np.array([2.0, 1.8, 1.6, 1.4])
    m = calculate_metrics(y_true, y_pred)
    assert m['mae'] == 0.0
    assert m['rmse'] == 0.0
    assert m['r2'] == 1.0
    assert m['mape'] == 0.0
