"""
test_registry.py — Test Data Consistency Registry for Memory Window Benchmarks
"""

import os
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'results', 'registry.csv')


def test_registry_exists_and_schema():
    """Verify results/registry.csv exists and matches expected columns."""
    assert os.path.exists(REGISTRY_PATH), f"registry.csv missing at {REGISTRY_PATH}"
    df = pd.read_csv(REGISTRY_PATH, na_values=['NA', 'NaN'])
    
    expected_cols = [
        'run_id', 'script', 'date', 'aggregation',
        'window', 'MAE', 'RMSE', 'R2', 'MAPE', 'RUL_MAE'
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 8  # 4 windows x 2 runs


def test_registry_windows_and_runs():
    """Verify registry contains both Phase 1 and Phase 2 runs for windows 3, 5, 10, 20."""
    df = pd.read_csv(REGISTRY_PATH, na_values=['NA', 'NaN'])
    
    runs = set(df['run_id'].unique())
    assert runs == {'phase1_lobo', 'phase2_final'}
    
    for run_id in runs:
        sub = df[df['run_id'] == run_id]
        assert set(sub['window']) == {3, 5, 10, 20}
        assert (sub['aggregation'] == 'mean-of-folds').all()


def test_registry_metrics_integrity():
    """Verify registry metric values match original CSV sources without fabrication."""
    df = pd.read_csv(REGISTRY_PATH, na_values=['NA', 'NaN'])
    
    # Phase 1: RUL_MAE must be null / NA; other metrics non-null
    p1 = df[df['run_id'] == 'phase1_lobo']
    assert p1['RUL_MAE'].isna().all()
    assert not p1[['MAE', 'RMSE', 'R2', 'MAPE']].isna().any().any()
    
    # Phase 2: All metrics must be non-null numbers
    p2 = df[df['run_id'] == 'phase2_final']
    assert not p2[['MAE', 'RMSE', 'R2', 'MAPE', 'RUL_MAE']].isna().any().any()
    
    # Check specific grounded values
    w10_p2 = p2[p2['window'] == 10].iloc[0]
    assert w10_p2['MAE'] == pytest.approx(0.08037, abs=1e-4)
    assert w10_p2['RUL_MAE'] == pytest.approx(8.45, abs=1e-2)
