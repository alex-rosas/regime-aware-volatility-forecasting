"""
tests/test_pipeline.py
----------------------
Integration test for src/pipeline.py

Runs the full pipeline on real data and verifies all artifacts
are produced correctly. This test takes ~5 seconds to run.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@pytest.mark.integration
def test_pipeline_produces_all_artifacts():
    """Full pipeline should produce all expected artifacts in data/processed/."""
    from src.pipeline import run_full_pipeline
    run_full_pipeline()

    expected_files = [
        'garch.pkl',
        'egarch.pkl',
        'hmm.pkl',
        'hybrid_model.pkl',
        'conformal.pkl',
        'hmm_regimes.csv',
        'garch_egarch_volatilities.csv',
        'hybrid_predictions.csv',
        'hybrid_val_predictions.csv',
        'conformal_intervals.csv',
    ]

    processed = ROOT / 'data' / 'processed'
    for fname in expected_files:
        assert (processed / fname).exists(), f'Missing artifact: {fname}'


@pytest.mark.integration
def test_pipeline_hmm_regimes_values():
    """Regime values should be in {0, 1, 2} with no NaNs."""
    regimes = pd.read_csv(
        ROOT / 'data/processed/hmm_regimes.csv',
        index_col='Date', parse_dates=True
    )
    assert regimes['regime'].isna().sum() == 0
    assert set(regimes['regime'].unique()).issubset({0, 1, 2})

@pytest.mark.integration
def test_pipeline_volatilities_positive():
    """All conditional volatility values should be positive."""
    vols = pd.read_csv(
        ROOT / 'data/processed/garch_egarch_volatilities.csv',
        index_col='Date', parse_dates=True
    )
    assert (vols > 0).all().all()

@pytest.mark.integration
def test_pipeline_hybrid_predictions_shape():
    """Predictions CSV should have 4 columns and no NaNs."""
    preds = pd.read_csv(
        ROOT / 'data/processed/hybrid_predictions.csv',
        index_col='Date', parse_dates=True
    )
    assert preds.shape[1] == 4
    assert preds.isna().sum().sum() == 0

@pytest.mark.integration
def test_pipeline_conformal_intervals_bounds():
    """Upper bounds should always exceed lower bounds."""
    intervals = pd.read_csv(
        ROOT / 'data/processed/conformal_intervals.csv',
        index_col='Date', parse_dates=True
    )
    for level in ['80', '90', '95']:
        assert (intervals[f'upper_{level}'] > intervals[f'lower_{level}']).all()

@pytest.mark.integration
def test_pipeline_conformal_coverage_90():
    """90% conformal interval should achieve at least 85% empirical coverage."""
    intervals = pd.read_csv(
        ROOT / 'data/processed/conformal_intervals.csv',
        index_col='Date', parse_dates=True
    )
    y_true = intervals['y_true'].values
    upper  = intervals['upper_90'].values
    lower  = intervals['lower_90'].values
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    assert coverage >= 0.85, f'Coverage too low: {coverage:.2%}'
