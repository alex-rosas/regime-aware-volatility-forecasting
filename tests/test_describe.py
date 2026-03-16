"""
tests/test_describe.py
----------------------
Tests for src/data/describe.py

All tests use synthetic data — no real data files required.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.describe import (
    compute_returns,
    compute_descriptive_stats,
    compute_rolling_volatility,
    run_arch_lm,
    run_jarque_bera,
    run_ljung_box,
    run_adf,
    run_all_tests,
    tests_to_frame,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """Simple price series with known log-returns."""
    np.random.seed(42)
    n      = 500
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))),
        index=pd.date_range('2000-01-01', periods=n, freq='B'),
        name='MXN_USD',
    )
    return prices


@pytest.fixture
def synthetic_returns(synthetic_prices):
    return compute_returns(synthetic_prices)


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

def test_compute_returns_length(synthetic_prices):
    """Returns series should have one fewer observation than prices."""
    returns = compute_returns(synthetic_prices)
    assert len(returns) == len(synthetic_prices) - 1


def test_compute_returns_no_nans(synthetic_prices):
    """Returns series should contain no NaN values."""
    returns = compute_returns(synthetic_prices)
    assert returns.isna().sum() == 0


def test_compute_returns_index_aligned(synthetic_prices):
    """Returns index should start at the second price date."""
    returns = compute_returns(synthetic_prices)
    assert returns.index[0] == synthetic_prices.index[1]


def test_compute_returns_values(synthetic_prices):
    """First return should equal log(price[1] / price[0])."""
    returns  = compute_returns(synthetic_prices)
    expected = np.log(synthetic_prices.iloc[1] / synthetic_prices.iloc[0])
    assert abs(returns.iloc[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# compute_descriptive_stats
# ---------------------------------------------------------------------------

def test_descriptive_stats_keys(synthetic_returns):
    """summary dict should contain all expected keys."""
    stats = compute_descriptive_stats(synthetic_returns)
    expected_keys = {
        'n', 'mean', 'std', 'std_annualised',
        'skewness', 'excess_kurtosis', 'min', 'max', 'nu_moments',
    }
    assert expected_keys.issubset(set(stats.keys()))


def test_descriptive_stats_n(synthetic_returns):
    """n should equal length of returns series."""
    stats = compute_descriptive_stats(synthetic_returns)
    assert stats['n'] == len(synthetic_returns)


def test_descriptive_stats_annualised_vol(synthetic_returns):
    """Annualised vol should equal daily std * sqrt(252)."""
    stats    = compute_descriptive_stats(synthetic_returns)
    expected = stats['std'] * np.sqrt(252)
    assert abs(stats['std_annualised'] - expected) < 1e-10


# ---------------------------------------------------------------------------
# compute_rolling_volatility
# ---------------------------------------------------------------------------

def test_rolling_volatility_length(synthetic_returns):
    """Rolling vol series should have same length as returns."""
    vol = compute_rolling_volatility(synthetic_returns, window=30)
    assert len(vol) == len(synthetic_returns)


def test_rolling_volatility_first_window_nan(synthetic_returns):
    """First (window - 1) values should be NaN."""
    window = 30
    vol    = compute_rolling_volatility(synthetic_returns, window=window)
    assert vol.iloc[:window - 1].isna().all()


def test_rolling_volatility_positive(synthetic_returns):
    """All non-NaN values should be positive."""
    vol = compute_rolling_volatility(synthetic_returns, window=30)
    assert (vol.dropna() > 0).all()


# ---------------------------------------------------------------------------
# run_arch_lm
# ---------------------------------------------------------------------------

def test_arch_lm_keys(synthetic_returns):
    result = run_arch_lm(synthetic_returns)
    assert {'lm_stat', 'lm_pvalue', 'f_stat', 'f_pvalue', 'nlags'}.issubset(
        set(result.keys())
    )


def test_arch_lm_pvalue_range(synthetic_returns):
    result = run_arch_lm(synthetic_returns)
    assert 0.0 <= result['lm_pvalue'] <= 1.0


# ---------------------------------------------------------------------------
# run_jarque_bera
# ---------------------------------------------------------------------------

def test_jarque_bera_keys(synthetic_returns):
    result = run_jarque_bera(synthetic_returns)
    assert {'jb_stat', 'jb_pvalue', 'skewness', 'excess_kurtosis'}.issubset(
        set(result.keys())
    )


def test_jarque_bera_pvalue_range(synthetic_returns):
    result = run_jarque_bera(synthetic_returns)
    assert 0.0 <= result['jb_pvalue'] <= 1.0


def test_jarque_bera_normal_data():
    """Truly normal data should fail to reject normality."""
    np.random.seed(0)
    returns = pd.Series(
        np.random.normal(0, 0.01, 1000),
        index=pd.date_range('2000-01-01', periods=1000, freq='B'),
    )
    result = run_jarque_bera(returns)
    assert result['jb_pvalue'] > 0.05


# ---------------------------------------------------------------------------
# run_ljung_box
# ---------------------------------------------------------------------------

def test_ljung_box_keys(synthetic_returns):
    result = run_ljung_box(synthetic_returns)
    assert {
        'lb_stat_r', 'lb_pvalue_r',
        'lb_stat_r2', 'lb_pvalue_r2', 'lags'
    }.issubset(set(result.keys()))


def test_ljung_box_pvalues_range(synthetic_returns):
    result = run_ljung_box(synthetic_returns)
    assert 0.0 <= result['lb_pvalue_r']  <= 1.0
    assert 0.0 <= result['lb_pvalue_r2'] <= 1.0


# ---------------------------------------------------------------------------
# run_adf
# ---------------------------------------------------------------------------

def test_adf_keys(synthetic_returns):
    result = run_adf(synthetic_returns)
    assert {'adf_stat', 'adf_pvalue', 'cv_1pct', 'cv_5pct', 'cv_10pct'}.issubset(
        set(result.keys())
    )


def test_adf_rejects_unit_root_on_returns(synthetic_returns):
    """Log-returns should be stationary — ADF should reject unit root."""
    result = run_adf(synthetic_returns)
    assert result['adf_pvalue'] < 0.05


# ---------------------------------------------------------------------------
# run_all_tests
# ---------------------------------------------------------------------------

def test_run_all_tests_keys(synthetic_returns):
    results = run_all_tests(synthetic_returns)
    assert set(results.keys()) == {'arch_lm', 'jarque_bera', 'ljung_box', 'adf'}


def test_run_all_tests_nested_dicts(synthetic_returns):
    results = run_all_tests(synthetic_returns)
    for key, val in results.items():
        assert isinstance(val, dict), f'{key} should return a dict'


# ---------------------------------------------------------------------------
# tests_to_frame
# ---------------------------------------------------------------------------

def test_tests_to_frame_shape(synthetic_returns):
    results = run_all_tests(synthetic_returns)
    df      = tests_to_frame(results)
    assert df.shape[0] == 5   # 5 rows: ARCH-LM, JB, LB r, LB r2, ADF


def test_tests_to_frame_columns(synthetic_returns):
    results  = run_all_tests(synthetic_returns)
    df       = tests_to_frame(results)
    expected = {'H0', 'Statistic', 'p-value', 'Reject H0 (5%)',
                'BS Assumption Violated', 'Modeling Response'}
    assert expected.issubset(set(df.columns))


def test_tests_to_frame_no_nans(synthetic_returns):
    results = run_all_tests(synthetic_returns)
    df      = tests_to_frame(results)
    assert not df.isna().any().any()
