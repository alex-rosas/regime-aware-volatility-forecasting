"""
src/data/describe.py
--------------------
Descriptive statistics and statistical tests for return series analysis.

All functions accept a pd.Series of log-returns and return plain dicts
so results are immediately usable in notebooks, MLflow logging, and pytest.

Usage
-----
from src.data.describe import compute_descriptive_stats, run_all_tests

stats = compute_descriptive_stats(returns)
tests = run_all_tests(returns)
"""

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log-returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex.

    Returns
    -------
    pd.Series of log-returns with the same name as prices + '_log_return'.
    First observation is dropped (NaN from shift).
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = f'{prices.name}_log_return' if prices.name else 'log_return'
    return returns


# ---------------------------------------------------------------------------
# descriptive statistics
# ---------------------------------------------------------------------------

def compute_descriptive_stats(returns: pd.Series) -> dict[str, float]:
    """
    Compute descriptive statistics for a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns in decimal units.

    Returns
    -------
    dict with keys:
        n, mean, std, std_annualised, skewness, excess_kurtosis,
        min, max, nu_moments
        where nu_moments is the moment-matching Student-t degrees of freedom
        estimate: nu = 6 / excess_kurtosis + 4  (valid when excess_kurtosis > 0)
    """
    n   = len(returns)
    mu  = float(returns.mean())
    std = float(returns.std())
    s   = float(skew(returns))
    k   = float(kurtosis(returns, fisher=True))  # excess kurtosis (fisher=True)

    # moment-matching Student-t df estimate — valid only when k > 0
    nu_moments = float(6.0 / k + 4.0) if k > 0 else float('nan')

    return {
        'n'              : n,
        'mean'           : mu,
        'std'            : std,
        'std_annualised' : std * np.sqrt(252),
        'skewness'       : s,
        'excess_kurtosis': k,
        'min'            : float(returns.min()),
        'max'            : float(returns.max()),
        'nu_moments'     : nu_moments,
    }


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 90,
    annualise: bool = True,
) -> pd.Series:
    """
    Compute rolling standard deviation of returns.

    Parameters
    ----------
    returns   : pd.Series — daily log-returns in decimal units
    window    : int       — rolling window in trading days (default 90)
    annualise : bool      — if True, multiply by sqrt(252) (default True)

    Returns
    -------
    pd.Series of rolling volatility with the same DatetimeIndex as returns.
    """
    vol = returns.rolling(window).std()
    if annualise:
        vol = vol * np.sqrt(252)
        vol.name = f'rolling_vol_{window}d_ann'
    else:
        vol.name = f'rolling_vol_{window}d'
    return vol


# ---------------------------------------------------------------------------
# statistical tests
# ---------------------------------------------------------------------------

def run_arch_lm(
    returns: pd.Series,
    nlags: int = 10,
) -> dict[str, float]:
    """
    ARCH-LM test for conditional heteroskedasticity.

    Tests H0: no ARCH effects (constant variance).
    A rejection (small p-value) confirms volatility clustering.

    Parameters
    ----------
    returns : pd.Series — daily log-returns
    nlags   : int       — number of lags (default 10)

    Returns
    -------
    dict with keys: lm_stat, lm_pvalue, f_stat, f_pvalue, nlags
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=nlags)
    return {
        'lm_stat'  : float(lm_stat),
        'lm_pvalue': float(lm_pvalue),
        'f_stat'   : float(f_stat),
        'f_pvalue' : float(f_pvalue),
        'nlags'    : nlags,
    }


def run_jarque_bera(returns: pd.Series) -> dict[str, float]:
    """
    Jarque-Bera test for normality.

    Tests H0: returns are normally distributed (skewness=0, excess kurtosis=0).
    A rejection confirms non-normality.

    Parameters
    ----------
    returns : pd.Series — daily log-returns

    Returns
    -------
    dict with keys: jb_stat, jb_pvalue, skewness, excess_kurtosis
    """
    jb_stat, jb_pvalue = jarque_bera(returns)
    return {
        'jb_stat'        : float(jb_stat),
        'jb_pvalue'      : float(jb_pvalue),
        'skewness'       : float(skew(returns)),
        'excess_kurtosis': float(kurtosis(returns, fisher=True)),
    }


def run_ljung_box(
    returns: pd.Series,
    lags: int = 10,
) -> dict[str, float]:
    """
    Ljung-Box test for autocorrelation in returns and squared returns.

    Tests H0: no autocorrelation up to lag k.
    - On r_t:   rejection indicates return predictability
    - On r_t^2: rejection confirms volatility clustering (GARCH signature)

    Parameters
    ----------
    returns : pd.Series — daily log-returns
    lags    : int       — maximum lag to test (default 10, reports Q(lags))

    Returns
    -------
    dict with keys:
        lb_stat_r, lb_pvalue_r       — Q(lags) on demeaned returns
        lb_stat_r2, lb_pvalue_r2     — Q(lags) on squared demeaned returns
        lags
    """
    r_demeaned = returns - returns.mean()

    lb_r  = acorr_ljungbox(r_demeaned,    lags=lags, return_df=True)
    lb_r2 = acorr_ljungbox(r_demeaned**2, lags=lags, return_df=True)

    return {
        'lb_stat_r'   : float(lb_r['lb_stat'].iloc[-1]),
        'lb_pvalue_r' : float(lb_r['lb_pvalue'].iloc[-1]),
        'lb_stat_r2'  : float(lb_r2['lb_stat'].iloc[-1]),
        'lb_pvalue_r2': float(lb_r2['lb_pvalue'].iloc[-1]),
        'lags'        : lags,
    }


def run_adf(returns: pd.Series) -> dict[str, float]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests H0: unit root present (non-stationary).
    A rejection confirms stationarity — a prerequisite for GARCH estimation,
    not a Black-Scholes violation.

    Parameters
    ----------
    returns : pd.Series — daily log-returns

    Returns
    -------
    dict with keys: adf_stat, adf_pvalue, cv_1pct, cv_5pct, cv_10pct
    """
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(returns)
    return {
        'adf_stat'  : float(adf_stat),
        'adf_pvalue': float(adf_pvalue),
        'cv_1pct'   : float(critical_values['1%']),
        'cv_5pct'   : float(critical_values['5%']),
        'cv_10pct'  : float(critical_values['10%']),
    }


# ---------------------------------------------------------------------------
# convenience runner
# ---------------------------------------------------------------------------

def run_all_tests(
    returns: pd.Series,
    nlags: int = 10,
) -> dict[str, dict]:
    """
    Run all four statistical tests and return results in a single dict.

    Parameters
    ----------
    returns : pd.Series — daily log-returns
    nlags   : int       — lags for ARCH-LM and Ljung-Box (default 10)

    Returns
    -------
    dict with keys: arch_lm, jarque_bera, ljung_box, adf
    Each value is the dict returned by the corresponding run_* function.

    Example
    -------
    tests = run_all_tests(returns)
    print(tests['arch_lm']['lm_pvalue'])
    print(tests['jarque_bera']['excess_kurtosis'])
    """
    return {
        'arch_lm'    : run_arch_lm(returns, nlags=nlags),
        'jarque_bera': run_jarque_bera(returns),
        'ljung_box'  : run_ljung_box(returns, lags=nlags),
        'adf'        : run_adf(returns),
    }
