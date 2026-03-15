"""
src/pipeline.py
---------------
Full end-to-end pipeline for the VolatilityRegimes project.

Runs all modeling steps in sequence from raw data to conformal
prediction intervals. Designed to be run as a script or imported
and called as run_full_pipeline().

Usage
-----
    # from terminal
    python src/pipeline.py

    # from Python
    from src.pipeline import run_full_pipeline
    run_full_pipeline()

Steps
-----
    1. Load and validate raw data
    2. Compute returns
    3. Fit GARCH(1,1)-t and EGARCH(1,1)-t
    4. Fit 3-state HMM
    5. Export regime and volatility features
    6. Build feature matrix and fit hybrid XGBoost model
    7. Calibrate conformal predictor
    8. Export all artifacts

All artifacts are saved to data/processed/.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.describe import compute_returns
from src.models.conformal import ConformalPredictor
from src.models.garch import VolatilityModel
from src.models.hmm import RegimeHMM
from src.models.hybrid import HybridVolatilityModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log(step: str, msg: str) -> None:
    print(f'[{step}] {msg}')


def _elapsed(t0: float) -> str:
    return f'{time.time() - t0:.1f}s'


# ---------------------------------------------------------------------------
# pipeline steps
# ---------------------------------------------------------------------------

def step_load_data() -> tuple[pd.Series, pd.DataFrame]:
    """
    Load raw MXN/USD prices and macro series from data/raw/.

    Returns
    -------
    prices : pd.Series  — MXN/USD price series
    macro  : pd.DataFrame — VIXCLS and T10Y2Y aligned to business days
    """
    t0 = time.time()
    _log('1/7', 'Loading raw data...')

    prices = pd.read_csv(
        ROOT / 'data/raw/mxn_usd.csv',
        index_col='Date', parse_dates=True
    )['MXN_USD']

    macro = pd.read_csv(
        ROOT / 'data/raw/macro.csv',
        index_col=0, parse_dates=True
    )
    macro.index.name = 'Date'
    macro = macro[['VIXCLS', 'T10Y2Y']]

    _log('1/7', f'Done — prices: {len(prices):,} rows, '
                f'macro: {len(macro):,} rows  ({_elapsed(t0)})')
    return prices, macro


def step_compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log-returns from price series.

    Returns
    -------
    pd.Series of log-returns with DatetimeIndex.
    """
    t0 = time.time()
    _log('2/7', 'Computing log-returns...')

    returns = compute_returns(prices)

    _log('2/7', f'Done — {len(returns):,} observations  '
                f'({returns.index[0].date()} → {returns.index[-1].date()})  '
                f'({_elapsed(t0)})')
    return returns


def step_fit_garch(returns: pd.Series) -> tuple[VolatilityModel, VolatilityModel]:
    """
    Fit GARCH(1,1)-t and EGARCH(1,1)-t models.

    Saves both models to data/processed/ as pickle files.

    Returns
    -------
    garch, egarch : fitted VolatilityModel instances
    """
    t0 = time.time()
    _log('3/7', 'Fitting GARCH(1,1)-t and EGARCH(1,1)-t...')

    garch  = VolatilityModel('GARCH').fit(returns)
    egarch = VolatilityModel('EGARCH').fit(returns)

    garch.save(ROOT  / 'data/processed/garch.pkl')
    egarch.save(ROOT / 'data/processed/egarch.pkl')

    gs = garch.summary()
    es = egarch.summary()
    _log('3/7', f'GARCH  — persistence={gs["persistence"]:.4f}, '
                f'nu={gs["nu"]:.2f}, BIC={gs["bic"]:.2f}')
    _log('3/7', f'EGARCH — persistence={es["persistence"]:.4f}, '
                f'gamma={es["gamma"]:.4f}, BIC={es["bic"]:.2f}  ({_elapsed(t0)})')

    return garch, egarch


def step_fit_hmm(returns: pd.Series) -> RegimeHMM:
    """
    Fit 3-state Gaussian HMM on percentage returns.

    Exports regime series and volatility CSVs to data/processed/.
    Saves fitted model as hmm.pkl.

    Returns
    -------
    hmm : fitted RegimeHMM instance
    """
    t0 = time.time()
    _log('4/7', 'Fitting 3-state HMM...')

    hmm = RegimeHMM(n_components=3, n_iter=1000, random_state=42)
    hmm.fit(returns)

    s = hmm.summary()
    _log('4/7', f'Converged={s["converged"]}, '
                f'BIC={s["bic"]:.2f}, '
                f'iterations={s["n_iter_run"]}')

    regimes = hmm.predict(returns)
    _log('4/7', f'Regime counts: {regimes.value_counts().sort_index().to_dict()}')

    hmm.save(ROOT / 'data/processed/hmm.pkl')

    return hmm


def step_export_features(
    returns: pd.Series,
    garch:   VolatilityModel,
    egarch:  VolatilityModel,
    hmm:     RegimeHMM,
) -> None:
    """
    Export regime series and volatility CSVs for the hybrid model.

    Saves:
        data/processed/hmm_regimes.csv
        data/processed/garch_egarch_volatilities.csv
    """
    t0 = time.time()
    _log('5/7', 'Exporting features...')

    regimes = hmm.predict(returns)
    regimes.to_frame('regime').to_csv(ROOT / 'data/processed/hmm_regimes.csv')

    garch_vol  = garch.conditional_volatility()
    egarch_vol = egarch.conditional_volatility()
    vols       = pd.concat([garch_vol, egarch_vol], axis=1).dropna()
    vols.to_csv(ROOT / 'data/processed/garch_egarch_volatilities.csv')

    _log('5/7', f'hmm_regimes.csv: {len(regimes):,} rows | '
                f'volatilities.csv: {len(vols):,} rows  ({_elapsed(t0)})')


def step_fit_hybrid(
    returns: pd.Series,
    macro:   pd.DataFrame,
) -> tuple[HybridVolatilityModel, pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix and fit hybrid XGBoost model.

    Saves:
        data/processed/hybrid_model.pkl
        data/processed/hybrid_predictions.csv
        data/processed/hybrid_val_predictions.csv

    Returns
    -------
    hybrid   : fitted HybridVolatilityModel
    X_val    : validation feature matrix
    y_val    : validation target
    """
    t0 = time.time()
    _log('6/7', 'Fitting hybrid XGBoost model...')

    # load processed features
    regimes = pd.read_csv(
        ROOT / 'data/processed/hmm_regimes.csv',
        index_col='Date', parse_dates=True
    )
    vols = pd.read_csv(
        ROOT / 'data/processed/garch_egarch_volatilities.csv',
        index_col='Date', parse_dates=True
    )

    # align macro to returns index
    macro_aligned = macro.reindex(returns.index).ffill()

    # build aligned dataset
    data = pd.concat([returns, macro_aligned, regimes, vols], axis=1).dropna()

    hybrid       = HybridVolatilityModel()
    X, y         = hybrid.build_features(data)
    X_train, y_train, X_val, y_val, X_test, y_test = HybridVolatilityModel.split(X, y)

    hybrid.fit(X_train, y_train, X_val, y_val)

    s = hybrid.summary()
    _log('6/7', f'Best iteration={s["best_iteration"]}, '
                f'train RMSE={s["train_rmse"]:.6f}, '
                f'val RMSE={s["val_rmse"]:.6f}')

    # save model
    hybrid.save(ROOT / 'data/processed/hybrid_model.pkl')

    # save test predictions
    y_pred_hybrid = hybrid.predict(X_test)
    y_pred_garch  = (X_test['sigma_garch_ann']  / np.sqrt(252)).values
    y_pred_egarch = (X_test['sigma_egarch_ann'] / np.sqrt(252)).values

    pd.DataFrame({
        'y_true'        : y_test.values,
        'y_pred_hybrid' : y_pred_hybrid,
        'y_pred_garch'  : y_pred_garch,
        'y_pred_egarch' : y_pred_egarch,
    }, index=y_test.index).to_csv(ROOT / 'data/processed/hybrid_predictions.csv')

    # save validation predictions for conformal calibration
    pd.DataFrame({
        'y_true'        : y_val.values,
        'y_pred_hybrid' : hybrid.predict(X_val),
    }, index=y_val.index).to_csv(ROOT / 'data/processed/hybrid_val_predictions.csv')

    _log('6/7', f'Predictions saved  ({_elapsed(t0)})')

    return hybrid, X_val, y_val


def step_fit_conformal(
    hybrid: HybridVolatilityModel,
) -> ConformalPredictor:
    """
    Calibrate conformal predictor on validation residuals.

    Saves:
        data/processed/conformal.pkl
        data/processed/conformal_intervals.csv

    Returns
    -------
    cp : fitted ConformalPredictor
    """
    t0 = time.time()
    _log('7/7', 'Calibrating conformal predictor...')

    val_data  = pd.read_csv(
        ROOT / 'data/processed/hybrid_val_predictions.csv',
        index_col='Date', parse_dates=True
    )
    test_data = pd.read_csv(
        ROOT / 'data/processed/hybrid_predictions.csv',
        index_col='Date', parse_dates=True
    )

    cp = ConformalPredictor()
    cp.calibrate(
        y_true = val_data['y_true'].values,
        y_pred = val_data['y_pred_hybrid'].values,
    )

    y_pred_test = test_data['y_pred_hybrid'].values
    y_true_test = test_data['y_true'].values
    results     = cp.predict_all(y_pred_test)

    pd.DataFrame({
        'y_true'   : y_true_test,
        'y_pred'   : y_pred_test,
        'lower_80' : results['0.2']['lower'],
        'upper_80' : results['0.2']['upper'],
        'lower_90' : results['0.1']['lower'],
        'upper_90' : results['0.1']['upper'],
        'lower_95' : results['0.05']['lower'],
        'upper_95' : results['0.05']['upper'],
    }, index=test_data.index).to_csv(
        ROOT / 'data/processed/conformal_intervals.csv'
    )

    cp.save(ROOT / 'data/processed/conformal.pkl')

    for alpha in [0.20, 0.10, 0.05]:
        cov = cp.coverage(y_true_test, y_pred_test, alpha)
        _log('7/7', f'Coverage {1-alpha:.0%}: {cov:.2%}')

    _log('7/7', f'Done  ({_elapsed(t0)})')
    return cp


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_full_pipeline() -> None:
    """
    Run the full VolatilityRegimes pipeline end-to-end.

    Steps:
        1. Load raw data
        2. Compute returns
        3. Fit GARCH + EGARCH
        4. Fit HMM
        5. Export features
        6. Fit hybrid model
        7. Calibrate conformal predictor
    """
    t_total = time.time()
    print()
    print('=' * 60)
    print('VolatilityRegimes — Full Pipeline')
    print('=' * 60)
    print()

    prices, macro           = step_load_data()
    returns                 = step_compute_returns(prices)
    garch, egarch           = step_fit_garch(returns)
    hmm                     = step_fit_hmm(returns)
    step_export_features(returns, garch, egarch, hmm)
    hybrid, X_val, y_val    = step_fit_hybrid(returns, macro)
    cp                      = step_fit_conformal(hybrid)

    print()
    print('=' * 60)
    print(f'Pipeline complete — total time: {_elapsed(t_total)}')
    print('Artifacts saved to data/processed/:')
    for f in sorted((ROOT / 'data/processed').glob('*.pkl')):
        print(f'  {f.name}')
    for f in sorted((ROOT / 'data/processed').glob('*.csv')):
        print(f'  {f.name}')
    print('=' * 60)
    print()


if __name__ == '__main__':
    run_full_pipeline()