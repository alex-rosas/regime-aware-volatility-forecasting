"""
src/pipeline.py
---------------
Full end-to-end pipeline for the VolatilityRegimes project.

Runs all modeling steps in sequence from raw data to conformal
prediction intervals in a single function call.

When to use this script
-----------------------
Use this script for:
    - Quick local runs during development and experimentation
    - Interactive use from notebooks or a REPL
    - Debugging the full pipeline flow end-to-end
    - Sanity checks after making changes to model logic

Do NOT use this script for:
    - Reproducible experiment tracking    → use: dvc repro
    - Partial re-runs after param changes → use: dvc repro
    - CI/CD or production runs            → use: dvc repro

In production and experiment contexts, DVC manages the pipeline
via dvc.yaml, calling individual stage scripts under stages/.
Those stage scripts reuse the same step functions defined here,
so no logic is duplicated.

Usage
-----
    # from terminal
    python src/pipeline.py

    # from Python
    from src.pipeline import run_pipeline_end_to_end
    run_pipeline_end_to_end()

Steps
-----
    1. Load raw data
    2. Compute log-returns
    3. Fit GARCH(1,1)-t and EGARCH(1,1)-t
    4. Fit 3-state HMM
    5. Export regime and volatility features
    6. Build feature matrix and fit hybrid XGBoost model
    7. Calibrate conformal predictor

All artifacts are saved to data/processed/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.describe import compute_returns
from src.logger import get_logger, setup_logging
from src.models.conformal import ConformalPredictor
from src.models.garch import VolatilityModel
from src.models.hmm import RegimeHMM
from src.models.hybrid import HybridVolatilityModel

logger = get_logger(__name__)


def _load_params() -> dict:
    """Load pipeline parameters from params.yaml at project root."""
    params_path = ROOT / 'params.yaml'
    with open(params_path) as f:
        return yaml.safe_load(f)


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
    logger.info('[1/7] Loading raw data...')

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

    prices.to_csv(ROOT / 'data/processed/prices.csv')
    macro.to_csv(ROOT / 'data/processed/macro.csv')

    logger.info(f'[1/7] Done — prices: {len(prices):,} rows, '
                f'macro: {len(macro):,} rows  ({time.time() - t0:.1f}s)')
    return prices, macro


def step_compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log-returns from price series.

    Returns
    -------
    pd.Series of log-returns with DatetimeIndex.
    """
    t0 = time.time()
    logger.info('[2/7] Computing log-returns...')

    returns = compute_returns(prices)

    returns.to_csv(ROOT / 'data/processed/returns.csv')

    logger.info(f'[2/7] Done — {len(returns):,} observations  '
                f'({returns.index[0].date()} → {returns.index[-1].date()})  '
                f'({time.time() - t0:.1f}s)')
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
    logger.info('[3/7] Fitting GARCH(1,1)-t and EGARCH(1,1)-t...')

    garch  = VolatilityModel('GARCH').fit(returns)
    egarch = VolatilityModel('EGARCH').fit(returns)

    garch.save(ROOT  / 'data/processed/garch.pkl')
    egarch.save(ROOT / 'data/processed/egarch.pkl')

    gs = garch.summary()
    es = egarch.summary()
    logger.info(f'[3/7] GARCH  — persistence={gs["persistence"]:.4f}, '
                f'nu={gs["nu"]:.2f}, BIC={gs["bic"]:.2f}')
    logger.info(f'[3/7] EGARCH — persistence={es["persistence"]:.4f}, '
                f'gamma={es["gamma"]:.4f}, BIC={es["bic"]:.2f}  ({time.time() - t0:.1f}s)')

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
    params = _load_params()
    hmm_cfg = params['hmm']
    logger.info(f'[4/7] Fitting {hmm_cfg["n_components"]}-state HMM...')

    hmm = RegimeHMM(
        n_components=hmm_cfg['n_components'],
        n_iter=hmm_cfg['n_iter'],
        random_state=hmm_cfg['random_state'],
    )
    hmm.fit(returns)

    s = hmm.summary()
    logger.info(f'[4/7] Converged={s["converged"]}, '
                f'BIC={s["bic"]:.2f}, '
                f'iterations={s["n_iter_run"]}')

    regimes = hmm.predict(returns)
    logger.info(f'[4/7] Regime counts: {regimes.value_counts().sort_index().to_dict()}')

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
    logger.info('[5/7] Exporting features...')

    regimes = hmm.predict(returns)
    regimes.to_frame('regime').to_csv(ROOT / 'data/processed/hmm_regimes.csv')

    garch_vol  = garch.conditional_volatility()
    egarch_vol = egarch.conditional_volatility()
    vols       = pd.concat([garch_vol, egarch_vol], axis=1).dropna()
    vols.to_csv(ROOT / 'data/processed/garch_egarch_volatilities.csv')

    logger.info(f'[5/7] hmm_regimes.csv: {len(regimes):,} rows | '
                f'volatilities.csv: {len(vols):,} rows  ({time.time() - t0:.1f}s)')


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
    params = _load_params()
    trading_days = params['data']['trading_days']
    logger.info('[6/7] Fitting hybrid XGBoost model...')

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

    xgb_cfg = params['xgboost']
    hybrid   = HybridVolatilityModel(
        n_estimators=xgb_cfg['n_estimators'],
        learning_rate=xgb_cfg['learning_rate'],
        max_depth=xgb_cfg['max_depth'],
        subsample=xgb_cfg['subsample'],
        colsample_bytree=xgb_cfg['colsample_bytree'],
        early_stopping_rounds=xgb_cfg['early_stopping_rounds'],
        random_state=xgb_cfg['random_state'],
    )
    X, y         = hybrid.build_features(data)
    X_train, y_train, X_val, y_val, X_test, y_test = HybridVolatilityModel.split(X, y)

    hybrid.fit(X_train, y_train, X_val, y_val)

    s = hybrid.summary()
    logger.info(f'[6/7] Best iteration={s["best_iteration"]}, '
                f'train RMSE={s["train_rmse"]:.6f}, '
                f'val RMSE={s["val_rmse"]:.6f}')

    # save model
    hybrid.save(ROOT / 'data/processed/hybrid_model.pkl')

    # save test predictions
    y_pred_hybrid = hybrid.predict(X_test)
    y_pred_garch  = (X_test['sigma_garch_ann']  / np.sqrt(trading_days)).values
    y_pred_egarch = (X_test['sigma_egarch_ann'] / np.sqrt(trading_days)).values

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

    logger.info(f'[6/7] Predictions saved  ({time.time() - t0:.1f}s)')

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
    params = _load_params()
    alpha_levels = params['conformal']['alpha_levels']
    logger.info('[7/7] Calibrating conformal predictor...')

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

    conf_cols = {'y_true': y_true_test, 'y_pred': y_pred_test}
    for alpha in alpha_levels:
        key = str(round(alpha, 2))
        coverage = int((1 - alpha) * 100)
        conf_cols[f'lower_{coverage}'] = results[key]['lower']
        conf_cols[f'upper_{coverage}'] = results[key]['upper']

    pd.DataFrame(conf_cols, index=test_data.index).to_csv(
        ROOT / 'data/processed/conformal_intervals.csv'
    )

    cp.save(ROOT / 'data/processed/conformal.pkl')

    metrics = {}
    for alpha in alpha_levels:
        cov     = cp.coverage(y_true_test, y_pred_test, alpha)
        key     = f'coverage_{int((1 - alpha) * 100)}'
        metrics[key] = round(cov, 4)
        logger.info(f'[7/7] Coverage {1 - alpha:.0%}: {cov:.2%}')

    with open(ROOT / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'[7/7] Metrics saved → metrics.json')

    logger.info(f'[7/7] Done  ({time.time() - t0:.1f}s)')
    return cp


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_pipeline_end_to_end() -> None:
    """
    Run the full VolatilityRegimes pipeline in a single shot.

    Intended for development, interactive use, and quick local runs.
    For reproducible experiment tracking and partial re-runs use dvc repro.

    Steps:
        1. Load raw data
        2. Compute returns
        3. Fit GARCH + EGARCH
        4. Fit HMM
        5. Export features
        6. Fit hybrid model
        7. Calibrate conformal predictor
    """
    setup_logging()
    t_total = time.time()
    logger.info('=' * 60)
    logger.info('VolatilityRegimes — Full Pipeline')
    logger.info('=' * 60)

    prices, macro           = step_load_data()
    returns                 = step_compute_returns(prices)
    garch, egarch           = step_fit_garch(returns)
    hmm                     = step_fit_hmm(returns)
    step_export_features(returns, garch, egarch, hmm)
    hybrid, X_val, y_val    = step_fit_hybrid(returns, macro)
    cp                      = step_fit_conformal(hybrid)

    logger.info('=' * 60)
    logger.info(f'Pipeline complete — total time: {time.time() - t_total:.1f}s')
    logger.info('Artifacts saved to data/processed/:')
    for f in sorted((ROOT / 'data/processed').glob('*.pkl')):
        logger.info(f'  {f.name}')
    for f in sorted((ROOT / 'data/processed').glob('*.csv')):
        logger.info(f'  {f.name}')
    logger.info('=' * 60)


if __name__ == '__main__':
    run_pipeline_end_to_end()