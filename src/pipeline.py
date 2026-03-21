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
from src.models.conformal import AsymmetricConformalPredictor, ConformalPredictor
from src.models.garch import VolatilityModel
from src.models.hmm import RegimeHMM
from src.models.hybrid import HybridVolatilityModel, diebold_mariano, qlike
from src.utils import atomic_write

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

    with atomic_write(ROOT / 'data/processed/prices.csv') as tmp:
        prices.to_csv(tmp)
    with atomic_write(ROOT / 'data/processed/macro.csv') as tmp:
        macro.to_csv(tmp)

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

    with atomic_write(ROOT / 'data/processed/returns.csv') as tmp:
        returns.to_csv(tmp)

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

    with atomic_write(ROOT / 'data/processed/garch.pkl') as tmp:
        garch.save(tmp)
    with atomic_write(ROOT / 'data/processed/egarch.pkl') as tmp:
        egarch.save(tmp)

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

    with atomic_write(ROOT / 'data/processed/hmm.pkl') as tmp:
        hmm.save(tmp)

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
    with atomic_write(ROOT / 'data/processed/hmm_regimes.csv') as tmp:
        regimes.to_frame('regime').to_csv(tmp)

    garch_vol  = garch.conditional_volatility()
    egarch_vol = egarch.conditional_volatility()
    vols       = pd.concat([garch_vol, egarch_vol], axis=1).dropna()
    with atomic_write(ROOT / 'data/processed/garch_egarch_volatilities.csv') as tmp:
        vols.to_csv(tmp)

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
    with atomic_write(ROOT / 'data/processed/hybrid_model.pkl') as tmp:
        hybrid.save(tmp)

    # save test predictions
    y_pred_hybrid = hybrid.predict(X_test)
    y_pred_garch  = (X_test['sigma_garch_ann']  / np.sqrt(trading_days)).values
    y_pred_egarch = (X_test['sigma_egarch_ann'] / np.sqrt(trading_days)).values

    with atomic_write(ROOT / 'data/processed/hybrid_predictions.csv') as tmp:
        pd.DataFrame({
            'y_true'        : y_test.values,
            'y_pred_hybrid' : y_pred_hybrid,
            'y_pred_garch'  : y_pred_garch,
            'y_pred_egarch' : y_pred_egarch,
        }, index=y_test.index).to_csv(tmp)

    # save validation predictions for conformal calibration
    with atomic_write(ROOT / 'data/processed/hybrid_val_predictions.csv') as tmp:
        pd.DataFrame({
            'y_true'        : y_val.values,
            'y_pred_hybrid' : hybrid.predict(X_val),
        }, index=y_val.index).to_csv(tmp)

    # save SHAP feature importances (mean |SHAP| on test set)
    import shap as _shap
    shap_vals = hybrid.shap_values(X_test)
    shap_importance = pd.DataFrame({
        'feature'       : hybrid.feature_names_,
        'mean_abs_shap' : np.abs(shap_vals.values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    with atomic_write(ROOT / 'data/processed/shap_importance.csv') as tmp:
        shap_importance.to_csv(tmp, index=False)
    logger.info(f'[6/7] Top feature: {shap_importance.iloc[0]["feature"]} '
                f'(mean |SHAP|={shap_importance.iloc[0]["mean_abs_shap"]:.6f})')

    # --- metrics -----------------------------------------------------------
    y_true = y_test.values

    rmse_hybrid  = float(np.sqrt(np.mean((y_true - y_pred_hybrid)  ** 2)))
    rmse_garch   = float(np.sqrt(np.mean((y_true - y_pred_garch)   ** 2)))
    rmse_egarch  = float(np.sqrt(np.mean((y_true - y_pred_egarch)  ** 2)))
    rmse_improvement_pct = round((rmse_garch - rmse_hybrid) / rmse_garch * 100, 2)

    qlike_hybrid = qlike(y_true, y_pred_hybrid)
    qlike_garch  = qlike(y_true, y_pred_garch)
    qlike_egarch = qlike(y_true, y_pred_egarch)

    dm_stat_vs_garch,  dm_pval_vs_garch  = diebold_mariano(y_true, y_pred_hybrid, y_pred_garch)
    dm_stat_vs_egarch, dm_pval_vs_egarch = diebold_mariano(y_true, y_pred_hybrid, y_pred_egarch)

    metrics = {
        'rmse_hybrid'           : round(rmse_hybrid,  6),
        'rmse_garch'            : round(rmse_garch,   6),
        'rmse_egarch'           : round(rmse_egarch,  6),
        'rmse_improvement_pct'  : rmse_improvement_pct,
        'qlike_hybrid'          : round(qlike_hybrid, 6),
        'qlike_garch'           : round(qlike_garch,  6),
        'qlike_egarch'          : round(qlike_egarch, 6),
        'dm_stat_vs_garch'      : round(dm_stat_vs_garch,  4),
        'dm_pval_vs_garch'      : round(dm_pval_vs_garch,  4),
        'dm_stat_vs_egarch'     : round(dm_stat_vs_egarch, 4),
        'dm_pval_vs_egarch'     : round(dm_pval_vs_egarch, 4),
    }

    with atomic_write(ROOT / 'metrics.json') as tmp:
        with open(tmp, 'w') as f:
            json.dump(metrics, f, indent=2)
    logger.info(
        f'[6/7] RMSE hybrid={rmse_hybrid:.4f} | '
        f'GARCH={rmse_garch:.4f} | '
        f'improvement={rmse_improvement_pct:+.1f}%  '
        f'({time.time() - t0:.1f}s)'
    )

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

    asymmetry = params['conformal']['asymmetry']
    cp = AsymmetricConformalPredictor(asymmetry=asymmetry)
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

    with atomic_write(ROOT / 'data/processed/conformal_intervals.csv') as tmp:
        pd.DataFrame(conf_cols, index=test_data.index).to_csv(tmp)

    with atomic_write(ROOT / 'data/processed/conformal.pkl') as tmp:
        cp.save(tmp)

    # read existing metrics written by step_fit_hybrid and append coverage
    metrics_path = ROOT / 'metrics.json'
    with open(metrics_path) as f:
        metrics = json.load(f)

    for alpha in alpha_levels:
        cov = cp.coverage(y_true_test, y_pred_test, alpha)
        key = f'coverage_{int((1 - alpha) * 100)}'
        metrics[key] = round(cov, 4)
        logger.info(f'[7/7] Coverage {1 - alpha:.0%}: {cov:.2%}')

    with atomic_write(metrics_path) as tmp:
        with open(tmp, 'w') as f:
            json.dump(metrics, f, indent=2)
    logger.info(f'[7/7] Metrics saved → metrics.json')

    logger.info(f'[7/7] Done  ({time.time() - t0:.1f}s)')
    return cp


# ---------------------------------------------------------------------------
# walk-forward validation
# ---------------------------------------------------------------------------

def step_walk_forward(
    returns: pd.Series,
    macro:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation of the XGBoost hybrid model.

    Refits XGBoost on each expanding fold using precomputed HMM and GARCH
    features. GARCH and HMM are NOT refit per fold — they are structural
    models whose outputs are stable inputs to the feature matrix.

    Saves:
        data/processed/walkforward_predictions.csv

    Updates:
        metrics.json  — appends wf_* metrics

    Returns
    -------
    pd.DataFrame of out-of-sample predictions across all folds.
    """
    t0 = time.time()
    params       = _load_params()
    wf_cfg       = params['walkforward']
    xgb_cfg      = params['xgboost']
    trading_days = params['data']['trading_days']

    n_splits      = wf_cfg['n_splits']
    step_size     = wf_cfg['step_size']
    min_train_size = wf_cfg['min_train_size']

    logger.info(f'[WF] Walk-forward validation — {n_splits} folds × {step_size} days')

    # load precomputed features
    regimes = pd.read_csv(
        ROOT / 'data/processed/hmm_regimes.csv',
        index_col='Date', parse_dates=True
    )
    vols = pd.read_csv(
        ROOT / 'data/processed/garch_egarch_volatilities.csv',
        index_col='Date', parse_dates=True
    )

    macro_aligned = macro.reindex(returns.index).ffill()
    data = pd.concat([returns, macro_aligned, regimes, vols], axis=1).dropna()

    # build full feature matrix using a temp model instance
    tmp_model = HybridVolatilityModel()
    X, y = tmp_model.build_features(data)
    n    = len(X)

    all_dates         = []
    all_y_true        = []
    all_y_pred_hybrid = []
    all_y_pred_garch  = []
    all_y_pred_egarch = []

    for fold in range(n_splits):
        test_end   = n - (n_splits - 1 - fold) * step_size
        test_start = test_end - step_size
        train_end  = test_start

        if train_end < min_train_size:
            logger.warning(f'[WF] Fold {fold + 1} skipped — insufficient training data')
            continue

        X_tr_full = X.iloc[:train_end]
        y_tr_full = y.iloc[:train_end]
        X_test    = X.iloc[test_start:test_end]
        y_test    = y.iloc[test_start:test_end]

        # internal val split for XGBoost early stopping (last 15% of train)
        val_size = max(1, int(len(X_tr_full) * 0.15))
        X_train  = X_tr_full.iloc[:-val_size]
        y_train  = y_tr_full.iloc[:-val_size]
        X_val    = X_tr_full.iloc[-val_size:]
        y_val    = y_tr_full.iloc[-val_size:]

        model = HybridVolatilityModel(
            n_estimators          = xgb_cfg['n_estimators'],
            learning_rate         = xgb_cfg['learning_rate'],
            max_depth             = xgb_cfg['max_depth'],
            subsample             = xgb_cfg['subsample'],
            colsample_bytree      = xgb_cfg['colsample_bytree'],
            early_stopping_rounds = xgb_cfg['early_stopping_rounds'],
            random_state          = xgb_cfg['random_state'],
        )
        model.fit(X_train, y_train, X_val, y_val)

        y_pred_hybrid = model.predict(X_test)
        y_pred_garch  = (X_test['sigma_garch_ann']  / np.sqrt(trading_days)).values
        y_pred_egarch = (X_test['sigma_egarch_ann'] / np.sqrt(trading_days)).values

        all_dates.extend(y_test.index)
        all_y_true.extend(y_test.values)
        all_y_pred_hybrid.extend(y_pred_hybrid)
        all_y_pred_garch.extend(y_pred_garch)
        all_y_pred_egarch.extend(y_pred_egarch)

        fold_rmse = float(np.sqrt(np.mean((y_test.values - y_pred_hybrid) ** 2)))
        logger.info(
            f'[WF] Fold {fold + 1}/{n_splits} — '
            f'train: {train_end:,} obs | '
            f'test: {y_test.index[0].date()} → {y_test.index[-1].date()} | '
            f'RMSE: {fold_rmse:.4f}'
        )

    y_true        = np.array(all_y_true)
    y_pred_hybrid = np.array(all_y_pred_hybrid)
    y_pred_garch  = np.array(all_y_pred_garch)
    y_pred_egarch = np.array(all_y_pred_egarch)

    wf_rmse_hybrid         = float(np.sqrt(np.mean((y_true - y_pred_hybrid)  ** 2)))
    wf_rmse_garch          = float(np.sqrt(np.mean((y_true - y_pred_garch)   ** 2)))
    wf_rmse_egarch         = float(np.sqrt(np.mean((y_true - y_pred_egarch)  ** 2)))
    wf_rmse_improvement_pct = round((wf_rmse_garch - wf_rmse_hybrid) / wf_rmse_garch * 100, 2)
    wf_qlike_hybrid        = qlike(y_true, y_pred_hybrid)
    wf_qlike_garch         = qlike(y_true, y_pred_garch)
    wf_qlike_egarch        = qlike(y_true, y_pred_egarch)

    # save out-of-sample predictions
    wf_df = pd.DataFrame({
        'y_true'        : y_true,
        'y_pred_hybrid' : y_pred_hybrid,
        'y_pred_garch'  : y_pred_garch,
        'y_pred_egarch' : y_pred_egarch,
    }, index=pd.DatetimeIndex(all_dates))
    wf_df.index.name = 'Date'

    with atomic_write(ROOT / 'data/processed/walkforward_predictions.csv') as tmp:
        wf_df.to_csv(tmp)

    # append wf metrics to metrics.json
    metrics_path = ROOT / 'metrics.json'
    with open(metrics_path) as f:
        metrics = json.load(f)

    metrics.update({
        'wf_rmse_hybrid'          : round(wf_rmse_hybrid,          6),
        'wf_rmse_garch'           : round(wf_rmse_garch,           6),
        'wf_rmse_egarch'          : round(wf_rmse_egarch,          6),
        'wf_rmse_improvement_pct' : wf_rmse_improvement_pct,
        'wf_qlike_hybrid'         : round(wf_qlike_hybrid,         6),
        'wf_qlike_garch'          : round(wf_qlike_garch,          6),
        'wf_qlike_egarch'         : round(wf_qlike_egarch,         6),
    })

    with atomic_write(metrics_path) as tmp:
        with open(tmp, 'w') as f:
            json.dump(metrics, f, indent=2)

    logger.info(
        f'[WF] Done — WF RMSE hybrid={wf_rmse_hybrid:.4f} | '
        f'GARCH={wf_rmse_garch:.4f} | '
        f'improvement={wf_rmse_improvement_pct:+.1f}%  '
        f'({time.time() - t0:.1f}s)'
    )
    return wf_df


# ---------------------------------------------------------------------------
# figure building
# ---------------------------------------------------------------------------

def _build_pipeline_diagram(path: Path) -> None:
    """
    Draw the nine DVC pipeline stages as a backwards-C (snake) flow diagram.

    Top row  (left → right) : 01 → 02 → 03 → 04 → 05
    Connector (top-right down): 05 ↓ 06
    Bottom row (right → left): 06 → 07 → 08 → 09

    Imports matplotlib locally — only called from step_build_figures.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from src.dark_viz import C, savefig

    # top row: (x_position, label, colour)
    top = [
        (0, '01\nLoad Data',    C.GARCH),
        (1, '02\nReturns',      C.GARCH),
        (2, '03\n(E)GARCH',     C.GARCH),
        (3, '04\nHMM',          C.MED),
        (4, '05\nFeatures',     C.MED),
    ]
    # bottom row right-to-left: 06 at x=4, 07 at x=3, 08 at x=2, 09 at x=1
    bot = [
        (4, '06\nHybrid XGB',   C.HYBRID),
        (3, '07\nConformal',    C.HYBRID),
        (2, '08\nWalk-Forward', C.HYBRID),
        (1, '09\nFigures',      C.LOW),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-0.38, 1.45)
    ax.axis('off')

    box_w, box_h = 0.82, 0.46
    Y_TOP, Y_BOT = 1.0, 0.0

    def draw_box(x, y, label, colour):
        face_rgba = (*mcolors.to_rgb(colour), 0.18)
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.04',
            linewidth=1.2, edgecolor=colour, facecolor=face_rgba,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                color=C.TEXT, fontsize=13, fontweight='bold', linespacing=1.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', color=C.MUTED, lw=1.1),
        )

    # draw top row + horizontal arrows
    for i, (x, label, colour) in enumerate(top):
        draw_box(x, Y_TOP, label, colour)
        if i < len(top) - 1:
            arrow(x + box_w / 2 + 0.04, Y_TOP,
                  top[i + 1][0] - box_w / 2 - 0.04, Y_TOP)

    # draw bottom row + horizontal arrows (right to left)
    for i, (x, label, colour) in enumerate(bot):
        draw_box(x, Y_BOT, label, colour)
        if i < len(bot) - 1:
            next_x = bot[i + 1][0]
            arrow(x - box_w / 2 - 0.04, Y_BOT,
                  next_x + box_w / 2 + 0.04, Y_BOT)

    # vertical connector: bottom of 05 (x=4, top) → top of 06 (x=4, bot)
    arrow(4, Y_TOP - box_h / 2 - 0.04,
          4, Y_BOT + box_h / 2 + 0.04)

    ax.set_title(
        'DVC Pipeline — regime-aware-volatility-forecasting',
        color=C.TEXT, fontsize=14, pad=12,
    )
    savefig(fig, path)


def step_build_figures() -> None:
    """
    Build all pre-rendered figures for the Streamlit app and README.

    Reads from data/processed/ — must be called after step_fit_conformal
    and step_walk_forward.

    Saves to assets/figures/:
        dark/01_vol_regimes.png     — full-period GARCH/EGARCH vol + regime bands
        dark/02_forecast.png        — test-set hybrid vs benchmarks
        dark/03_intervals_80.png    — conformal bands at 80%
        dark/03_intervals_90.png    — conformal bands at 90%
        dark/03_intervals_95.png    — conformal bands at 95%
        dark/04_walkforward.png     — walk-forward out-of-sample predictions
        dark/05_kupiec.png          — empirical upper violation rate vs asymmetric target
        dark/06_regime_coverage.png — per-regime coverage gap bar chart
        dark/07_shap.png            — SHAP mean |value| feature importance bar chart
        readme/hero.png             — wide 2-panel composite for README header
        readme/pipeline_diagram.png — DVC stage flow diagram
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from src.dark_viz import (
        apply_style, C, REGIME_COLOURS, regime_band_ax, savefig,
    )

    t0     = time.time()
    params = _load_params()
    alpha_levels = params['conformal']['alpha_levels']
    phi          = params['conformal']['asymmetry']

    logger.info('[FIG] Building figures...')
    apply_style()

    OUT_DARK   = ROOT / 'assets/figures/dark'
    OUT_README = ROOT / 'assets/figures/readme'

    # ------------------------------------------------------------------
    # load processed artifacts
    # ------------------------------------------------------------------
    intervals = pd.read_csv(
        ROOT / 'data/processed/conformal_intervals.csv',
        index_col='Date', parse_dates=True,
    )
    regimes_full = pd.read_csv(
        ROOT / 'data/processed/hmm_regimes.csv',
        index_col='Date', parse_dates=True,
    )['regime']
    vols_full = pd.read_csv(
        ROOT / 'data/processed/garch_egarch_volatilities.csv',
        index_col='Date', parse_dates=True,
    )
    predictions = pd.read_csv(
        ROOT / 'data/processed/hybrid_predictions.csv',
        index_col='Date', parse_dates=True,
    )

    wf_path = ROOT / 'data/processed/walkforward_predictions.csv'
    wf_df   = (
        pd.read_csv(wf_path, index_col='Date', parse_dates=True)
        if wf_path.exists() else None
    )
    shap_df = pd.read_csv(ROOT / 'data/processed/shap_importance.csv')

    regimes_test = regimes_full.reindex(intervals.index)

    y_true_test   = predictions['y_true'].values
    y_pred_hybrid = predictions['y_pred_hybrid'].values
    y_pred_garch  = predictions['y_pred_garch'].values
    y_pred_egarch = predictions['y_pred_egarch'].values
    dates_test    = predictions.index
    y_true_int    = intervals['y_true'].values

    # ------------------------------------------------------------------
    # Fig 1 — full-period conditional volatility + regime bands
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    regime_band_ax(ax, regimes_full)
    ax.plot(vols_full.index, vols_full['sigma_garch_ann'],
            color=C.GARCH,  lw=0.9, label='GARCH (ann.)')
    ax.plot(vols_full.index, vols_full['sigma_egarch_ann'],
            color=C.EGARCH, lw=0.9, label='EGARCH (ann.)', alpha=0.85)

    import matplotlib.patches as mpatches
    handles = [
        plt.Line2D([0], [0], color=C.GARCH,  lw=1.5, label='GARCH'),
        plt.Line2D([0], [0], color=C.EGARCH, lw=1.5, label='EGARCH'),
        mpatches.Patch(color=C.LOW,  alpha=0.6, label='Low vol'),
        mpatches.Patch(color=C.MED,  alpha=0.6, label='Medium vol'),
        mpatches.Patch(color=C.HIGH, alpha=0.6, label='High vol'),
    ]
    ax.legend(handles=handles, loc='upper right', ncol=3, framealpha=0.3)
    ax.set_title('Conditional Volatility — Full Period with HMM Regime Bands')
    ax.set_ylabel('Annualised Volatility')
    savefig(fig, OUT_DARK / '01_vol_regimes.png')
    logger.info('[FIG] 01_vol_regimes.png')

    # ------------------------------------------------------------------
    # Fig 2 — test-set forecast comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    regime_band_ax(ax, regimes_test)
    ax.plot(dates_test, y_true_test,   color=C.ACTUAL, lw=0.7, alpha=0.8, label='Realised')
    ax.plot(dates_test, y_pred_hybrid, color=C.HYBRID, lw=1.0,            label='Hybrid XGBoost')
    ax.plot(dates_test, y_pred_garch,  color=C.GARCH,  lw=0.8, alpha=0.8, label='GARCH')
    ax.plot(dates_test, y_pred_egarch, color=C.EGARCH, lw=0.8, alpha=0.8, label='EGARCH')
    ax.set_title('Test-Set Volatility Forecast — Hybrid vs Benchmarks')
    ax.set_ylabel('Daily Volatility')
    ax.legend(ncol=4, framealpha=0.3)
    savefig(fig, OUT_DARK / '02_forecast.png')
    logger.info('[FIG] 02_forecast.png')

    # ------------------------------------------------------------------
    # Fig 3 — conformal intervals (one per alpha level)
    # ------------------------------------------------------------------
    for alpha in alpha_levels:
        coverage  = int((1 - alpha) * 100)
        lower     = intervals[f'lower_{coverage}'].values
        upper     = intervals[f'upper_{coverage}'].values
        alpha_upper = alpha * phi
        alpha_lower = alpha * (1 - phi)
        fig, ax   = plt.subplots(figsize=(14, 5))
        regime_band_ax(ax, regimes_test, alpha=0.12)
        ax.fill_between(intervals.index, lower, upper,
                        color=C.BAND, alpha=0.15)
        ax.plot(intervals.index, upper, color=C.UPPER, lw=1.0,
                label=f'Upper bound  (α_upper = {alpha_upper:.0%})')
        ax.plot(intervals.index, lower, color=C.LOWER, lw=1.0,
                label=f'Lower bound  (α_lower = {alpha_lower:.0%})')
        ax.plot(intervals.index, y_true_int, color=C.ACTUAL, lw=0.7, alpha=0.9,
                label='Realised volatility')
        ax.set_title(
            f'Asymmetric Conformal Intervals — {coverage}%  '
            f'(φ={phi},  α_upper={alpha * phi:.1%})'
        )
        ax.set_ylabel('Daily Volatility')
        ax.legend(ncol=3, framealpha=0.3)
        savefig(fig, OUT_DARK / f'03_intervals_{coverage}.png')
        logger.info(f'[FIG] 03_intervals_{coverage}.png')

    # ------------------------------------------------------------------
    # Fig 4 — walk-forward validation
    # ------------------------------------------------------------------
    if wf_df is not None:
        regimes_wf = regimes_full.reindex(wf_df.index)
        fig, ax    = plt.subplots(figsize=(14, 5))
        regime_band_ax(ax, regimes_wf)
        ax.plot(wf_df.index, wf_df['y_true'],        color=C.ACTUAL, lw=0.7, alpha=0.8, label='Realised')
        ax.plot(wf_df.index, wf_df['y_pred_hybrid'], color=C.HYBRID, lw=1.0,            label='Hybrid XGBoost')
        ax.plot(wf_df.index, wf_df['y_pred_garch'],  color=C.GARCH,  lw=0.8, alpha=0.8, label='GARCH')

        step_size = params['walkforward']['step_size']
        for i in range(step_size, len(wf_df), step_size):
            ax.axvline(wf_df.index[i], color=C.MUTED, lw=0.8, linestyle=':', alpha=0.6)

        ax.set_title('Walk-Forward Validation — Expanding Window (XGBoost)')
        ax.set_ylabel('Daily Volatility')
        ax.legend(ncol=4, framealpha=0.3)
        savefig(fig, OUT_DARK / '04_walkforward.png')
        logger.info('[FIG] 04_walkforward.png')

    # ------------------------------------------------------------------
    # Fig 5 — Kupiec POF violation rates vs asymmetric targets
    # ------------------------------------------------------------------
    coverages, emp_rates, targets = [], [], []
    for alpha in alpha_levels:
        coverage = int((1 - alpha) * 100)
        upper    = intervals[f'upper_{coverage}'].values
        coverages.append(f'{coverage}%')
        emp_rates.append(float(np.mean(y_true_int > upper)))
        targets.append(alpha * phi)

    x          = np.arange(len(coverages))
    bar_colours = [C.NEGATIVE if e > t else C.POSITIVE
                   for e, t in zip(emp_rates, targets)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, emp_rates, color=bar_colours, alpha=0.75, width=0.5,
           label='Empirical upper viol. rate')
    for xi, t in zip(x, targets):
        ax.plot([xi - 0.25, xi + 0.25], [t, t], color=C.ACTUAL, lw=2.0, zorder=5)
    ax.scatter(x, targets, color=C.ACTUAL, s=50, zorder=6,
               label=f'Target α_upper  (φ={phi})')
    ax.set_xticks(x)
    ax.set_xticklabels(coverages)
    ax.set_title('Kupiec POF — Upper Violation Rate vs Asymmetric Target')
    ax.set_ylabel('Upper Violation Rate')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
    ax.legend(framealpha=0.3)
    savefig(fig, OUT_DARK / '05_kupiec.png')
    logger.info('[FIG] 05_kupiec.png')

    # ------------------------------------------------------------------
    # Fig 6 — regime-conditioned coverage gap (90% interval)
    # ------------------------------------------------------------------
    alpha_upper_90 = 0.10 * phi
    regime_labels  = ['Low', 'Medium', 'High']
    gaps, ns       = [], []

    for k in range(3):
        mask = regimes_test.values == k
        if mask.sum() == 0:
            gaps.append(0.0); ns.append(0); continue
        viol = float(np.mean(y_true_int[mask] > intervals['upper_90'].values[mask]))
        gaps.append(viol - alpha_upper_90)
        ns.append(int(mask.sum()))

    x          = np.arange(3)
    bar_colours = [C.NEGATIVE if g > 0 else C.POSITIVE for g in gaps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, gaps, color=bar_colours, alpha=0.75, width=0.5)
    ax.axhline(0, color=C.MUTED, lw=1.0, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(regime_labels, ns)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1%}'))
    ax.set_title(
        f'Regime Coverage Gap vs {alpha_upper_90:.0%} Target  '
        f'(90% interval, φ={phi})'
    )
    ax.set_ylabel('Empirical viol. rate − target')
    savefig(fig, OUT_DARK / '06_regime_coverage.png')
    logger.info('[FIG] 06_regime_coverage.png')

    # ------------------------------------------------------------------
    # Fig 7 — SHAP feature importance bar chart
    # ------------------------------------------------------------------
    features   = shap_df['feature'].tolist()
    importances = shap_df['mean_abs_shap'].tolist()

    # highlight the top feature (expected: regime) in HYBRID orange
    bar_colours = [C.HYBRID if i == 0 else C.GARCH for i in range(len(features))]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = np.arange(len(features))
    ax.barh(y_pos, importances, color=bar_colours, alpha=0.80, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()   # most important at top
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('SHAP Feature Importance — XGBoost Hybrid')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.4f}'))

    # annotate value on each bar
    for i, val in enumerate(importances):
        ax.text(val + max(importances) * 0.01, i, f'{val:.5f}',
                va='center', fontsize=8.5, color=C.TEXT)

    # legend: regime bar is highlighted
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=C.HYBRID, alpha=0.8, label='Regime (rank 1)'),
            Patch(color=C.GARCH,  alpha=0.8, label='Other features'),
        ],
        loc='lower right', framealpha=0.3,
    )
    savefig(fig, OUT_DARK / '07_shap.png')
    logger.info('[FIG] 07_shap.png')

    # ------------------------------------------------------------------
    # README hero — 2-panel composite
    # ------------------------------------------------------------------
    apply_style(font_scale=1.1)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 5))

    regime_band_ax(ax_l, regimes_test)
    ax_l.plot(dates_test, y_true_test,   color=C.ACTUAL, lw=0.7, alpha=0.8, label='Realised')
    ax_l.plot(dates_test, y_pred_hybrid, color=C.HYBRID, lw=1.0,            label='Hybrid XGBoost')
    ax_l.plot(dates_test, y_pred_garch,  color=C.GARCH,  lw=0.8, alpha=0.8, label='GARCH')
    ax_l.set_title('Volatility Forecast — Test Set')
    ax_l.set_ylabel('Daily Volatility')
    ax_l.legend(ncol=3, framealpha=0.3, fontsize=9)

    lower_90 = intervals['lower_90'].values
    upper_90 = intervals['upper_90'].values
    regime_band_ax(ax_r, regimes_test, alpha=0.12)
    ax_r.fill_between(intervals.index, lower_90, upper_90,
                      color=C.BAND, alpha=0.15)
    ax_r.plot(intervals.index, upper_90,   color=C.UPPER, lw=1.0,
              label=f'Upper bound  (α_upper = {0.10 * phi:.0%})')
    ax_r.plot(intervals.index, lower_90,   color=C.LOWER, lw=1.0,
              label=f'Lower bound  (α_lower = {0.10 * (1-phi):.0%})')
    ax_r.plot(intervals.index, y_true_int, color=C.ACTUAL, lw=0.7, alpha=0.9,
              label='Realised volatility')
    ax_r.set_title(f'Asymmetric Conformal Intervals — 90%  (φ={phi})')
    ax_r.set_ylabel('Daily Volatility')
    ax_r.legend(ncol=3, framealpha=0.3, fontsize=9)

    savefig(fig, OUT_README / 'hero.png')
    logger.info('[FIG] readme/hero.png')

    # ------------------------------------------------------------------
    # README pipeline diagram
    # ------------------------------------------------------------------
    apply_style()
    _build_pipeline_diagram(OUT_README / 'pipeline_diagram.png')
    logger.info('[FIG] readme/pipeline_diagram.png')

    logger.info(f'[FIG] All figures built  ({time.time() - t0:.1f}s)')


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
        8. Walk-forward validation
        9. Build figures
    """
    setup_logging()
    t_total = time.time()
    logger.info('=' * 60)
    logger.info('VolatilityRegimes — Full Pipeline')
    logger.info('=' * 60)

    prices, macro        = step_load_data()
    returns              = step_compute_returns(prices)
    garch, egarch        = step_fit_garch(returns)
    hmm                  = step_fit_hmm(returns)
    step_export_features(returns, garch, egarch, hmm)
    hybrid, X_val, y_val = step_fit_hybrid(returns, macro)
    cp                   = step_fit_conformal(hybrid)
    step_walk_forward(returns, macro)
    step_build_figures()

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