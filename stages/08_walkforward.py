"""
stages/08_walkforward.py
------------------------
DVC stage: expanding-window walk-forward validation of XGBoost hybrid model.

Inputs:  data/processed/returns.csv
         data/processed/macro.csv
         data/processed/hmm_regimes.csv
         data/processed/garch_egarch_volatilities.csv
         data/processed/metrics.json  (appends wf_* metrics)
Outputs: data/processed/walkforward_predictions.csv
         metrics.json  (updated with wf_* metrics)
Params:  params.yaml -> walkforward, xgboost, data
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.logger import setup_logging
from src.pipeline import step_walk_forward

setup_logging()
logger = logging.getLogger(__name__)

try:
    returns = pd.read_csv(
        ROOT / 'data/processed/returns.csv',
        index_col='Date', parse_dates=True
    ).squeeze()
    macro = pd.read_csv(
        ROOT / 'data/processed/macro.csv',
        index_col='Date', parse_dates=True
    )
    step_walk_forward(returns, macro)
except Exception as e:
    logger.error(f'Stage walkforward failed: {e}', exc_info=True)
    raise SystemExit(1)
