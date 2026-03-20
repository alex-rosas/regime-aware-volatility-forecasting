"""
stages/06_fit_hybrid.py
-----------------------
DVC stage: fit hybrid XGBoost model.

Inputs:  data/processed/returns.csv, data/processed/macro.csv
         data/processed/hmm_regimes.csv
         data/processed/garch_egarch_volatilities.csv
Outputs: data/processed/hybrid_model.pkl
         data/processed/hybrid_predictions.csv
         data/processed/hybrid_val_predictions.csv
Params:  params.yaml -> xgboost, data
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.logger import setup_logging
from src.pipeline import step_fit_hybrid

setup_logging()

returns = pd.read_csv(
    ROOT / 'data/processed/returns.csv',
    index_col='Date', parse_dates=True
).squeeze()

macro = pd.read_csv(
    ROOT / 'data/processed/macro.csv',
    index_col='Date', parse_dates=True
)

step_fit_hybrid(returns, macro)
