"""
stages/05_export_features.py
-----------------------------
DVC stage: export regime and volatility features.

Inputs:  data/processed/returns.csv
         data/processed/garch.pkl, data/processed/egarch.pkl
         data/processed/hmm.pkl
Outputs: data/processed/hmm_regimes.csv
         data/processed/garch_egarch_volatilities.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.logger import setup_logging
from src.models.garch import VolatilityModel
from src.models.hmm import RegimeHMM
from src.pipeline import step_export_features

setup_logging()

returns = pd.read_csv(
    ROOT / 'data/processed/returns.csv',
    index_col='Date', parse_dates=True
).squeeze()

garch  = VolatilityModel.load(ROOT / 'data/processed/garch.pkl')
egarch = VolatilityModel.load(ROOT / 'data/processed/egarch.pkl')
hmm    = RegimeHMM.load(ROOT / 'data/processed/hmm.pkl')

step_export_features(returns, garch, egarch, hmm)
