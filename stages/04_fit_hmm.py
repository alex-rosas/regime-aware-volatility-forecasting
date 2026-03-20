"""
stages/04_fit_hmm.py
--------------------
DVC stage: fit HMM regime model.

Inputs:  data/processed/returns.csv
Outputs: data/processed/hmm.pkl
Params:  params.yaml -> hmm
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.logger import setup_logging
from src.pipeline import step_fit_hmm

setup_logging()

returns = pd.read_csv(
    ROOT / 'data/processed/returns.csv',
    index_col='Date', parse_dates=True
).squeeze()

step_fit_hmm(returns)
