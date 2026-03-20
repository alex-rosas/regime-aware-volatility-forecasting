"""
stages/03_fit_garch.py
----------------------
DVC stage: fit GARCH(1,1)-t and EGARCH(1,1)-t models.

Inputs:  data/processed/returns.csv
Outputs: data/processed/garch.pkl, data/processed/egarch.pkl
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.logger import setup_logging
from src.pipeline import step_fit_garch

setup_logging()

returns = pd.read_csv(
    ROOT / 'data/processed/returns.csv',
    index_col='Date', parse_dates=True
).squeeze()

step_fit_garch(returns)
