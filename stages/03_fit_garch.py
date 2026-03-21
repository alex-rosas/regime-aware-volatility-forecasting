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
import logging
from src.logger import setup_logging
from src.pipeline import step_fit_garch

setup_logging()
logger = logging.getLogger(__name__)

try:
    returns = pd.read_csv(
        ROOT / 'data/processed/returns.csv',
        index_col='Date', parse_dates=True
    ).squeeze()
    step_fit_garch(returns)
except Exception as e:
    logger.error(f'Stage fit_garch failed: {e}', exc_info=True)
    raise SystemExit(1)
