"""
stages/02_compute_returns.py
-----------------------------
DVC stage: compute log-returns from raw prices.

Inputs:  data/processed/prices.csv
Outputs: data/processed/returns.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import logging
from src.logger import setup_logging
from src.pipeline import step_compute_returns

setup_logging()
logger = logging.getLogger(__name__)

try:
    prices = pd.read_csv(
        ROOT / 'data/processed/prices.csv',
        index_col='Date', parse_dates=True
    ).squeeze()
    step_compute_returns(prices)
except Exception as e:
    logger.error(f'Stage compute_returns failed: {e}', exc_info=True)
    raise SystemExit(1)
