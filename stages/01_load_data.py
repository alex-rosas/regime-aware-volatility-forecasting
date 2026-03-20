"""
stages/01_load_data.py
----------------------
DVC stage: load raw MXN/USD prices and macro data.

Inputs:  data/raw/mxn_usd.csv, data/raw/macro.csv
Outputs: data/processed/prices.csv, data/processed/macro.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.logger import setup_logging
from src.pipeline import step_load_data

setup_logging()
step_load_data()
