"""
stages/07_fit_conformal.py
--------------------------
DVC stage: calibrate conformal predictor.

Inputs:  data/processed/hybrid_val_predictions.csv
         data/processed/hybrid_predictions.csv
Outputs: data/processed/conformal.pkl
         data/processed/conformal_intervals.csv
Params:  params.yaml -> conformal
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
from src.logger import setup_logging
from src.models.hybrid import HybridVolatilityModel
from src.pipeline import step_fit_conformal

setup_logging()
logger = logging.getLogger(__name__)

try:
    hybrid = HybridVolatilityModel.load(ROOT / 'data/processed/hybrid_model.pkl')
    step_fit_conformal(hybrid)
except Exception as e:
    logger.error(f'Stage fit_conformal failed: {e}', exc_info=True)
    raise SystemExit(1)
