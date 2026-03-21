"""
stages/09_build_figures.py
--------------------------
DVC stage: build all pre-rendered dark-mode figures for the Streamlit
app and README.

Inputs:  data/processed/conformal_intervals.csv
         data/processed/hmm_regimes.csv
         data/processed/garch_egarch_volatilities.csv
         data/processed/hybrid_predictions.csv
         data/processed/walkforward_predictions.csv
Outputs: assets/figures/dark/   (8 PNG files)
         assets/figures/readme/ (2 PNG files)
Params:  params.yaml -> conformal, hmm, walkforward
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.logger import setup_logging
from src.pipeline import step_build_figures

setup_logging()
logger = logging.getLogger(__name__)

try:
    step_build_figures()
except Exception as e:
    logger.error(f'Stage build_figures failed: {e}', exc_info=True)
    raise SystemExit(1)
