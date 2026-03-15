"""
src/models/conformal.py
-----------------------
Split conformal prediction wrapper for volatility forecast intervals.

Method
------
Split conformal prediction (Vovk et al., 2005) produces distribution-free
prediction intervals with a finite-sample marginal coverage guarantee:

    P(y_{T+1} in [L_{T+1}, U_{T+1}]) >= 1 - alpha

for any significance level alpha in (0, 1), without any assumption on the
data-generating process, as long as the calibration data are exchangeable
with the test data.

The method requires three inputs:
    1. A fitted point forecaster (any model with a predict() method)
    2. A calibration set of (y_true, y_pred) pairs not used during training
    3. A desired coverage level 1 - alpha

The nonconformity score is the absolute residual:
    s_i = |y_i - y_hat_i|

The prediction interval for a new point is:
    [y_hat - q_hat, y_hat + q_hat]

where q_hat is the (1-alpha) empirical quantile of the calibration scores.
The interval width is constant across all test points (marginal coverage).

Usage
-----
    from src.models.conformal import ConformalPredictor

    cp = ConformalPredictor()
    cp.calibrate(y_true_val, y_pred_val)
    intervals = cp.predict(y_pred_test, alpha=0.10)  # 90% coverage
    cp.save('data/processed/conformal.pkl')
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class ConformalPredictor:
    """
    Split conformal prediction wrapper.

    Wraps any point forecast into a symmetric prediction interval
    with a finite-sample marginal coverage guarantee. The calibration
    set should be the validation set residuals from the point forecaster
    — data that was not used during model training.

    Parameters
    ----------
    None — all state is set during calibrate().

    Attributes
    ----------
    cal_residuals_ : np.ndarray or None
        Absolute residuals on the calibration set. Populated by
        calibrate(). None before calibrate() is called.
    n_cal_ : int or None
        Number of calibration observations.
    """

    def __init__(self) -> None:
        self.cal_residuals_ = None
        self.n_cal_         = None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _check_calibrated(self) -> None:
        """Raise RuntimeError if calibrate() has not been called."""
        if self.cal_residuals_ is None:
            raise RuntimeError(
                'Predictor has not been calibrated. '
                'Call .calibrate(y_true, y_pred) first.'
            )

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> 'ConformalPredictor':
        """
        Compute and store nonconformity scores from the calibration set.

        The nonconformity score is the absolute residual:
            s_i = |y_true_i - y_pred_i|

        Parameters
        ----------
        y_true : np.ndarray
            Realised values on the calibration set.
        y_pred : np.ndarray
            Point forecast values on the calibration set.

        Returns
        -------
        self : ConformalPredictor
            Returns self for method chaining.
        """
        self.cal_residuals_ = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        self.n_cal_         = len(self.cal_residuals_)
        return self

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def quantile(self, alpha: float) -> float:
        """
        Return the (1-alpha) empirical quantile of the calibration scores.

        This is the half-width of the symmetric prediction interval at
        coverage level 1 - alpha.

        Parameters
        ----------
        alpha : float
            Significance level in (0, 1). For 90% coverage use alpha=0.10.

        Returns
        -------
        float — scalar quantile value (interval half-width).
        """
        self._check_calibrated()
        return float(np.quantile(self.cal_residuals_, 1 - alpha))

    def predict(
        self,
        y_pred: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute symmetric prediction intervals around a point forecast.

        For each test point, the interval is:
            [y_pred - q_hat, y_pred + q_hat]

        where q_hat = quantile(cal_residuals, 1 - alpha).

        Parameters
        ----------
        y_pred : np.ndarray
            Point forecast values on the test set.
        alpha : float
            Significance level in (0, 1). For 90% coverage use alpha=0.10.

        Returns
        -------
        lower : np.ndarray — lower bound of the prediction interval.
        upper : np.ndarray — upper bound of the prediction interval.
        """
        self._check_calibrated()
        q     = self.quantile(alpha)
        y_pred = np.asarray(y_pred)
        return y_pred - q, y_pred + q

    def predict_all(
        self,
        y_pred: np.ndarray,
        alphas: list[float] | None = None,
    ) -> dict[str, dict]:
        """
        Compute prediction intervals for multiple coverage levels at once.

        Parameters
        ----------
        y_pred : np.ndarray
            Point forecast values on the test set.
        alphas : list[float] or None
            List of significance levels. Defaults to [0.20, 0.10, 0.05]
            corresponding to 80%, 90%, 95% coverage.

        Returns
        -------
        dict keyed by str(alpha), each containing:
            'quantile' : float   — interval half-width
            'lower'    : ndarray — lower bounds
            'upper'    : ndarray — upper bounds
        """
        self._check_calibrated()
        if alphas is None:
            alphas = [0.20, 0.10, 0.05]

        results = {}
        for alpha in alphas:
            lower, upper = self.predict(y_pred, alpha)
            results[str(alpha)] = {
                'quantile' : self.quantile(alpha),
                'lower'    : lower,
                'upper'    : upper,
            }
        return results

    # ------------------------------------------------------------------
    # coverage evaluation
    # ------------------------------------------------------------------

    def coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Compute empirical coverage on the test set.

        Empirical coverage is the fraction of test observations that
        fall inside the prediction interval:

            coverage = mean(lower <= y_true <= upper)

        Parameters
        ----------
        y_true : np.ndarray
            Realised values on the test set.
        y_pred : np.ndarray
            Point forecast values on the test set.
        alpha : float
            Significance level used to construct the interval.

        Returns
        -------
        float — empirical coverage in [0, 1].
        """
        self._check_calibrated()
        lower, upper = self.predict(y_pred, alpha)
        y_true = np.asarray(y_true)
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    def coverage_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alphas: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Return a summary DataFrame of nominal vs empirical coverage.

        Parameters
        ----------
        y_true : np.ndarray
            Realised values on the test set.
        y_pred : np.ndarray
            Point forecast values on the test set.
        alphas : list[float] or None
            Significance levels. Defaults to [0.20, 0.10, 0.05].

        Returns
        -------
        pd.DataFrame with columns:
            Empirical Coverage, Gap, Interval Width
        Index: Nominal Coverage (80%, 90%, 95%)
        """
        self._check_calibrated()
        if alphas is None:
            alphas = [0.20, 0.10, 0.05]

        rows = []
        for alpha in alphas:
            nominal  = 1 - alpha
            empirical = self.coverage(y_true, y_pred, alpha)
            q        = self.quantile(alpha)
            rows.append({
                'Nominal Coverage'   : f'{nominal:.0%}',
                'Empirical Coverage' : f'{empirical:.2%}',
                'Gap'                : f'{nominal - empirical:+.2%}',
                'Interval Width'     : f'{2 * q:.6f}',
            })

        return pd.DataFrame(rows).set_index('Nominal Coverage')

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """
        Pickle the fitted ConformalPredictor to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created
            if they do not exist.
        """
        self._check_calibrated()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'ConformalPredictor saved → {path}')

    @classmethod
    def load(cls, path) -> 'ConformalPredictor':
        """
        Load a pickled ConformalPredictor from disk.

        Parameters
        ----------
        path : str or Path
            Path to a .pkl file saved by ConformalPredictor.save().

        Returns
        -------
        ConformalPredictor with cal_residuals_ restored.

        Raises
        ------
        FileNotFoundError if path does not exist.
        TypeError if the unpickled object is not a ConformalPredictor.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'No file found at {path}')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Expected ConformalPredictor, got {type(obj)}')
        print(f'ConformalPredictor loaded ← {path}')
        return obj

    # ------------------------------------------------------------------
    # dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.cal_residuals_ is None:
            return 'ConformalPredictor(status=\'uncalibrated\')'
        return (
            f'ConformalPredictor('
            f'n_cal={self.n_cal_}, '
            f'q_90={self.quantile(0.10):.6f})'
        )
