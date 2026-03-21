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


# ---------------------------------------------------------------------------
# Asymmetric conformal predictor
# ---------------------------------------------------------------------------

class AsymmetricConformalPredictor(ConformalPredictor):
    """
    Asymmetric split conformal predictor for volatility forecasting.

    Extends ConformalPredictor by replacing absolute residuals with
    signed residuals, then splitting the alpha miscoverage budget
    asymmetrically between the lower and upper tails.

    Motivation
    ----------
    In volatility forecasting, the cost of underestimating risk (actual
    volatility exceeds the upper bound) is far greater than the cost of
    overestimating it (actual falls below the lower bound). The symmetric
    predictor allocates equal strictness to both sides, which is
    suboptimal from a risk management perspective.

    The asymmetry parameter phi in (0, 1) controls how the alpha budget
    is split:

        alpha_upper = alpha * phi        (upper tail — strict)
        alpha_lower = alpha * (1 - phi)  (lower tail — relaxed)

    With phi = 0.7 and alpha = 0.10:
        alpha_upper = 0.07  → upper bound wrong only 7% of the time
        alpha_lower = 0.03  → lower bound wrong only 3% of the time
        total miscoverage  = 10%  →  90% coverage maintained ✓

    The symmetric case (phi = 0.5) exactly reproduces ConformalPredictor.

    Nonconformity scores
    --------------------
    Signed residuals are used instead of absolute residuals:

        r_i = y_true_i − y_pred_i

    Positive r_i means the model underestimated (dangerous).
    Negative r_i means the model overestimated (less costly).

    Prediction interval
    -------------------
        q_lower = Quantile(r, alpha_lower)       — negative value
        q_upper = Quantile(r, 1 − alpha_upper)   — positive value

        lower = y_pred + q_lower
        upper = y_pred + q_upper

    Parameters
    ----------
    asymmetry : float
        Fraction of the alpha budget allocated to the upper tail.
        Must be in (0, 1). Default 0.7 allocates 70% of miscoverage
        tolerance to the upper bound (risk-management oriented).
        Use 0.5 to recover fully symmetric intervals.
    """

    def __init__(self, asymmetry: float = 0.7) -> None:
        super().__init__()
        if not 0 < asymmetry < 1:
            raise ValueError(f'asymmetry must be in (0, 1), got {asymmetry}')
        self.asymmetry              = asymmetry
        self.cal_signed_residuals_  = None

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> 'AsymmetricConformalPredictor':
        """
        Compute and store signed nonconformity scores.

        Also calls the parent calibrate() to retain absolute residuals
        for backward-compatible methods (coverage, coverage_summary
        from parent are overridden below).

        Parameters
        ----------
        y_true : np.ndarray  — realised values on calibration set.
        y_pred : np.ndarray  — point forecast values on calibration set.

        Returns
        -------
        self
        """
        super().calibrate(y_true, y_pred)
        self.cal_signed_residuals_ = (
            np.asarray(y_true) - np.asarray(y_pred)
        )
        return self

    # ------------------------------------------------------------------
    # quantiles
    # ------------------------------------------------------------------

    def quantile_lower(self, alpha: float) -> float:
        """
        Lower shift for the prediction interval.

        Returns the alpha_lower quantile of signed residuals — a negative
        value that shifts the lower bound below the point forecast.

        Parameters
        ----------
        alpha : float — total significance level.

        Returns
        -------
        float — negative scalar (lower bound shift).
        """
        self._check_calibrated()
        alpha_lower = alpha * (1 - self.asymmetry)
        return float(np.quantile(self.cal_signed_residuals_, alpha_lower))

    def quantile_upper(self, alpha: float) -> float:
        """
        Upper shift for the prediction interval.

        Returns the (1 - alpha_upper) quantile of signed residuals —
        a positive value that shifts the upper bound above the point
        forecast.

        Parameters
        ----------
        alpha : float — total significance level.

        Returns
        -------
        float — positive scalar (upper bound shift).
        """
        self._check_calibrated()
        alpha_upper = alpha * self.asymmetry
        return float(np.quantile(self.cal_signed_residuals_, 1 - alpha_upper))

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        y_pred: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute asymmetric prediction intervals around a point forecast.

        Parameters
        ----------
        y_pred : np.ndarray — point forecast values on the test set.
        alpha  : float      — total significance level.

        Returns
        -------
        lower : np.ndarray — lower bound (y_pred + q_lower).
        upper : np.ndarray — upper bound (y_pred + q_upper).
        """
        self._check_calibrated()
        y_pred  = np.asarray(y_pred)
        q_lower = self.quantile_lower(alpha)
        q_upper = self.quantile_upper(alpha)
        return y_pred + q_lower, y_pred + q_upper

    # ------------------------------------------------------------------
    # coverage evaluation
    # ------------------------------------------------------------------

    def coverage_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alphas: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Return a summary DataFrame showing total, lower, and upper
        coverage — including the asymmetric violation split.

        Parameters
        ----------
        y_true : np.ndarray — realised values on the test set.
        y_pred : np.ndarray — point forecast values on the test set.
        alphas : list[float] or None — significance levels.

        Returns
        -------
        pd.DataFrame with columns:
            Empirical Coverage, Lower Violations, Upper Violations,
            Gap, Interval Width
        """
        self._check_calibrated()
        if alphas is None:
            alphas = [0.20, 0.10, 0.05]

        y_true = np.asarray(y_true)
        rows   = []

        for alpha in alphas:
            nominal  = 1 - alpha
            lower, upper = self.predict(y_pred, alpha)

            total_cov        = float(np.mean((y_true >= lower) & (y_true <= upper)))
            lower_violations = float(np.mean(y_true < lower))
            upper_violations = float(np.mean(y_true > upper))
            width            = self.quantile_upper(alpha) - self.quantile_lower(alpha)

            rows.append({
                'Nominal Coverage'  : f'{nominal:.0%}',
                'Empirical Coverage': f'{total_cov:.2%}',
                'Lower Violations'  : f'{lower_violations:.2%}',
                'Upper Violations'  : f'{upper_violations:.2%}',
                'Gap'               : f'{nominal - total_cov:+.2%}',
                'Interval Width'    : f'{width:.6f}',
            })

        return pd.DataFrame(rows).set_index('Nominal Coverage')

    # ------------------------------------------------------------------
    # dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.cal_signed_residuals_ is None:
            return (
                f'AsymmetricConformalPredictor('
                f'asymmetry={self.asymmetry}, '
                f'status=\'uncalibrated\')'
            )
        return (
            f'AsymmetricConformalPredictor('
            f'n_cal={self.n_cal_}, '
            f'asymmetry={self.asymmetry}, '
            f'q_upper_90={self.quantile_upper(0.10):.6f})'
        )
