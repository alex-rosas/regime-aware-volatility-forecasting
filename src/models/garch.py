"""
src/models/garch.py
-------------------
VolatilityModel — a unified wrapper around arch's GARCH(1,1)-t and
EGARCH(1,1)-t estimators.

Usage
-----
from src.models.garch import VolatilityModel

garch = VolatilityModel(model_type='GARCH')
garch.fit(returns)
print(garch.summary())
garch.save(ROOT / 'data/processed/garch.pkl')

loaded = VolatilityModel.load(ROOT / 'data/processed/garch.pkl')
sigma  = loaded.conditional_volatility()
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

import mlflow
import tempfile

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

VALID_TYPES = ('GARCH', 'EGARCH')
SCALE       = 100.0   # arch is numerically unstable on raw returns (~0.006)
                       # we scale to percent returns (~0.6) before fitting


# ---------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------

class VolatilityModel:
    """
    Unified wrapper for GARCH(1,1)-t and EGARCH(1,1)-t estimation.

    Both models are fitted on percent returns (returns * 100) for numerical
    stability. Conditional volatility is rescaled back to the original units
    (decimal returns) before being exposed via public methods.

    Parameters
    ----------
    model_type : str
        Either 'GARCH' or 'EGARCH'.

    Attributes
    ----------
    model_type : str
    result_    : ARCHModelResult | None
        Set after calling fit(). None before fitting.
    returns_   : pd.Series | None
        The raw (unscaled) return series passed to fit().
    Methods
    -------
    fit(returns, disp='off') -> self
        Fit the model to the returns. Returns self for chaining.
    summary() -> dict
        Return a flat dict of parameter estimates and fit metrics.
    to_frame() -> pd.DataFrame
        Return the summary as a formatted DataFrame with standard errors.
    conditional_volatility() -> pd.Series
        Return the annualised conditional volatility in decimal units.
    std_resid() -> pd.Series
        Return the standardized residuals for diagnostics.
    save(path) -> None
        Pickle the fitted model to disk.
    load(path) -> VolatilityModel
        Load a fitted model from disk.
    log_to_mlflow(run_name=None) -> None
        Log the model's parameters and metrics to MLflow.
    """

    def __init__(self, model_type: str) -> None:
        if model_type not in VALID_TYPES:
            raise ValueError(
                f"model_type must be one of {VALID_TYPES}, got '{model_type}'"
            )
        self.model_type: str                    = model_type
        self.result_   : ARCHModelResult | None = None
        self.returns_  : pd.Series | None       = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series, disp: str = 'off') -> 'VolatilityModel':
        """
        Fit the model to a return series.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns in decimal units (e.g. 0.006, not 0.6%).
        disp : str
            Passed to arch's fit(). 'off' suppresses convergence output.

        Returns
        -------
        self  (allows chaining: model.fit(returns).summary())
        """
        self.returns_ = returns

        r = returns * SCALE  # scale to percent returns

        if self.model_type == 'GARCH':
            am = arch_model(r, vol='Garch', p=1, q=1, dist='t')
        else:  # EGARCH — o=1 adds the asymmetry term (gamma)
            am = arch_model(r, vol='EGARCH', p=1, o=1, q=1, dist='t')

        self.result_ = am.fit(disp=disp)
        return self

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, float]:
        """
        Return a flat dict of all estimated parameters and fit metrics.

        Keys
        ----
        GARCH  : mu, omega, alpha, beta, nu, persistence, aic, bic,
                 log_likelihood
        EGARCH : mu, omega, alpha, gamma, beta, nu, persistence, aic, bic,
                 log_likelihood
        """
        self._check_fitted()
        p = self.result_.params
        n = len(self.returns_)

        base = {
            'mu'             : float(p['mu']),
            'omega'          : float(p['omega']),
            'alpha'          : float(p['alpha[1]']),
            'beta'           : float(p['beta[1]']),
            'nu'             : float(p['nu']),
            'aic'            : float(self.result_.aic),
            'bic'            : float(self.result_.bic),
            'log_likelihood' : float(self.result_.loglikelihood),
        }

        if self.model_type == 'GARCH':
            base['persistence'] = base['alpha'] + base['beta']
        else:
            base['gamma']       = float(p['gamma[1]'])
            base['persistence'] = base['beta']  # in EGARCH persistence = beta

        return base

    def to_frame(self) -> pd.DataFrame:
        """
        Return summary as a formatted DataFrame for notebook display.
        Includes standard errors alongside point estimates.
        """
        self._check_fitted()
        p  = self.result_.params
        se = self.result_.std_err

        param_names = list(p.index)
        rows = [
            {
                'Parameter'  : name,
                'Estimate'   : round(float(p[name]), 6),
                'Std Error'  : round(float(se[name]), 6),
                't-stat'     : round(float(p[name]) / float(se[name]), 3),
            }
            for name in param_names
        ]

        extra = pd.DataFrame([
            {'Parameter': 'persistence',
             'Estimate' : round(self.summary()['persistence'], 6),
             'Std Error': float('nan'),
             't-stat'   : float('nan')},
            {'Parameter': 'AIC',
             'Estimate' : round(self.result_.aic, 4),
             'Std Error': float('nan'),
             't-stat'   : float('nan')},
            {'Parameter': 'BIC',
             'Estimate' : round(self.result_.bic, 4),
             'Std Error': float('nan'),
             't-stat'   : float('nan')},
        ])

        return pd.concat([pd.DataFrame(rows), extra], ignore_index=True)

    def conditional_volatility(self) -> pd.Series:
        """
        Return annualised conditional volatility in decimal units.

        The arch library returns volatility in the same units as the input
        (percent returns). We divide by SCALE to convert back to decimal,
        then multiply by sqrt(252) to annualise.

        Returns
        -------
        pd.Series with the same DatetimeIndex as the input returns.
        """
        self._check_fitted()
        SCALE = 100
        sigma_daily = self.result_.conditional_volatility / SCALE
        sigma_ann   = sigma_daily * np.sqrt(252)
        
        # align index safely regardless of whether arch drops first obs
        sigma_ann.index = self.returns_.index[len(self.returns_) - len(sigma_ann):]
        sigma_ann.name  = f'sigma_{self.model_type.lower()}_ann'
        return sigma_ann

    def std_resid(self) -> pd.Series:
        """
        Return standardized residuals z_t = epsilon_t / sigma_t.

        Used for residual diagnostics (ARCH-LM, Ljung-Box, QQ-plot).
        """
        self._check_fitted()
        z = pd.Series(
            self.result_.std_resid,
            index=self.returns_.index[1:],
            name=f'std_resid_{self.model_type.lower()}'
        )
        return z.dropna()

    # ------------------------------------------------------------------
    # persistence / serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Pickle the fitted VolatilityModel to disk.

        Parameters
        ----------
        path : Path
            Destination file. Conventionally data/processed/garch.pkl
            or data/processed/egarch.pkl.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Saved {self.model_type} model → {path}')

    @classmethod
    def load(cls, path: Path) -> 'VolatilityModel':
        """
        Load a pickled VolatilityModel from disk.

        Parameters
        ----------
        path : Path
            Path to a .pkl file saved by VolatilityModel.save().

        Returns
        -------
        VolatilityModel with result_ and returns_ restored.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'No model file found at {path}')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Expected VolatilityModel, got {type(obj)}')
        print(f'Loaded {obj.model_type} model ← {path}')
        return obj

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError(
                'Model has not been fitted yet. Call fit(returns) first.'
            )

    def __repr__(self) -> str:
        status = 'fitted' if self.result_ is not None else 'unfitted'
        return f'VolatilityModel(model_type={self.model_type!r}, status={status!r})'
    
    # --------------------------------------------------
    # mlflow saving
    # --------------------------------------------------

    def log_to_mlflow(self, run_name: str | None = None) -> None:
        """
        Log parameters, metrics, and the fitted model artifact to MLflow.

        Saves the model as a temporary pickle, logs it as an artifact,
        then cleans up the temp file.

        Parameters
        ----------
        run_name : str | None
            MLflow run name. Defaults to model_type (e.g. 'GARCH').
        """


        self._check_fitted()

        run_name = run_name or self.model_type

        with mlflow.start_run(run_name=run_name):
            s = self.summary()

            # log params
            mlflow.log_param('model_type', self.model_type)
            mlflow.log_param('dist',       't')
            mlflow.log_param('p',          1)
            mlflow.log_param('q',          1)
            if self.model_type == 'EGARCH':
                mlflow.log_param('o', 1)

            # log metrics
            for key, value in s.items():
                mlflow.log_metric(key, value)

            # log model artifact as pickle
            with tempfile.NamedTemporaryFile(
                suffix='.pkl', delete=False
            ) as tmp:
                pickle.dump(self, tmp)
                tmp_path = tmp.name

            mlflow.log_artifact(tmp_path, artifact_path='model')

            Path(tmp_path).unlink()  # clean up temp file

        print(f'MLflow: logged {self.model_type} run "{run_name}"')