"""
src/models/hybrid.py
--------------------
Hybrid econometric–machine learning volatility forecasting model.

Architecture
------------
The hybrid model combines outputs from the econometric pipeline
(GARCH, EGARCH, HMM) with macro features as inputs to an XGBoost
regressor. The econometric models encode known structure — volatility
persistence, asymmetry, and regime dynamics. The XGBoost layer learns
nonlinear interactions between these signals and macroeconomic conditions
that the econometric models cannot capture by construction.

Feature matrix
--------------
    sigma_garch_ann    : GARCH(1,1)-t annualised conditional volatility
    sigma_egarch_ann   : EGARCH(1,1)-t annualised conditional volatility
    regime             : HMM discrete regime label (0=low, 1=medium, 2=high)
    r_lag1/2/3         : lagged log-returns (t-1, t-2, t-3)
    VIXCLS             : VIX index (global risk appetite)
    T10Y2Y             : 10Y-2Y Treasury yield spread

Target
------
    y = |r_{t+1}|      : next-day absolute return (realized volatility proxy)

Workflow
--------
    from src.models.hybrid import HybridVolatilityModel, qlike, diebold_mariano

    model = HybridVolatilityModel()
    X, y  = model.build_features(data)
    X_train, y_train, X_val, y_val, X_test, y_test = model.split(X, y)
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)
    model.save('data/processed/hybrid_model.pkl')
"""

import pickle
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    'MXN_USD_log_return',
    'VIXCLS',
    'T10Y2Y',
    'regime',
    'sigma_garch_ann',
    'sigma_egarch_ann',
}

FEATURE_COLS = [
    'sigma_garch_ann',
    'sigma_egarch_ann',
    'regime',
    'r_lag1',
    'r_lag2',
    'r_lag3',
    'VIXCLS',
    'T10Y2Y',
]


# ---------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------

class HybridVolatilityModel:
    """
    XGBoost hybrid volatility forecasting model.

    Combines GARCH, EGARCH, and HMM outputs with macro features to
    forecast next-day realised volatility. Feature construction,
    splitting, fitting, evaluation, and serialisation are encapsulated
    here so that the notebook only calls the interface.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds. Early stopping will
        typically stop well before this limit. Default 1000.
    learning_rate : float
        Step size shrinkage. Smaller values require more trees but
        generalise better. Default 0.05.
    max_depth : int
        Maximum tree depth. Controls model capacity. Default 4.
    subsample : float
        Fraction of training rows sampled per tree. Default 0.8.
    colsample_bytree : float
        Fraction of features sampled per tree. Default 0.8.
    early_stopping_rounds : int
        Stop training if validation metric does not improve for
        this many consecutive rounds. Default 50.
    random_state : int
        Random seed for reproducibility. Default 42.
    eval_metric : str
        XGBoost evaluation metric for early stopping. Default 'rmse'.

    Attributes
    ----------
    model_ : XGBRegressor or None
        The fitted XGBoost model. None before fit() is called.
    feature_names_ : list[str] or None
        Column names of the feature matrix used during fit().
    train_rmse_ : float or None
        RMSE on the training set after fitting.
    val_rmse_ : float or None
        RMSE on the validation set after fitting.
    """

    def __init__(
        self,
        n_estimators: int        = 1000,
        learning_rate: float     = 0.05,
        max_depth: int           = 4,
        subsample: float         = 0.8,
        colsample_bytree: float  = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int        = 42,
        eval_metric: str         = 'rmse',
    ) -> None:
        
        self.n_estimators           = n_estimators
        self.learning_rate          = learning_rate
        self.max_depth              = max_depth
        self.subsample              = subsample
        self.colsample_bytree       = colsample_bytree
        self.early_stopping_rounds  = early_stopping_rounds
        self.random_state           = random_state
        self.eval_metric            = eval_metric

        # populated after fit()
        self.model_         = None
        self.feature_names_ = None
        self.train_rmse_    = None
        self.val_rmse_      = None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called."""
        if self.model_ is None:
            raise RuntimeError(
                'Model has not been fitted yet. Call .fit() first.'
            )

    # ------------------------------------------------------------------
    # feature engineering
    # ------------------------------------------------------------------

    def build_features(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build the feature matrix X and target vector y from the
        aligned dataset.

        Adds three lagged return columns (r_lag1, r_lag2, r_lag3) and
        the forward absolute return target y = |r_{t+1}|. Drops rows
        with NaNs introduced by the lags and the forward shift.

        The input DataFrame is not mutated — a copy is made internally.

        Parameters
        ----------
        data : pd.DataFrame
            Aligned dataset with a DatetimeIndex. Must contain all
            columns in REQUIRED_COLUMNS.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix with columns defined by FEATURE_COLS.
        y : pd.Series
            Target vector of next-day absolute returns.

        Raises
        ------
        ValueError if any required column is missing from data.
        """
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f'Missing required columns: {missing}')

        df = data.copy()

        # lagged returns — all are known at time t before r_t is observed
        df['r_lag1'] = df['MXN_USD_log_return'].shift(1)
        df['r_lag2'] = df['MXN_USD_log_return'].shift(2)
        df['r_lag3'] = df['MXN_USD_log_return'].shift(3)

        # forward target — next-day absolute return
        df['y'] = df['MXN_USD_log_return'].shift(-1).abs()

        # drop NaNs from lags (first 3 rows) and target (last row)
        df = df.dropna(subset=['r_lag1', 'r_lag2', 'r_lag3', 'y'])

        X = df[FEATURE_COLS]
        y = df['y']

        return X, y

    # ------------------------------------------------------------------
    # splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split(
        X: pd.DataFrame,
        y: pd.Series,
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
    ) -> tuple[pd.DataFrame, pd.Series,
               pd.DataFrame, pd.Series,
               pd.DataFrame, pd.Series]:
        """
        Temporally split X and y into train, validation, and test sets.

        The split is strictly sequential — no shuffling. Proportions
        are applied to the integer index so the temporal order is
        always preserved.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with DatetimeIndex.
        y : pd.Series
            Target vector with the same DatetimeIndex as X.
        train_frac : float
            Fraction of observations for training. Default 0.70.
        val_frac : float
            Fraction of observations for validation. Default 0.15.
            The test set receives the remaining 1 - train_frac - val_frac.

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test
        """
        n         = len(X)
        train_end = int(n * train_frac)
        val_end   = int(n * (train_frac + val_frac))

        X_train, y_train = X.iloc[:train_end],        y.iloc[:train_end]
        X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test,  y_test  = X.iloc[val_end:],          y.iloc[val_end:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> 'HybridVolatilityModel':
        """
        Fit the XGBoost regressor with early stopping on the validation set.

        Stores the fitted model in model_, the feature names in
        feature_names_, and the train/validation RMSE in train_rmse_
        and val_rmse_.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target vector.
        X_val : pd.DataFrame
            Validation feature matrix for early stopping.
        y_val : pd.Series
            Validation target vector for early stopping.

        Returns
        -------
        self : HybridVolatilityModel
            Returns self for method chaining.
        """
        model = XGBRegressor(
            n_estimators            = self.n_estimators,
            learning_rate           = self.learning_rate,
            max_depth               = self.max_depth,
            subsample               = self.subsample,
            colsample_bytree        = self.colsample_bytree,
            early_stopping_rounds   = self.early_stopping_rounds,
            random_state            = self.random_state,
            eval_metric             = self.eval_metric,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.model_         = model
        self.feature_names_ = list(X_train.columns)
        self.train_rmse_    = float(root_mean_squared_error(
                                  y_train, model.predict(X_train)))
        self.val_rmse_      = float(root_mean_squared_error(
                                  y_val, model.predict(X_val)))

        return self

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point forecasts for next-day realised volatility.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns as used during fit().

        Returns
        -------
        np.ndarray of shape (n,) with predicted absolute returns.
        """
        self._check_fitted()
        return self.model_.predict(X)

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Compute exact SHAP values for the fitted XGBoost model.

        Uses shap.Explainer which selects the TreeExplainer backend
        automatically for XGBoost, giving exact (not approximate)
        Shapley values.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to explain — typically X_test.

        Returns
        -------
        shap.Explanation object. Pass to shap.plots.bar() or
        shap.plots.scatter() in the notebook for visualisation.
        """
        self._check_fitted()
        explainer = shap.Explainer(self.model_)
        return explainer(X)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a flat dict of fit metrics, MLflow-ready.

        Returns
        -------
        dict with keys:
            best_iteration : int   — XGBoost best tree count
            train_rmse     : float — RMSE on training set
            val_rmse       : float — RMSE on validation set
        """
        self._check_fitted()
        return {
            'best_iteration' : int(self.model_.best_iteration),
            'train_rmse'     : self.train_rmse_,
            'val_rmse'       : self.val_rmse_,
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """
        Pickle the fitted HybridVolatilityModel to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created
            if they do not exist. Conventionally ends in .pkl.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'HybridVolatilityModel saved → {path}')

    @classmethod
    def load(cls, path) -> 'HybridVolatilityModel':
        """
        Load a pickled HybridVolatilityModel from disk.

        Parameters
        ----------
        path : str or Path
            Path to a .pkl file saved by HybridVolatilityModel.save().

        Returns
        -------
        HybridVolatilityModel with model_ and all attributes restored.

        Raises
        ------
        FileNotFoundError if path does not exist.
        TypeError if the unpickled object is not a HybridVolatilityModel.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'No model file found at {path}')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Expected HybridVolatilityModel, got {type(obj)}')
        print(f'HybridVolatilityModel loaded ← {path}')
        return obj

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        experiment: str = 'hybrid_model',
    ) -> None:
        """
        Log hyperparameters, metrics, and the fitted model to MLflow.

        Parameters
        ----------
        run_name : str or None
            MLflow run name. Defaults to 'HybridVolatilityModel'.
        tracking_uri : str or None
            MLflow tracking URI. If None, uses the current MLflow
            tracking URI — set this in the notebook before calling.
        experiment : str
            MLflow experiment name. Default 'hybrid_model'.
        """
        self._check_fitted()

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment)
        run_name = run_name or 'HybridVolatilityModel'
        s = self.summary()

        with mlflow.start_run(run_name=run_name):
            # log hyperparameters
            mlflow.log_param('n_estimators',          self.n_estimators)
            mlflow.log_param('learning_rate',         self.learning_rate)
            mlflow.log_param('max_depth',             self.max_depth)
            mlflow.log_param('subsample',             self.subsample)
            mlflow.log_param('colsample_bytree',      self.colsample_bytree)
            mlflow.log_param('early_stopping_rounds', self.early_stopping_rounds)
            mlflow.log_param('random_state',          self.random_state)

            # log metrics
            for key, value in s.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # log model artifact as pickle
            with tempfile.NamedTemporaryFile(
                suffix='.pkl', delete=False
            ) as tmp:
                pickle.dump(self, tmp)
                tmp_path = tmp.name

            mlflow.log_artifact(tmp_path, artifact_path='model')
            Path(tmp_path).unlink()

        print(f'MLflow: logged run "{run_name}"')

    # ------------------------------------------------------------------
    # dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = 'fitted' if self.model_ is not None else 'unfitted'
        return (
            f'HybridVolatilityModel('
            f'n_estimators={self.n_estimators}, '
            f'max_depth={self.max_depth}, '
            f'status={status!r})'
        )


# ---------------------------------------------------------------------------
# standalone evaluation functions
# ---------------------------------------------------------------------------

def qlike(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    QLIKE volatility loss function.

    Penalises underestimation of volatility more heavily than
    overestimation — the standard loss function for volatility
    forecast evaluation in the academic literature (Patton, 2011).

    Formula: mean( s²/h - log(s²/h) - 1 )
    where s² = y_true², h = y_pred²

    Both inputs are clipped at eps to avoid division by zero or log(0),
    which can occur if y_true contains exact zeros (days with zero return).

    Parameters
    ----------
    y_true : np.ndarray
        Realised volatility proxy (absolute returns).
    y_pred : np.ndarray
        Predicted volatility (same units as y_true).
    eps : float
        Minimum value for clipping. Default 1e-8.

    Returns
    -------
    float — scalar QLIKE loss. Lower is better.
    """
    s2 = np.clip(y_true, eps, None) ** 2
    h  = np.clip(y_pred, eps, None) ** 2
    return float(np.mean(s2 / h - np.log(s2 / h) - 1))


def diebold_mariano(
    y_true:  np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy (two-sided).

    Tests H0: model 1 and model 2 have equal predictive accuracy,
    measured by squared error loss. A negative DM statistic means
    model 1 has smaller squared errors than model 2 — i.e. model 1
    is better. A p-value below 0.05 means the difference is
    statistically significant.

    Parameters
    ----------
    y_true : np.ndarray
        Realised values.
    y_pred1 : np.ndarray
        Predictions from model 1 (the model being tested).
    y_pred2 : np.ndarray
        Predictions from model 2 (the baseline).

    Returns
    -------
    dm_stat : float
        Test statistic. Negative favours model 1.
    p_value : float
        Two-sided p-value under the standard normal approximation.

    References
    ----------
    Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive
    accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.
    """
    e1      = (y_true - y_pred1) ** 2
    e2      = (y_true - y_pred2) ** 2
    d       = e1 - e2
    n       = len(d)
    dm_stat = d.mean() / (d.std() / np.sqrt(n))
    p_value = 2 * stats.norm.sf(np.abs(dm_stat))
    return float(dm_stat), float(p_value)