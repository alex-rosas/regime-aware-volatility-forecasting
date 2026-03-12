"""
src/models/hmm.py
-----------------
Gaussian Hidden Markov Model for volatility regime detection.

The model fits a K-state Gaussian HMM to a return series and decodes
the most likely regime sequence via the Viterbi algorithm. States are
sorted by emission variance so that state 0 is always the low-volatility
regime and state K-1 is always the high-volatility regime.

Usage
-----
from src.models.hmm import RegimeHMM

hmm = RegimeHMM(n_components=3)
hmm.fit(returns)
regimes = hmm.predict(returns)   # pd.Series of 0/1/2 labels
hmm.save('data/processed/hmm.pkl')
"""

import pickle
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    """
    Gaussian HMM wrapper for volatility regime detection.

    States are automatically sorted by emission variance after fitting,
    so state 0 = low volatility, state 1 = medium, state 2 = high.
    This remapping is applied consistently in predict() and regime_stats().

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes). Default 3.
    n_iter : int
        Maximum EM iterations. Default 1000.
    random_state : int
        Random seed for reproducibility. Default 42.

    Attributes
    ----------
    model_ : GaussianHMM or None
        The fitted hmmlearn model. None before fit() is called.
    returns_ : pd.Series or None
        The return series passed to fit(). None before fit() is called.
    remap_ : dict
        Mapping from raw hmmlearn state indices to sorted indices.
        Populated by fit().
    """

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_components  = n_components
        self.n_iter        = n_iter
        self.random_state  = random_state
        self.model_        = None
        self.returns_      = None
        self.remap_        = None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called."""
        if self.model_ is None:
            raise RuntimeError(
                'Model has not been fitted yet. Call .fit(returns) first.'
            )

    @staticmethod
    def _build_features(returns: pd.Series) -> np.ndarray:
        """
        Convert a return series to the HMM feature matrix.

        Scales returns to percentage units and reshapes to (n, 1).

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns in decimal units.

        Returns
        -------
        np.ndarray of shape (n, 1) in percentage units.
        """
        return (returns.values * 100).reshape(-1, 1)

    def _build_remap(self) -> dict:
        """
        Build the state remapping dict after fitting.

        Sorts raw hmmlearn states by emission variance (ascending)
        so that state 0 = low volatility, state 2 = high volatility.

        Returns
        -------
        dict mapping raw state index -> sorted state index.
        """
        variances    = [float(self.model_.covars_[k][0, 0])
                        for k in range(self.n_components)]
        sorted_order = np.argsort(variances)          # ascending variance
        return {int(sorted_order[i]): i
                for i in range(self.n_components)}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> 'RegimeHMM':
        """
        Fit the Gaussian HMM to a return series.

        Internally scales returns to percentage units for numerical
        stability, fits via Baum-Welch (EM), then builds the state
        remapping by ascending emission variance.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns with a DatetimeIndex, in decimal units
            (e.g. 0.01 for a 1% daily return).

        Returns
        -------
        self : RegimeHMM
            Returns self for method chaining.
        """
        self.returns_ = returns
        X = self._build_features(returns)

        model = GaussianHMM(
            n_components    = self.n_components,
            covariance_type = 'full',
            n_iter          = self.n_iter,
            random_state    = self.random_state,
        )
        self.model_ = model.fit(X)
        self.remap_ = self._build_remap()
        return self

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Decode the most likely state sequence via the Viterbi algorithm.

        States are remapped by ascending emission variance so that
        0 = low, 1 = medium, 2 = high volatility.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns with a DatetimeIndex. Does not have to be
            the same series used in fit() — can be new out-of-sample data
            as long as the DatetimeIndex is valid.

        Returns
        -------
        pd.Series of int (0, 1, or 2) with the same DatetimeIndex
        as returns.
        """
        self._check_fitted()
        X      = self._build_features(returns)
        raw    = self.model_.predict(X)
        mapped = [self.remap_[s] for s in raw]
        return pd.Series(mapped, index=returns.index, name='regime',
                         dtype=int)

    def transition_matrix(self) -> pd.DataFrame:
        """
        Return the transition matrix as a labelled DataFrame.

        Rows are the 'from' state, columns are the 'to' state.
        Labels use 'State 0', 'State 1', ... before regime interpretation
        is confirmed. After fitting, state 0 = low, state 2 = high.

        Returns
        -------
        pd.DataFrame of shape (n_components, n_components).
        """
        self._check_fitted()
        labels = [f'State {i}' for i in range(self.n_components)]
        return pd.DataFrame(
            self.model_.transmat_,
            index   = [f'from {l}' for l in labels],
            columns = [f'to {l}'   for l in labels],
        ).round(4)

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a flat dict of fit metrics, MLflow-ready.

        BIC formula: -2 * log_likelihood + n_params * log(n)
        For a K-state 1D Gaussian HMM:
            n_params = K*(K-1)      transition parameters
                     + 2*K          emission parameters (mean + variance)
                     + (K-1)        initial distribution
                   = K^2 + 2K - 1

        Returns
        -------
        dict with keys: log_likelihood, bic, n_components, converged,
                        n_iter_run.
        """
        self._check_fitted()
        X   = self._build_features(self.returns_)
        ll  = float(self.model_.score(X))
        n   = len(self.returns_)
        K   = self.n_components
        n_params = K * (K - 1) + 2 * K + (K - 1)   # = K^2 + 2K - 1
        bic = -2 * ll + n_params * np.log(n)

        return {
            'log_likelihood' : ll,
            'bic'            : float(bic),
            'n_components'   : K,
            'converged'      : bool(self.model_.monitor_.converged),
            'n_iter_run'     : len(self.model_.monitor_.history),
        }

    def regime_stats(self) -> pd.DataFrame:
        """
        Compute per-regime descriptive statistics.

        Requires fit() to have been called. Uses self.returns_ and
        the Viterbi-decoded regime labels from predict().

        Returns
        -------
        pd.DataFrame with one row per regime and columns:
            Regime, Count, Pct (%), Mean Return, Daily Std,
            Ann. Vol, Avg Duration (days)
        """
        self._check_fitted()
        returns = self.returns_
        regimes = self.predict(returns)

        if self.n_components == 2:
            regime_labels = {0: 'Low Volatility', 1: 'High Volatility'}
        elif self.n_components == 3:
            regime_labels = {0: 'Low Volatility', 1: 'Medium Volatility', 2: 'High Volatility'}
        elif self.n_components == 4:
            regime_labels = {0: 'Low Volatility', 1: 'Medium-Low Volatility',
                         2: 'Medium-High Volatility', 3: 'High Volatility'}
        else:
            regime_labels = {k: f'State {k}' for k in range(self.n_components)}

        count     = regimes.value_counts().sort_index()
        pct       = regimes.value_counts(normalize=True).sort_index() * 100
        runs      = regimes.groupby(
            (regimes != regimes.shift()).cumsum()
        ).agg(['first', 'count'])
        avg_dur   = runs.groupby('first')['count'].mean()

        rows = []
        for k in range(self.n_components):
            r = returns[regimes == k]
            rows.append({
                'Regime'              : regime_labels.get(k, f'State {k}'),
                'Count'               : int(count[k]),
                'Pct (%)'             : round(float(pct[k]), 1),
                'Mean Return'         : f'{r.mean():.4f}',
                'Daily Std'           : f'{r.std():.4f}',
                'Ann. Vol'            : f'{r.std() * np.sqrt(252):.2%}',
                'Avg Duration (days)' : f'{avg_dur[k]:.1f}',
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # BIC convenience
    # ------------------------------------------------------------------

    def bic(self) -> float:
        """
        Return the Bayesian Information Criterion for the fitted model.

        Convenience wrapper around summary()['bic'].

        Returns
        -------
        float
        """
        return self.summary()['bic']

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """
        Pickle the fitted RegimeHMM to disk.

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
        print(f'RegimeHMM saved → {path}')

    @classmethod
    def load(cls, path) -> 'RegimeHMM':
        """
        Load a pickled RegimeHMM from disk.

        Parameters
        ----------
        path : str or Path
            Path to a .pkl file saved by RegimeHMM.save().

        Returns
        -------
        RegimeHMM with model_, returns_, and remap_ restored.

        Raises
        ------
        FileNotFoundError if path does not exist.
        TypeError if the unpickled object is not a RegimeHMM.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'No model file found at {path}')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Expected RegimeHMM, got {type(obj)}')
        print(f'RegimeHMM loaded ← {path}')
        return obj

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    def log_to_mlflow(self, run_name: str | None = None) -> None:
        """
        Log parameters, metrics, and the fitted model to MLflow.

        Logs all keys from summary() as metrics, n_components /
        n_iter / random_state as params, and saves the model as a
        pickle artifact under 'model/'.

        Parameters
        ----------
        run_name : str or None
            MLflow run name. Defaults to 'RegimeHMM_{n_components}states'.
        """
        self._check_fitted()
        run_name = run_name or f'RegimeHMM_{self.n_components}states'
        s = self.summary()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('n_components', self.n_components)
            mlflow.log_param('n_iter',       self.n_iter)
            mlflow.log_param('random_state', self.random_state)

            for key, value in s.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

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
            f'RegimeHMM(n_components={self.n_components}, '
            f'status={status!r})'
        )
