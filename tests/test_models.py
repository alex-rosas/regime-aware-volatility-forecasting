"""
tests/test_models.py
--------------------
Tests for src/models/ — VolatilityModel, RegimeHMM,
HybridVolatilityModel, ConformalPredictor.

All tests use synthetic data — no real data files required.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.models.garch import VolatilityModel
from src.models.hmm import RegimeHMM
from src.models.hybrid import HybridVolatilityModel, qlike, diebold_mariano
from src.models.conformal import ConformalPredictor


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_returns():
    """500 observations of synthetic log-returns."""
    np.random.seed(42)
    n = 500
    return pd.Series(
        np.random.normal(0, 0.006, n),
        index=pd.date_range('2000-01-01', periods=n, freq='B'),
        name='MXN_USD_log_return',
    )


@pytest.fixture
def synthetic_dataset(synthetic_returns):
    """Aligned dataset matching HybridVolatilityModel.build_features() input."""
    np.random.seed(42)
    n = len(synthetic_returns)
    idx = synthetic_returns.index
    return pd.DataFrame({
        'MXN_USD_log_return' : synthetic_returns.values,
        'VIXCLS'             : np.random.uniform(10, 40, n),
        'T10Y2Y'             : np.random.uniform(-1, 2, n),
        'regime'             : np.random.randint(0, 3, n),
        'sigma_garch_ann'    : np.random.uniform(0.05, 0.30, n),
        'sigma_egarch_ann'   : np.random.uniform(0.05, 0.30, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# VolatilityModel
# ---------------------------------------------------------------------------

class TestVolatilityModel:

    def test_garch_fits(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        assert model.result_ is not None

    def test_egarch_fits(self, synthetic_returns):
        model = VolatilityModel('EGARCH').fit(synthetic_returns)
        assert model.result_ is not None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            VolatilityModel('INVALID')

    def test_check_fitted_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            VolatilityModel('GARCH').summary()

    def test_summary_keys(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        s     = model.summary()
        for key in ['omega', 'alpha', 'beta', 'nu', 'persistence',
                    'aic', 'bic', 'log_likelihood']:
            assert key in s, f'Missing key: {key}'

    def test_to_frame_shape(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        df    = model.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0

    def test_conditional_volatility_length(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        vol   = model.conditional_volatility()
        assert len(vol) <= len(synthetic_returns)
        assert (vol > 0).all()

    def test_conditional_volatility_annualised(self, synthetic_returns):
        """Annualised vol should be in a plausible range for synthetic data."""
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        vol   = model.conditional_volatility()
        assert vol.mean() < 1.0   # not > 100% annualised vol

    def test_std_resid_length(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        resid = model.std_resid()
        assert len(resid) <= len(synthetic_returns)

    def test_persistence_garch_range(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        p     = model.summary()['persistence']
        assert 0 < p < 1

    def test_save_load_roundtrip(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'garch.pkl'
            model.save(path)
            loaded = VolatilityModel.load(path)
        assert loaded.summary()['bic'] == model.summary()['bic']

    def test_repr_fitted(self, synthetic_returns):
        model = VolatilityModel('GARCH').fit(synthetic_returns)
        assert 'fitted' in repr(model)

    def test_repr_unfitted(self):
        assert 'unfitted' in repr(VolatilityModel('GARCH'))


# ---------------------------------------------------------------------------
# RegimeHMM
# ---------------------------------------------------------------------------

class TestRegimeHMM:

    def test_fits(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        assert hmm.model_ is not None

    def test_check_fitted_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            RegimeHMM().predict(pd.Series([0.01, 0.02]))

    def test_predict_length(self, synthetic_returns):
        hmm     = RegimeHMM(n_components=3).fit(synthetic_returns)
        regimes = hmm.predict(synthetic_returns)
        assert len(regimes) == len(synthetic_returns)

    def test_predict_values_in_range(self, synthetic_returns):
        hmm     = RegimeHMM(n_components=3).fit(synthetic_returns)
        regimes = hmm.predict(synthetic_returns)
        assert set(regimes.unique()).issubset({0, 1, 2})

    def test_predict_index_aligned(self, synthetic_returns):
        hmm     = RegimeHMM(n_components=3).fit(synthetic_returns)
        regimes = hmm.predict(synthetic_returns)
        assert (regimes.index == synthetic_returns.index).all()

    def test_remap_sorted_by_variance(self, synthetic_returns):
        """State 0 should have lowest emission variance."""
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        variances = [float(hmm.model_.covars_[k][0, 0]) for k in range(3)]
        raw_order = sorted(range(3), key=lambda k: variances[k])
        assert hmm.remap_[raw_order[0]] == 0

    def test_transition_matrix_shape(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        A   = hmm.transition_matrix()
        assert A.shape == (3, 3)

    def test_transition_matrix_rows_sum_to_one(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        A   = hmm.transition_matrix()
        assert np.allclose(A.values.sum(axis=1), 1.0, atol=1e-6)

    def test_summary_keys(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        s   = hmm.summary()
        for key in ['log_likelihood', 'bic', 'converged', 'n_iter_run']:
            assert key in s

    def test_bic_returns_float(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        assert isinstance(hmm.bic(), float)

    def test_regime_stats_shape(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        df  = hmm.regime_stats()
        assert df.shape[0] == 3

    def test_save_load_roundtrip(self, synthetic_returns):
        hmm = RegimeHMM(n_components=3).fit(synthetic_returns)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hmm.pkl'
            hmm.save(path)
            loaded = RegimeHMM.load(path)
        assert loaded.summary()['bic'] == hmm.summary()['bic']


# ---------------------------------------------------------------------------
# HybridVolatilityModel
# ---------------------------------------------------------------------------

class TestHybridVolatilityModel:

    def test_build_features_shape(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        assert X.shape[1] == 8
        assert len(X) == len(y)

    def test_build_features_no_nans(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_build_features_does_not_mutate(self, synthetic_dataset):
        original_cols = list(synthetic_dataset.columns)
        hybrid = HybridVolatilityModel()
        hybrid.build_features(synthetic_dataset)
        assert list(synthetic_dataset.columns) == original_cols

    def test_split_sizes(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        X_train, y_train, X_val, y_val, X_test, y_test = \
            HybridVolatilityModel.split(X, y)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)

    def test_split_no_overlap(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        X_train, _, X_val, _, X_test, _ = HybridVolatilityModel.split(X, y)
        assert X_train.index[-1] < X_val.index[0]
        assert X_val.index[-1]   < X_test.index[0]

    def test_fit_and_predict(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        X_train, y_train, X_val, y_val, X_test, y_test = \
            HybridVolatilityModel.split(X, y)
        hybrid.fit(X_train, y_train, X_val, y_val)
        preds = hybrid.predict(X_test)
        assert len(preds) == len(X_test)
        assert not np.isnan(preds).any()

    def test_summary_keys(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        X_train, y_train, X_val, y_val, _, _ = \
            HybridVolatilityModel.split(X, y)
        hybrid.fit(X_train, y_train, X_val, y_val)
        s = hybrid.summary()
        for key in ['best_iteration', 'train_rmse', 'val_rmse']:
            assert key in s

    def test_check_fitted_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            HybridVolatilityModel().predict(pd.DataFrame())

    def test_save_load_roundtrip(self, synthetic_dataset):
        hybrid = HybridVolatilityModel()
        X, y   = hybrid.build_features(synthetic_dataset)
        X_train, y_train, X_val, y_val, X_test, _ = \
            HybridVolatilityModel.split(X, y)
        hybrid.fit(X_train, y_train, X_val, y_val)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hybrid.pkl'
            hybrid.save(path)
            loaded = HybridVolatilityModel.load(path)
        assert loaded.summary()['best_iteration'] == hybrid.summary()['best_iteration']


# ---------------------------------------------------------------------------
# ConformalPredictor
# ---------------------------------------------------------------------------

class TestConformalPredictor:

    @pytest.fixture
    def calibrated_cp(self):
        np.random.seed(42)
        y_true = np.abs(np.random.normal(0, 0.005, 200))
        y_pred = y_true + np.random.normal(0, 0.002, 200)
        cp = ConformalPredictor()
        cp.calibrate(y_true, y_pred)
        return cp, y_true, y_pred

    def test_calibrate_stores_residuals(self, calibrated_cp):
        cp, _, _ = calibrated_cp
        assert cp.cal_residuals_ is not None
        assert cp.n_cal_ == 200

    def test_check_calibrated_raises_before_calibrate(self):
        with pytest.raises(RuntimeError):
            ConformalPredictor().quantile(0.10)

    def test_quantile_increases_with_coverage(self, calibrated_cp):
        cp, _, _ = calibrated_cp
        q80 = cp.quantile(0.20)
        q90 = cp.quantile(0.10)
        q95 = cp.quantile(0.05)
        assert q80 < q90 < q95

    def test_predict_shapes(self, calibrated_cp):
        cp, _, y_pred = calibrated_cp
        lower, upper = cp.predict(y_pred, alpha=0.10)
        assert lower.shape == y_pred.shape
        assert upper.shape == y_pred.shape

    def test_predict_lower_less_than_upper(self, calibrated_cp):
        cp, _, y_pred = calibrated_cp
        lower, upper = cp.predict(y_pred, alpha=0.10)
        assert (lower <= upper).all()

    def test_coverage_at_least_nominal(self, calibrated_cp):
        cp, y_true, y_pred = calibrated_cp
        for alpha in [0.20, 0.10, 0.05]:
            cov = cp.coverage(y_true, y_pred, alpha)
            assert cov >= (1 - alpha) - 0.05  # allow 5% tolerance

    def test_predict_all_keys(self, calibrated_cp):
        cp, _, y_pred = calibrated_cp
        results = cp.predict_all(y_pred)
        assert set(results.keys()) == {'0.2', '0.1', '0.05'}

    def test_coverage_summary_shape(self, calibrated_cp):
        cp, y_true, y_pred = calibrated_cp
        df = cp.coverage_summary(y_true, y_pred)
        assert df.shape[0] == 3
        assert 'Empirical Coverage' in df.columns

    def test_save_load_roundtrip(self, calibrated_cp):
        cp, y_true, y_pred = calibrated_cp
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'conformal.pkl'
            cp.save(path)
            loaded = ConformalPredictor.load(path)
        assert loaded.n_cal_ == cp.n_cal_
        assert abs(loaded.quantile(0.10) - cp.quantile(0.10)) < 1e-10


# ---------------------------------------------------------------------------
# standalone functions: qlike and diebold_mariano
# ---------------------------------------------------------------------------

class TestEvaluationFunctions:

    def test_qlike_positive(self):
        y_true = np.array([0.01, 0.02, 0.015])
        y_pred = np.array([0.01, 0.02, 0.015])
        assert qlike(y_true, y_pred) >= 0

    def test_qlike_perfect_forecast_is_zero(self):
        y = np.array([0.01, 0.02, 0.015])
        assert abs(qlike(y, y)) < 1e-10

    def test_qlike_handles_zeros(self):
        y_true = np.array([0.0, 0.01, 0.02])
        y_pred = np.array([0.01, 0.01, 0.02])
        result = qlike(y_true, y_pred)
        assert np.isfinite(result)

    def test_diebold_mariano_returns_tuple(self):
        np.random.seed(0)
        y     = np.random.normal(0, 0.01, 100)
        pred1 = y + np.random.normal(0, 0.005, 100)
        pred2 = y + np.random.normal(0, 0.010, 100)
        stat, pval = diebold_mariano(y, pred1, pred2)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_diebold_mariano_pvalue_range(self):
        np.random.seed(0)
        y     = np.random.normal(0, 0.01, 100)
        pred1 = y + np.random.normal(0, 0.005, 100)
        pred2 = y + np.random.normal(0, 0.010, 100)
        _, pval = diebold_mariano(y, pred1, pred2)
        assert 0.0 <= pval <= 1.0

    def test_diebold_mariano_better_model_negative_stat(self):
        """pred1 with smaller errors should give negative DM stat."""
        np.random.seed(0)
        y     = np.random.normal(0, 0.01, 200)
        pred1 = y + np.random.normal(0, 0.001, 200)  # better
        pred2 = y + np.random.normal(0, 0.010, 200)  # worse
        stat, _ = diebold_mariano(y, pred1, pred2)
        assert stat < 0
