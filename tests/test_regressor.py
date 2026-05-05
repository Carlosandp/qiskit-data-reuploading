"""Tests for DataReuploadingRegressor."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from qdr.models import DataReuploadingRegressor


@pytest.fixture
def small_regression():
    X, y = make_regression(n_samples=20, n_features=2, noise=5.0, random_state=0)
    return X, y


class TestDataReuploadingRegressor:
    def test_fit_returns_self(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        assert model.fit(X, y) is model

    def test_predict_shape(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_in_target_range(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        # Predictions should be within the target range seen during training
        assert preds.min() >= y.min() - 1e-6
        assert preds.max() <= y.max() + 1e-6

    def test_y_min_max_stored(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=3)
        model.fit(X, y)
        assert model.y_min_ == pytest.approx(y.min())
        assert model.y_max_ == pytest.approx(y.max())

    def test_score_returns_float(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        score = model.score(X, y)
        assert isinstance(score, float)

    def test_save_load(self, small_regression, tmp_path):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5, seed=0)
        model.fit(X, y)
        path = tmp_path / "regressor.pkl"
        model.save(path)
        loaded = DataReuploadingRegressor.load(path)
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_n_features_in(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=3)
        model.fit(X, y)
        assert model.n_features_in_ == 2

    def test_loss_history(self, small_regression):
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=8)
        model.fit(X, y)
        assert len(model.loss_history_) > 0
