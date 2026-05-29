"""Tests for DataReuploadingRegressor."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from qdr.models import DataReuploadingRegressor


@pytest.fixture
def small_regression():
    """Return a small regression dataset with 20 samples and 2 features."""
    X, y = make_regression(n_samples=20, n_features=2, noise=5.0, random_state=0)
    return X, y


class TestDataReuploadingRegressor:
    """Tests for the DataReuploadingRegressor fit/predict API and validation."""

    def test_fit_returns_self(self, small_regression):
        """fit() returns the regressor instance for method chaining."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        assert model.fit(X, y) is model

    def test_predict_shape(self, small_regression):
        """predict() returns an array with one value per sample."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_in_target_range(self, small_regression):
        """Predictions fall within the target range seen during training."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        # Predictions should be within the target range seen during training
        assert preds.min() >= y.min() - 1e-6
        assert preds.max() <= y.max() + 1e-6

    def test_y_min_max_stored(self, small_regression):
        """fit() stores y_min_ and y_max_ from the training targets."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=3)
        model.fit(X, y)
        assert model.y_min_ == pytest.approx(y.min())
        assert model.y_max_ == pytest.approx(y.max())

    def test_score_returns_float(self, small_regression):
        """score() returns a float R² value."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        score = model.score(X, y)
        assert isinstance(score, float)

    def test_save_load(self, small_regression, tmp_path):
        """Saved and reloaded model produces identical predictions."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=2, max_iter=5, seed=0)
        model.fit(X, y)
        path = tmp_path / "regressor.pkl"
        model.save(path)
        loaded = DataReuploadingRegressor.load(path)
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_n_features_in(self, small_regression):
        """fit() records n_features_in_ from the training data."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=3)
        model.fit(X, y)
        assert model.n_features_in_ == 2

    def test_loss_history(self, small_regression):
        """loss_history_ is non-empty after training."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=8)
        model.fit(X, y)
        assert len(model.loss_history_) > 0

    def test_backend_argument_is_not_silently_ignored(self, small_regression):
        """fit() raises ValueError when a named backend is passed instead of using hardware API."""
        X, y = small_regression
        model = DataReuploadingRegressor(
            n_qubits=1,
            n_layers=1,
            backend="ibm_brisbane",
            max_iter=1,
        )
        with pytest.raises(ValueError, match="Use qdr.hardware.run_on_ibm_backend"):
            model.fit(X, y)

    def test_fit_rejects_non_finite_X(self, small_regression):
        """fit() raises ValueError when X contains infinite values."""
        X, y = small_regression
        X_bad = X.copy()
        X_bad[0, 0] = np.inf
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            model.fit(X_bad, y)

    def test_fit_rejects_non_finite_y(self, small_regression):
        """fit() raises ValueError when y contains NaN values."""
        X, y = small_regression
        y_bad = y.copy()
        y_bad[0] = np.nan
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="y contains NaN or Inf"):
            model.fit(X, y_bad)

    def test_fit_rejects_y_length_mismatch(self, small_regression):
        """fit() raises ValueError when X and y have different sample counts."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="X and y have inconsistent lengths"):
            model.fit(X, y[:-1])

    def test_fit_rejects_multidimensional_y(self, small_regression):
        """fit() raises ValueError when y is a 2D column vector instead of 1D."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="y must be a 1D array"):
            model.fit(X, y.reshape(-1, 1))

    def test_fit_rejects_more_features_than_encoding_slots(self):
        """fit() raises ValueError when n_features exceeds available encoding slots."""
        X = np.zeros((6, 4))
        y = np.arange(6, dtype=float)
        model = DataReuploadingRegressor(
            n_qubits=1,
            n_layers=1,
            encoding="rx_ry_rz",
            max_iter=1,
        )
        with pytest.raises(ValueError, match="n_features_in_=4 exceeds the number of encoding slots"):
            model.fit(X, y)

    def test_invalid_encoding_error_comes_from_circuit_validation(self, small_regression):
        """fit() raises ValueError from circuit validation when encoding name is unknown."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, encoding="bad", max_iter=1)
        with pytest.raises(ValueError, match="encoding must be one of"):
            model.fit(X, y)

    def test_predict_rejects_feature_mismatch(self, small_regression):
        """predict() raises ValueError when X has more features than seen during fit()."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, max_iter=3)
        model.fit(X, y)
        X_bad = np.zeros((len(X), X.shape[1] + 1))
        with pytest.raises(
            ValueError,
            match=f"X has {X.shape[1] + 1} features, but this regressor was fitted with "
            f"n_features_in_={X.shape[1]}.",
        ):
            model.predict(X_bad)

    def test_invalid_shots_raises(self, small_regression):
        """fit() raises ValueError when shots is not None or a positive integer."""
        X, y = small_regression
        model = DataReuploadingRegressor(n_qubits=1, n_layers=1, shots=False, max_iter=1)
        with pytest.raises(ValueError, match="shots must be None or a positive integer"):
            model.fit(X, y)

    def test_shots_configures_aer_estimator(self):
        """_build_estimator() sets shots and seed_simulator on the Aer estimator."""
        model = DataReuploadingRegressor(shots=321, seed=11)
        estimator = model._build_estimator()
        assert estimator.options.run_options["shots"] == 321
        assert estimator.options.run_options["seed_simulator"] == 11

    def test_scale_to_target_clips_out_of_range_expectations(self):
        """_scale_to_target() linearly maps [-1, 1] expectations to [y_min_, y_max_]."""
        model = DataReuploadingRegressor()
        model.y_min_ = -2.0
        model.y_max_ = 3.0
        scaled = model._scale_to_target(np.array([-2.0, 0.0, 2.0]))
        np.testing.assert_allclose(scaled, np.array([-2.0, 0.5, 3.0]))

    def test_callback_does_not_recompute_loss(self, small_regression, monkeypatch):
        """The optimizer callback records the loss without triggering an extra circuit evaluation."""
        from qdr.training.optimizers import OptimizeResult

        import qdr.models.regressor as regressor_module

        class SingleEvalCOBYLA:
            def __init__(self, *args, **kwargs):
                pass

            def minimize(self, fun, x0, callback=None):
                val = fun(x0)
                if callback is not None:
                    callback(x0)
                return OptimizeResult(x=x0, fun=val, nit=1, loss_history=[])

        class CountingRegressor(DataReuploadingRegressor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.eval_calls = 0

            def _evaluate_batch(self, weights, X):
                self.eval_calls += 1
                return np.zeros(X.shape[0])

        monkeypatch.setattr(regressor_module, "COBYLA", SingleEvalCOBYLA)
        X, y = small_regression
        model = CountingRegressor(n_qubits=1, n_layers=1, optimizer="COBYLA", max_iter=1)
        model.fit(X, y)

        assert model.eval_calls == 1
        assert len(model.loss_history_) == 1
