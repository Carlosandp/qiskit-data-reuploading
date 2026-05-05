"""Tests for DataReuploadingClassifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons

from qdr.models import DataReuploadingClassifier


@pytest.fixture
def small_binary():
    X, y = make_moons(n_samples=20, noise=0.1, random_state=0)
    return X, y


@pytest.fixture
def small_multiclass():
    X, y = make_classification(
        n_samples=30,
        n_features=2,
        n_classes=3,
        n_informative=2,
        n_redundant=0,
        random_state=1,
    )
    return X, y


class TestDataReuploadingClassifierAPI:
    def test_fit_returns_self(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        result = model.fit(X, y)
        assert result is model

    def test_classes_set_after_fit(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        assert set(model.classes_) == {0, 1}

    def test_n_features_in(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        assert model.n_features_in_ == 2

    def test_weights_shape(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=2, n_layers=3, max_iter=5)
        model.fit(X, y)
        # 2 qubits × 3 layers × 3 rotations = 18
        assert model.weights_.shape == (18,)

    def test_predict_shape(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_classes_valid(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape_binary(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_score_in_range(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_loss_history_populated(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=10)
        model.fit(X, y)
        assert isinstance(model.loss_history_, list)
        assert len(model.loss_history_) > 0

    def test_string_labels(self, small_binary):
        X, y = small_binary
        y_str = np.where(y == 0, "cat", "dog")
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y_str)
        preds = model.predict(X)
        assert set(preds).issubset({"cat", "dog"})

    def test_get_params_set_params(self):
        model = DataReuploadingClassifier(n_qubits=3, n_layers=4, optimizer="SPSA")
        params = model.get_params()
        assert params["n_qubits"] == 3
        assert params["optimizer"] == "SPSA"
        model.set_params(n_layers=2)
        assert model.n_layers == 2


class TestDataReuploadingClassifierOptimizers:
    @pytest.mark.parametrize("opt", ["COBYLA", "SPSA"])
    def test_optimizers_run(self, opt, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, optimizer=opt, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_adam_optimizer(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(
            n_qubits=1, n_layers=1, optimizer="ADAM", max_iter=5, learning_rate=0.05
        )
        model.fit(X, y)
        assert model.weights_ is not None

    def test_invalid_optimizer_raises(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, optimizer="SGD", max_iter=3)
        with pytest.raises(ValueError, match="optimizer"):
            model.fit(X, y)


class TestDataReuploadingClassifierPersistence:
    def test_save_load(self, small_binary, tmp_path):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5, seed=0)
        model.fit(X, y)
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = DataReuploadingClassifier.load(path)
        np.testing.assert_array_equal(model.weights_, loaded.weights_)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_not_fitted_raises(self, small_binary, tmp_path):
        from sklearn.exceptions import NotFittedError

        model = DataReuploadingClassifier(n_qubits=1, n_layers=2)
        X, y = small_binary
        with pytest.raises(NotFittedError):
            model.predict(X)
