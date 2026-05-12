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
        n_clusters_per_class=1,
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

    def test_multiclass_requires_one_qubit_per_class(self, small_multiclass):
        X, y = small_multiclass
        model = DataReuploadingClassifier(n_qubits=2, n_layers=1, max_iter=1)
        with pytest.raises(
            ValueError,
            match="Para 3 clases se necesitan al menos 3 qubits. Actual: n_qubits=2.",
        ):
            model.fit(X, y)

    def test_fit_rejects_non_finite_X(self, small_binary):
        X, y = small_binary
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            model.fit(X_bad, y)

    def test_fit_rejects_y_length_mismatch(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="X and y have inconsistent lengths"):
            model.fit(X, y[:-1])

    def test_fit_rejects_multidimensional_y(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="y must be a 1D array"):
            model.fit(X, y.reshape(-1, 1))

    def test_fit_rejects_missing_y_label(self, small_binary):
        X, y = small_binary
        y_bad = y.astype(float)
        y_bad[0] = np.nan
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="y contains NaN, Inf, or None labels"):
            model.fit(X, y_bad)

    def test_fit_rejects_single_class(self, small_binary):
        X, _ = small_binary
        y = np.zeros(X.shape[0], dtype=int)
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="Classifier requires at least 2 classes"):
            model.fit(X, y)

    def test_fit_rejects_more_features_than_encoding_slots(self):
        X = np.zeros((6, 4))
        y = np.array([0, 1, 0, 1, 0, 1])
        model = DataReuploadingClassifier(
            n_qubits=1,
            n_layers=1,
            encoding="rx_ry_rz",
            max_iter=1,
        )
        with pytest.raises(ValueError, match="n_features_in_=4 exceeds the number of encoding slots"):
            model.fit(X, y)

    def test_invalid_encoding_error_comes_from_circuit_validation(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, encoding="bad", max_iter=1)
        with pytest.raises(ValueError, match="encoding must be one of"):
            model.fit(X, y)

    def test_invalid_n_qubits_error_comes_from_circuit_validation(self, small_multiclass):
        X, y = small_multiclass
        model = DataReuploadingClassifier(n_qubits="bad", n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
            model.fit(X, y)

    def test_predict_proba_rejects_feature_mismatch(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3)
        model.fit(X, y)
        X_bad = np.zeros((len(X), X.shape[1] + 1))
        with pytest.raises(
            ValueError,
            match=f"X has {X.shape[1] + 1} features, but this classifier was fitted with "
            f"n_features_in_={X.shape[1]}.",
        ):
            model.predict_proba(X_bad)


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

    def test_backend_argument_is_not_silently_ignored(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(
            n_qubits=1,
            n_layers=1,
            backend="ibm_brisbane",
            max_iter=1,
        )
        with pytest.raises(ValueError, match="Use qdr.hardware.run_on_ibm_backend"):
            model.fit(X, y)

    def test_invalid_shots_raises(self, small_binary):
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, shots=0, max_iter=1)
        with pytest.raises(ValueError, match="shots must be None or a positive integer"):
            model.fit(X, y)

    def test_shots_configures_aer_estimator(self):
        model = DataReuploadingClassifier(shots=123, seed=7)
        estimator = model._build_estimator()
        assert estimator.options.run_options["shots"] == 123
        assert estimator.options.run_options["seed_simulator"] == 7

    def test_callback_does_not_recompute_loss(self, small_binary, monkeypatch):
        from qdr.training.optimizers import OptimizeResult

        import qdr.models.classifier as classifier_module

        class SingleEvalCOBYLA:
            def __init__(self, *args, **kwargs):
                pass

            def minimize(self, fun, x0, callback=None):
                val = fun(x0)
                if callback is not None:
                    callback(x0)
                return OptimizeResult(x=x0, fun=val, nit=1, loss_history=[])

        class CountingClassifier(DataReuploadingClassifier):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.loss_calls = 0

            def _loss(self, weights, X, y_mapped, observables):
                self.loss_calls += 1
                return 0.25

        monkeypatch.setattr(classifier_module, "COBYLA", SingleEvalCOBYLA)
        X, y = small_binary
        model = CountingClassifier(n_qubits=1, n_layers=1, optimizer="COBYLA", max_iter=1)
        model.fit(X, y)

        assert model.loss_calls == 1
        assert model.loss_history_ == [0.25]

    def test_multiclass_gradient_sums_class_gradients(self, monkeypatch):
        from qiskit.quantum_info import SparsePauliOp

        import qdr.training.gradients as gradients

        created_values = []

        class FakeGradient:
            def __init__(self, *args, **kwargs):
                self.value = len(created_values) + 1
                created_values.append(self.value)

            def compute(self, weights, X, y):
                return np.full_like(weights, float(self.value))

        monkeypatch.setattr(gradients, "ParameterShiftGradient", FakeGradient)

        model = DataReuploadingClassifier(n_qubits=3)
        model._circuit_ = object()
        model._estimator_ = object()
        observables = [SparsePauliOp("IIZ"), SparsePauliOp("IZI"), SparsePauliOp("ZII")]
        grad_fn = model._make_grad_fn(
            X=np.zeros((2, 1)),
            y_mapped=np.zeros((2, 3)),
            observables=observables,
        )

        grad = grad_fn(np.ones(4))
        np.testing.assert_allclose(grad, np.full(4, 6.0))

    def test_multiclass_loss_is_sum_of_class_mse(self):
        class FixedEvsClassifier(DataReuploadingClassifier):
            def _evaluate_batch(self, weights, X, observables):
                return np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])

        model = FixedEvsClassifier(n_qubits=3)
        y_mapped = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
        loss = model._loss(np.zeros(1), np.zeros((2, 1)), y_mapped, [object(), object(), object()])
        # Class-wise MSEs are [0.5, 0.0, 0.5]; multiclass objective is their sum.
        assert loss == pytest.approx(1.0)

    def test_predict_proba_clips_binary_expectations(self):
        class OutOfRangeClassifier(DataReuploadingClassifier):
            def _evaluate_batch(self, weights, X, observables):
                return np.array([-1.5, 1.5])

        model = OutOfRangeClassifier()
        model.weights_ = np.zeros(1)
        model.n_features_in_ = 1
        model.classes_ = np.array([0, 1])
        model._observables_ = [object()]
        proba = model.predict_proba(np.zeros((2, 1)))
        np.testing.assert_allclose(proba, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_binary_gradient_does_not_do_dead_evaluation(self, monkeypatch):
        from qiskit.quantum_info import SparsePauliOp

        import qdr.training.gradients as gradients

        class FakeGradient:
            def __init__(self, *args, **kwargs):
                pass

            def compute(self, weights, X, y):
                return np.ones_like(weights)

        class GuardedClassifier(DataReuploadingClassifier):
            def _evaluate_batch(self, weights, X, observables):
                raise AssertionError("_evaluate_batch should not be called before compute()")

        monkeypatch.setattr(gradients, "ParameterShiftGradient", FakeGradient)
        model = GuardedClassifier(n_qubits=1)
        model._circuit_ = object()
        model._estimator_ = object()
        grad_fn = model._make_grad_fn(
            X=np.zeros((2, 1)),
            y_mapped=np.zeros(2),
            observables=[SparsePauliOp("Z")],
        )

        np.testing.assert_allclose(grad_fn(np.ones(3)), np.ones(3))


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
