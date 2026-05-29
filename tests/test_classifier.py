"""Tests for DataReuploadingClassifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons

from qdr.models import DataReuploadingClassifier


@pytest.fixture
def small_binary():
    """Return a small two-moons binary dataset with 20 samples."""
    X, y = make_moons(n_samples=20, noise=0.1, random_state=0)
    return X, y


@pytest.fixture
def small_multiclass():
    """Return a small three-class dataset with 30 samples and 2 features."""
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
    """Tests for the sklearn-compatible fit/predict/score API of the classifier."""

    def test_fit_returns_self(self, small_binary):
        """fit() returns the classifier instance for method chaining."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        result = model.fit(X, y)
        assert result is model

    def test_classes_set_after_fit(self, small_binary):
        """fit() sets classes_ to the unique class labels."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        assert set(model.classes_) == {0, 1}

    def test_n_features_in(self, small_binary):
        """fit() records n_features_in_ from the training data."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        assert model.n_features_in_ == 2

    def test_weights_shape(self, small_binary):
        """Trained weights have shape (n_qubits * n_layers * n_rotations,)."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=2, n_layers=3, max_iter=5)
        model.fit(X, y)
        # 2 qubits × 3 layers × 3 rotations = 18
        assert model.weights_.shape == (18,)

    def test_predict_shape(self, small_binary):
        """predict() returns an array with one label per sample."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_classes_valid(self, small_binary):
        """predict() returns only labels seen during training."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape_binary(self, small_binary):
        """predict_proba() returns an (n_samples, 2) array for binary tasks."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, small_binary):
        """Each row of predict_proba() sums to 1."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_score_in_range(self, small_binary):
        """score() returns a value in [0, 1]."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_loss_history_populated(self, small_binary):
        """loss_history_ is a non-empty list after training."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=10)
        model.fit(X, y)
        assert isinstance(model.loss_history_, list)
        assert len(model.loss_history_) > 0

    def test_string_labels(self, small_binary):
        """Classifier handles string class labels via LabelEncoder."""
        X, y = small_binary
        y_str = np.where(y == 0, "cat", "dog")
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5)
        model.fit(X, y_str)
        preds = model.predict(X)
        assert set(preds).issubset({"cat", "dog"})

    def test_get_params_set_params(self):
        """get_params() and set_params() follow the sklearn estimator contract."""
        model = DataReuploadingClassifier(n_qubits=3, n_layers=4, optimizer="SPSA")
        params = model.get_params()
        assert params["n_qubits"] == 3
        assert params["optimizer"] == "SPSA"
        model.set_params(n_layers=2)
        assert model.n_layers == 2

    def test_multiclass_requires_one_qubit_per_class(self, small_multiclass):
        """fit() raises ValueError when n_qubits < n_classes for multiclass."""
        X, y = small_multiclass
        model = DataReuploadingClassifier(n_qubits=2, n_layers=1, max_iter=1)
        with pytest.raises(
            ValueError,
            match="Para 3 clases se necesitan al menos 3 qubits. Actual: n_qubits=2.",
        ):
            model.fit(X, y)

    def test_fit_rejects_non_finite_X(self, small_binary):
        """fit() raises ValueError when X contains NaN."""
        X, y = small_binary
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            model.fit(X_bad, y)

    def test_fit_rejects_y_length_mismatch(self, small_binary):
        """fit() raises ValueError when X and y have inconsistent lengths."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="X and y have inconsistent lengths"):
            model.fit(X, y[:-1])

    def test_fit_rejects_multidimensional_y(self, small_binary):
        """fit() raises ValueError when y is 2D."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="y must be a 1D array"):
            model.fit(X, y.reshape(-1, 1))

    def test_fit_rejects_missing_y_label(self, small_binary):
        """fit() raises ValueError when y contains NaN."""
        X, y = small_binary
        y_bad = y.astype(float)
        y_bad[0] = np.nan
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="y contains NaN, Inf, or None labels"):
            model.fit(X, y_bad)

    def test_fit_rejects_single_class(self, small_binary):
        """fit() raises ValueError when y has only one unique class."""
        X, _ = small_binary
        y = np.zeros(X.shape[0], dtype=int)
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="Classifier requires at least 2 classes"):
            model.fit(X, y)

    def test_fit_rejects_more_features_than_encoding_slots(self):
        """fit() raises ValueError when features exceed encoding capacity."""
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
        """fit() propagates ValueError from circuit when encoding is invalid."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, encoding="bad", max_iter=1)
        with pytest.raises(ValueError, match="encoding must be one of"):
            model.fit(X, y)

    def test_invalid_n_qubits_error_comes_from_circuit_validation(self, small_multiclass):
        """fit() propagates ValueError from circuit when n_qubits is invalid."""
        X, y = small_multiclass
        model = DataReuploadingClassifier(n_qubits="bad", n_layers=1, max_iter=1)
        with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
            model.fit(X, y)

    def test_predict_proba_rejects_feature_mismatch(self, small_binary):
        """predict_proba() raises ValueError when X has wrong feature count."""
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
    """Tests for optimizer selection and gradient computation in the classifier."""

    @pytest.mark.parametrize("opt", ["COBYLA", "SPSA"])
    def test_optimizers_run(self, opt, small_binary):
        """COBYLA and SPSA optimizers complete training and return predictions."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, optimizer=opt, max_iter=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_adam_optimizer(self, small_binary):
        """ADAM optimizer completes training and sets weights_."""
        X, y = small_binary
        model = DataReuploadingClassifier(
            n_qubits=1, n_layers=1, optimizer="ADAM", max_iter=5, learning_rate=0.05
        )
        model.fit(X, y)
        assert model.weights_ is not None

    def test_invalid_optimizer_raises(self, small_binary):
        """An unknown optimizer name raises ValueError during fit()."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, optimizer="SGD", max_iter=3)
        with pytest.raises(ValueError, match="optimizer"):
            model.fit(X, y)

    def test_backend_argument_is_not_silently_ignored(self, small_binary):
        """Passing a backend string raises ValueError directing to the hardware module."""
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
        """shots=0 raises ValueError during fit()."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=1, shots=0, max_iter=1)
        with pytest.raises(ValueError, match="shots must be None or a positive integer"):
            model.fit(X, y)

    def test_shots_configures_aer_estimator(self):
        """Positive shots value builds an AerEstimatorV2 with matching run options."""
        model = DataReuploadingClassifier(shots=123, seed=7)
        estimator = model._build_estimator()
        assert estimator.options.run_options["shots"] == 123
        assert estimator.options.run_options["seed_simulator"] == 7

    def test_callback_does_not_recompute_loss(self, small_binary, monkeypatch):
        """Optimizer callback records the cached loss without triggering an extra circuit evaluation."""
        from qdr.training.optimizers import OptimizeResult

        import qdr.models.classifier as classifier_module

        class SingleEvalCOBYLA:
            """Fake COBYLA that calls the objective exactly once then fires the callback."""

            def __init__(self, *args, **kwargs):
                pass

            def minimize(self, fun, x0, callback=None):
                """Run one objective evaluation and invoke the callback."""
                val = fun(x0)
                if callback is not None:
                    callback(x0)
                return OptimizeResult(x=x0, fun=val, nit=1, loss_history=[])

        class CountingClassifier(DataReuploadingClassifier):
            """Classifier subclass that counts calls to _loss."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.loss_calls = 0

            def _loss(self, weights, X, y_mapped, observables):
                """Increment counter and return a fixed loss value."""
                self.loss_calls += 1
                return 0.25

        monkeypatch.setattr(classifier_module, "COBYLA", SingleEvalCOBYLA)
        X, y = small_binary
        model = CountingClassifier(n_qubits=1, n_layers=1, optimizer="COBYLA", max_iter=1)
        model.fit(X, y)

        assert model.loss_calls == 1
        assert model.loss_history_ == [0.25]

    def test_multiclass_gradient_sums_class_gradients(self, monkeypatch):
        """_make_grad_fn() accumulates per-class PSR gradients for multiclass problems."""
        from qiskit.quantum_info import SparsePauliOp

        import qdr.training.gradients as gradients

        created_values = []

        class FakeGradient:
            """Stub gradient that returns a constant vector equal to its creation index."""

            def __init__(self, *args, **kwargs):
                self.value = len(created_values) + 1
                created_values.append(self.value)

            def compute(self, weights, X, y):
                """Return a gradient array filled with the instance's fixed value."""
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
        """_loss() sums per-class MSE values for multiclass tasks."""
        class FixedEvsClassifier(DataReuploadingClassifier):
            """Classifier with hardcoded expectation values for deterministic loss testing."""

            def _evaluate_batch(self, weights, X, observables):
                """Return fixed per-class expectation values."""
                return np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])

        model = FixedEvsClassifier(n_qubits=3)
        y_mapped = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
        loss = model._loss(np.zeros(1), np.zeros((2, 1)), y_mapped, [object(), object(), object()])
        # Class-wise MSEs are [0.5, 0.0, 0.5]; multiclass objective is their sum.
        assert loss == pytest.approx(1.0)

    def test_predict_proba_clips_binary_expectations(self):
        """predict_proba() clips Z-expectations outside [-1, 1] before converting to probabilities."""
        class OutOfRangeClassifier(DataReuploadingClassifier):
            """Classifier that returns out-of-range expectation values to test clipping."""

            def _evaluate_batch(self, weights, X, observables):
                """Return expectation values outside the valid [-1, 1] range."""
                return np.array([-1.5, 1.5])

        model = OutOfRangeClassifier()
        model.weights_ = np.zeros(1)
        model.n_features_in_ = 1
        model.classes_ = np.array([0, 1])
        model._observables_ = [object()]
        proba = model.predict_proba(np.zeros((2, 1)))
        np.testing.assert_allclose(proba, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_binary_gradient_does_not_do_dead_evaluation(self, monkeypatch):
        """_make_grad_fn() delegates entirely to ParameterShiftGradient without a separate forward pass."""
        from qiskit.quantum_info import SparsePauliOp

        import qdr.training.gradients as gradients

        class FakeGradient:
            """Stub gradient that returns a unit vector without using the estimator."""

            def __init__(self, *args, **kwargs):
                pass

            def compute(self, weights, X, y):
                """Return an all-ones gradient of the same shape as weights."""
                return np.ones_like(weights)

        class GuardedClassifier(DataReuploadingClassifier):
            """Classifier that raises if _evaluate_batch is called unexpectedly."""

            def _evaluate_batch(self, weights, X, observables):
                """Raise to detect unexpected circuit evaluations during gradient computation."""
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
    """Tests for save() and load() round-trip persistence of the classifier."""

    def test_save_load(self, small_binary, tmp_path):
        """Saved and reloaded model produces identical weights and predictions."""
        X, y = small_binary
        model = DataReuploadingClassifier(n_qubits=1, n_layers=2, max_iter=5, seed=0)
        model.fit(X, y)
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = DataReuploadingClassifier.load(path)
        np.testing.assert_array_equal(model.weights_, loaded.weights_)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_not_fitted_raises(self, small_binary, tmp_path):
        """predict() raises NotFittedError before fit() is called."""
        from sklearn.exceptions import NotFittedError

        model = DataReuploadingClassifier(n_qubits=1, n_layers=2)
        X, y = small_binary
        with pytest.raises(NotFittedError):
            model.predict(X)
