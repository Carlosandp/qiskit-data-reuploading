"""Tests for BenchmarkRunner."""

import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification, make_moons

from qdr.benchmarks import BenchmarkResult, BenchmarkRunner
from qdr.models import DataReuploadingClassifier


@pytest.fixture
def small_data():
    """Return a two-moons dataset with 30 samples for benchmarking tests."""
    X, y = make_moons(n_samples=30, noise=0.1, random_state=0)
    return X, y


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner run/summary workflow and validation."""

    def test_run_returns_self(self, small_data):
        """run() returns the BenchmarkRunner instance for method chaining."""
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=5)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        result = runner.run(
            X,
            y,
            qdr_model=qdr,
            include_logreg=False,
            include_svm=False,
            include_mlp=False,
            include_rf=False,
        )
        assert result is runner

    def test_summary_is_dataframe(self, small_data):
        """summary() returns a pandas DataFrame with an 'accuracy' column."""
        import pandas as pd

        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=5)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(
            X,
            y,
            qdr_model=qdr,
            include_svm=True,
            include_mlp=False,
            include_rf=False,
        )
        df = runner.summary()
        assert isinstance(df, pd.DataFrame)
        assert "accuracy" in df.columns

    def test_summary_before_run_raises(self):
        """summary() raises ValueError when called before run()."""
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        with pytest.raises(ValueError, match="Call run\\(\\) before summary"):
            runner.summary()

    def test_summary_preserves_metric_precision(self):
        """summary() returns metrics at full floating-point precision."""
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner._results = [
            BenchmarkResult(
                model_name="model",
                accuracy=0.123456789,
                f1=0.987654321,
                train_time_s=1.23456789,
                predict_time_s=0.000123456,
            )
        ]
        df = runner.summary()
        assert df.loc["model", "accuracy"] == pytest.approx(0.123456789)
        assert df.loc["model", "predict_time_s"] == pytest.approx(0.000123456)

    def test_all_models_included(self, small_data):
        """All included baseline models appear in the results by name."""
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(X, y, qdr_model=qdr, include_svm=True, include_mlp=True)
        names = [r.model_name for r in runner.results]
        assert "DataReuploadingClassifier" in names
        assert "Logistic Regression" in names
        assert "SVM (RBF)" in names
        assert "Random Forest" in names
        assert "MLP (32-16)" in names

    def test_accuracy_in_range(self, small_data):
        """All reported accuracy values are in [0.0, 1.0]."""
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=5)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(
            X,
            y,
            qdr_model=qdr,
            include_logreg=False,
            include_svm=False,
            include_mlp=False,
            include_rf=False,
        )
        for r in runner.results:
            assert 0.0 <= r.accuracy <= 1.0

    def test_default_qdr_model_used_when_none(self, small_data):
        """qdr_model=None causes run() to create a default QDR model."""
        X, y = small_data
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(
            X,
            y,
            qdr_model=None,
            include_logreg=False,
            include_svm=False,
            include_mlp=False,
            include_rf=False,
        )
        assert len(runner.results) == 1
        assert runner.results[0].model_name == "DataReuploadingClassifier"

    def test_default_qdr_model_adapts_to_classes_and_features(self, monkeypatch):
        """Auto-created QDR model uses n_qubits=n_classes and n_layers scaled to n_features."""
        import qdr.models.classifier as classifier_module

        captured: dict[str, int] = {}

        class FakeQDR(BaseEstimator, ClassifierMixin):
            def __init__(
                self,
                n_qubits=2,
                n_layers=3,
                optimizer="COBYLA",
                max_iter=50,
                seed=None,
            ):
                captured["n_qubits"] = n_qubits
                captured["n_layers"] = n_layers
                self.n_qubits = n_qubits
                self.n_layers = n_layers
                self.optimizer = optimizer
                self.max_iter = max_iter
                self.seed = seed

            def fit(self, X, y):
                self.majority_ = np.bincount(y).argmax()
                self.loss_history_ = [0.0]
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.majority_, dtype=int)

        monkeypatch.setattr(classifier_module, "DataReuploadingClassifier", FakeQDR)
        X, y = make_classification(
            n_samples=18,
            n_features=40,
            n_classes=3,
            n_informative=6,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=0,
        )
        runner = BenchmarkRunner(test_size=0.5, cv_folds=0, verbose=False)
        runner.run(
            X,
            y,
            include_logreg=False,
            include_svm=False,
            include_mlp=False,
            include_rf=False,
        )

        assert captured["n_qubits"] == 3
        assert captured["n_layers"] == 5
        assert runner.results[0].extra["classes"] == [0, 1, 2]

    def test_train_time_positive(self, small_data):
        """train_time_s is strictly positive after fitting the QDR model."""
        X, y = small_data
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(
            X,
            y,
            qdr_model=DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3),
            include_logreg=False,
            include_svm=False,
            include_mlp=False,
        )
        assert runner.results[0].train_time_s > 0

    def test_string_labels_supported(self, small_data):
        """run() accepts string class labels and encodes them to integers internally."""
        X, y = small_data
        labels = np.where(y == 0, "left", "right")
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(
            X,
            labels,
            qdr_model=qdr,
            include_logreg=False,
            include_svm=False,
            include_mlp=False,
            include_rf=False,
        )
        assert runner.classes_.tolist() == ["left", "right"]

    def test_input_validation(self, small_data):
        """run() raises ValueError for NaN/Inf values, empty features, length mismatch, and single-class y."""
        X, y = small_data
        runner = BenchmarkRunner(cv_folds=0, verbose=False)

        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            runner.run(X_bad, y)

        with pytest.raises(ValueError, match="X must contain at least one feature"):
            runner.run(np.zeros((len(y), 0)), y)

        with pytest.raises(ValueError, match="X and y have inconsistent lengths"):
            runner.run(X, y[:-1])

        with pytest.raises(ValueError, match="requires at least 2 classes"):
            runner.run(X, np.zeros_like(y))

    def test_constructor_validation(self):
        """BenchmarkRunner.__init__() raises ValueError for invalid test_size, cv_folds, random_state, and verbose."""
        with pytest.raises(ValueError, match="test_size must satisfy"):
            BenchmarkRunner(test_size=1.0)
        with pytest.raises(ValueError, match="cv_folds must be >= 0"):
            BenchmarkRunner(cv_folds=-1)
        with pytest.raises(ValueError, match="random_state must be an integer or None"):
            BenchmarkRunner(random_state=True)
        with pytest.raises(ValueError, match="verbose must be a bool"):
            BenchmarkRunner(verbose="no")

    def test_include_flag_validation(self, small_data):
        """run() raises ValueError when an include_* flag receives a non-bool value."""
        X, y = small_data
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        with pytest.raises(ValueError, match="include_svm must be a bool"):
            runner.run(X, y, include_svm="yes")

    def test_cv_folds_must_fit_class_counts(self):
        """run() raises ValueError when cv_folds exceeds the number of samples per class."""
        X = np.zeros((6, 2))
        y = np.array([0, 0, 0, 1, 1, 1])
        runner = BenchmarkRunner(test_size=0.5, cv_folds=4, verbose=False)
        with pytest.raises(ValueError, match="cv_folds=4 exceeds"):
            runner.run(X, y)

    def test_test_size_must_leave_each_split_with_all_classes(self):
        """run() raises ValueError when test_size is too small to keep all classes in both splits."""
        X = np.zeros((6, 2))
        y = np.array([0, 0, 1, 1, 2, 2])
        runner = BenchmarkRunner(test_size=0.2, cv_folds=0, verbose=False)
        with pytest.raises(ValueError, match="test_size leaves too few samples"):
            runner.run(X, y)

    def test_xgboost_missing_fails_before_training(self, small_data, monkeypatch):
        """run() raises ImportError immediately when xgboost is requested but not installed."""
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        monkeypatch.setitem(sys.modules, "xgboost", None)

        with pytest.raises(ImportError, match="include_xgboost=True requires"):
            runner.run(
                X,
                y,
                qdr_model=qdr,
                include_logreg=False,
                include_svm=False,
                include_mlp=False,
                include_rf=False,
                include_xgboost=True,
            )

        assert runner.results == []
