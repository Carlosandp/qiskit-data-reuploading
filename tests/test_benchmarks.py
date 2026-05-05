"""Tests for BenchmarkRunner."""

import numpy as np
import pytest
from sklearn.datasets import make_moons

from qdr.benchmarks import BenchmarkRunner
from qdr.models import DataReuploadingClassifier


@pytest.fixture
def small_data():
    X, y = make_moons(n_samples=30, noise=0.1, random_state=0)
    return X, y


class TestBenchmarkRunner:
    def test_run_returns_self(self, small_data):
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=5)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        result = runner.run(X, y, qdr_model=qdr, include_svm=False, include_mlp=False)
        assert result is runner

    def test_summary_is_dataframe(self, small_data):
        import pandas as pd

        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=5)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(X, y, qdr_model=qdr, include_svm=True, include_mlp=False)
        df = runner.summary()
        assert isinstance(df, pd.DataFrame)
        assert "accuracy" in df.columns

    def test_all_models_included(self, small_data):
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(X, y, qdr_model=qdr, include_svm=True, include_mlp=True)
        names = [r.model_name for r in runner.results]
        assert "DataReuploadingClassifier" in names
        assert "SVM (RBF)" in names
        assert "MLP (32-16)" in names

    def test_accuracy_in_range(self, small_data):
        X, y = small_data
        qdr = DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=5)
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(X, y, qdr_model=qdr, include_svm=False, include_mlp=False)
        for r in runner.results:
            assert 0.0 <= r.accuracy <= 1.0

    def test_default_qdr_model_used_when_none(self, small_data):
        X, y = small_data
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(X, y, qdr_model=None, include_svm=False, include_mlp=False)
        assert len(runner.results) == 1
        assert runner.results[0].model_name == "DataReuploadingClassifier"

    def test_train_time_positive(self, small_data):
        X, y = small_data
        runner = BenchmarkRunner(cv_folds=0, verbose=False)
        runner.run(
            X,
            y,
            qdr_model=DataReuploadingClassifier(n_qubits=1, n_layers=1, max_iter=3),
            include_svm=False,
            include_mlp=False,
        )
        assert runner.results[0].train_time_s > 0
