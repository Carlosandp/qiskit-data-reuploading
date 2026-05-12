"""Tests for plotting utilities."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qdr.visualization import (
    plot_benchmark_comparison,
    plot_bloch_sphere,
    plot_decision_boundary,
    plot_loss_curve,
)


class RecordingModel:
    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.last_X: np.ndarray | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.last_X = np.asarray(X, dtype=float)
        return (self.last_X[:, 2] > self.last_X[:, 3]).astype(int)


class TestDecisionBoundary:
    def test_uses_selected_feature_columns_and_mean_slice(self):
        X = np.array(
            [
                [1.0, 10.0, -1.0, -2.0],
                [2.0, 20.0, 0.0, 1.0],
                [3.0, 30.0, 1.0, 0.0],
                [4.0, 40.0, 2.0, 2.0],
            ]
        )
        y = np.array([0, 0, 1, 1])
        model = RecordingModel()

        fig = plot_decision_boundary(model, X, y, resolution=5, feature_indices=(2, 3))

        assert model.last_X is not None
        assert model.last_X.shape == (25, 4)
        np.testing.assert_allclose(model.last_X[:, 0], X[:, 0].mean())
        np.testing.assert_allclose(model.last_X[:, 1], X[:, 1].mean())
        assert np.unique(model.last_X[:, 2]).size == 5
        assert np.unique(model.last_X[:, 3]).size == 5
        assert len(fig.axes) == 2  # plot axes + colorbar axes
        plt.close(fig)

    def test_rejects_invalid_decision_boundary_inputs(self):
        X = np.ones((3, 2))
        y = np.array([0, 1, 0])
        model = RecordingModel()

        with pytest.raises(ValueError, match="resolution must be >= 2"):
            plot_decision_boundary(model, X, y, resolution=1)
        with pytest.raises(ValueError, match="resolution must be an integer"):
            plot_decision_boundary(model, X, y, resolution=True)
        with pytest.raises(ValueError, match="feature_indices must contain exactly two"):
            plot_decision_boundary(model, X, y, feature_indices=0)
        with pytest.raises(ValueError, match="feature_indices must be distinct"):
            plot_decision_boundary(model, X, y, feature_indices=(0, 0))
        with pytest.raises(ValueError, match="feature_indices must be integers"):
            plot_decision_boundary(model, X, y, feature_indices=(0, True))
        with pytest.raises(ValueError, match="feature_indices=.*n_features=2"):
            plot_decision_boundary(model, X, y, feature_indices=(0, 2))
        with pytest.raises(ValueError, match="y must be a 1D array"):
            plot_decision_boundary(model, X, np.array([[0], [1], [0]]))
        with pytest.raises(ValueError, match="y must have length 3"):
            plot_decision_boundary(model, X, np.array([0, 1]))
        with pytest.raises(ValueError, match="X must contain at least one sample"):
            plot_decision_boundary(model, np.empty((0, 2)), np.array([]))
        with pytest.raises(ValueError, match="X must contain at least one feature"):
            plot_decision_boundary(model, np.empty((2, 0)), np.array([0, 1]))
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            plot_decision_boundary(model, np.array([[0.0, np.nan], [1.0, 2.0]]), np.array([0, 1]))


class TestLossCurve:
    def test_plots_loss_curve(self):
        fig = plot_loss_curve([1.0, 0.5, 0.25])
        ax = fig.axes[0]

        assert ax.get_xlabel() == "Iteration"
        assert ax.get_ylabel() == "Loss"
        np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([1.0, 0.5, 0.25]))
        plt.close(fig)

    def test_rejects_invalid_loss_history(self):
        with pytest.raises(ValueError, match="loss_history must contain at least one value"):
            plot_loss_curve([])
        with pytest.raises(ValueError, match="loss_history contains NaN or Inf"):
            plot_loss_curve([1.0, np.inf])
        with pytest.raises(ValueError, match="loss_history must be a 1D sequence"):
            plot_loss_curve([[1.0, 2.0]])


class TestBenchmarkComparison:
    def test_supports_probability_and_time_metrics(self):
        df = pd.DataFrame(
            {
                "accuracy": [0.8, 0.9],
                "train_time_s": [1.5, 3.0],
            },
            index=["QDR", "RF"],
        )

        fig_acc = plot_benchmark_comparison(df, metric="accuracy")
        assert fig_acc.axes[0].get_ylim()[1] == pytest.approx(1.05)
        plt.close(fig_acc)

        fig_time = plot_benchmark_comparison(df, metric="train_time_s")
        assert fig_time.axes[0].get_ylim()[0] == pytest.approx(0.0)
        assert fig_time.axes[0].get_ylim()[1] > 3.0
        plt.close(fig_time)

    def test_rejects_invalid_benchmark_inputs(self):
        df = pd.DataFrame({"accuracy": [0.8]}, index=["QDR"])

        with pytest.raises(ValueError, match="metric 'f1' is not present"):
            plot_benchmark_comparison(df, metric="f1")
        with pytest.raises(ValueError, match="summary_df must contain at least one"):
            plot_benchmark_comparison(pd.DataFrame({"accuracy": []}), metric="accuracy")
        with pytest.raises(ValueError, match="metric 'accuracy' contains NaN or Inf"):
            plot_benchmark_comparison(
                pd.DataFrame({"accuracy": [np.nan]}, index=["QDR"]),
                metric="accuracy",
            )
        with pytest.raises(ValueError, match="metric 'accuracy' must contain numeric values"):
            plot_benchmark_comparison(
                pd.DataFrame({"accuracy": ["high"]}, index=["QDR"]),
                metric="accuracy",
            )


class TestBlochSphere:
    def test_plots_normalized_single_qubit_state(self):
        fig = plot_bloch_sphere(np.array([1.0 + 0.0j, 0.0 + 0.0j]))

        assert hasattr(fig.axes[0], "get_zlim")
        plt.close(fig)

    def test_rejects_invalid_bloch_inputs(self):
        with pytest.raises(ValueError, match="statevector must have shape"):
            plot_bloch_sphere(np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="statevector must be normalized"):
            plot_bloch_sphere(np.array([2.0 + 0.0j, 0.0 + 0.0j]))
        with pytest.raises(ValueError, match="statevector contains NaN or Inf"):
            plot_bloch_sphere(np.array([np.nan + 0.0j, 0.0 + 0.0j]))

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="ax must be a 3D axes"):
            plot_bloch_sphere(np.array([1.0 + 0.0j, 0.0 + 0.0j]), ax=ax)
        plt.close(fig)
