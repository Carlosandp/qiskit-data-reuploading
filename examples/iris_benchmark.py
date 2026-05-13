"""Benchmark DataReuploadingClassifier against classical baselines on Iris."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from qdr.benchmarks import BenchmarkRunner
from qdr.models import DataReuploadingClassifier


def main() -> None:
    """Run a leakage-safe binary Iris benchmark."""
    iris = load_iris()
    mask = iris.target < 2  # drop virginica for a fast binary example
    X = iris.data[mask, :2]  # use only first 2 features for speed
    y = iris.target[mask]

    qdr = Pipeline(
        [
            ("scale_to_angles", MinMaxScaler(feature_range=(-np.pi, np.pi))),
            (
                "qdr",
                DataReuploadingClassifier(
                    n_qubits=2,
                    n_layers=4,
                    encoding="rx_ry_rz",
                    entanglement="linear",
                    optimizer="COBYLA",
                    max_iter=100,
                    seed=0,
                ),
            ),
        ]
    )

    runner = BenchmarkRunner(test_size=0.25, cv_folds=3, random_state=0, verbose=True)
    runner.run(
        X,
        y,
        qdr_model=qdr,
        include_logreg=True,
        include_svm=True,
        include_rf=True,
        include_mlp=True,
    )
    df = runner.summary()

    try:
        import matplotlib.pyplot as plt
        from qdr.visualization import plot_benchmark_comparison

        plot_benchmark_comparison(df, metric="accuracy", title="Iris Benchmark (binary)")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
