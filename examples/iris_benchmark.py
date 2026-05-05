"""Benchmark DataReuploadingClassifier vs MLP and SVM on the Iris dataset."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qdr.benchmarks import BenchmarkRunner
from qdr.models import DataReuploadingClassifier

# --- Data (binary: setosa vs versicolor) ---
iris = load_iris()
mask = iris.target < 2  # drop virginica for binary classification
X = iris.data[mask, :2]  # use only first 2 features for speed
y = iris.target[mask]

scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = scaler.fit_transform(X)

# --- Quantum model ---
qdr = DataReuploadingClassifier(
    n_qubits=2,
    n_layers=4,
    encoding="rx_ry_rz",
    entanglement="linear",
    optimizer="COBYLA",
    max_iter=100,
    seed=0,
)

# --- Run benchmark ---
runner = BenchmarkRunner(test_size=0.25, cv_folds=3, random_state=0, verbose=True)
runner.run(X_scaled, y, qdr_model=qdr, include_svm=True, include_mlp=True)
df = runner.summary()

try:
    import matplotlib.pyplot as plt
    from qdr.visualization import plot_benchmark_comparison

    fig = plot_benchmark_comparison(df, metric="accuracy", title="Iris Benchmark (binary)")
    plt.show()
except ImportError:
    pass
