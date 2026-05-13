"""Minimal example: binary classification on a moons dataset."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qdr.models import DataReuploadingClassifier
from qdr.visualization import plot_decision_boundary, plot_loss_curve


def main() -> None:
    """Train a binary QDR classifier and display basic diagnostics."""
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Fit preprocessing on the training split only to avoid test leakage.
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_plot = scaler.transform(X)

    model = DataReuploadingClassifier(
        n_qubits=2,
        n_layers=5,
        encoding="rx_ry_rz",
        entanglement="full",
        optimizer="COBYLA",
        max_iter=80,
        seed=42,
    )

    print("Fitting DataReuploadingClassifier...")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    try:
        import matplotlib.pyplot as plt

        plot_loss_curve(model.loss_history_, title="Training Loss (COBYLA)")
        plot_decision_boundary(model, X_plot, y, title="Decision Boundary")
        plt.show()
    except ImportError:
        print("matplotlib not available - skipping plots")


if __name__ == "__main__":
    main()
