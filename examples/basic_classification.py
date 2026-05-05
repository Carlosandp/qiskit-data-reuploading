"""Minimal example: binary classification on a moons dataset."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qdr.models import DataReuploadingClassifier
from qdr.visualization import plot_decision_boundary, plot_loss_curve

# --- Data ---
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model ---
model = DataReuploadingClassifier(
    n_qubits=2,
    n_layers=5,
    encoding="rx_ry_rz",
    entanglement="full",
    optimizer="COBYLA",
    max_iter=80,
    seed=42,
)

print("Fitting DataReuploadingClassifier …")
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test  accuracy: {test_acc:.3f}")

# --- Visualize ---
try:
    import matplotlib.pyplot as plt

    fig1 = plot_loss_curve(model.loss_history_, title="Training Loss (COBYLA)")
    fig2 = plot_decision_boundary(model, X_scaled, y, title="Decision Boundary")
    plt.show()
except ImportError:
    print("matplotlib not available — skipping plots")
