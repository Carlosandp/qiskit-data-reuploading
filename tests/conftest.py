"""Shared test fixtures."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons, make_regression


@pytest.fixture(scope="session")
def binary_dataset():
    """Return a two-moons binary classification dataset with 40 samples."""
    X, y = make_moons(n_samples=40, noise=0.15, random_state=0)
    return X, y


@pytest.fixture(scope="session")
def multiclass_dataset():
    """Return a three-class classification dataset with string labels."""
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=1,
        random_state=0,
    )
    return X, y.astype(str)  # string labels to test LabelEncoder


@pytest.fixture(scope="session")
def regression_dataset():
    """Return a regression dataset with 40 samples and 2 features."""
    X, y = make_regression(n_samples=40, n_features=2, noise=5.0, random_state=0)
    return X, y


@pytest.fixture(scope="session")
def tiny_weights_2q_3l():
    """Fixed weight vector for a 2-qubit, 3-layer rx_ry_rz circuit (18 weights)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-np.pi, np.pi, 2 * 3 * 3)  # n_qubits=2, n_layers=3, 3 rotations
