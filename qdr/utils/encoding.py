"""Encoding helpers: gate vocabularies and feature-cycling utilities."""

from __future__ import annotations

import numpy as np

ENCODING_GATES: dict[str, list[str]] = {
    "rx": ["rx"],
    "ry": ["ry"],
    "rz": ["rz"],
    "rx_ry_rz": ["rx", "ry", "rz"],
}

N_ROTATIONS: dict[str, int] = {k: len(v) for k, v in ENCODING_GATES.items()}


def cycle_features(x: np.ndarray, n_slots: int) -> np.ndarray:
    """Repeat or truncate *x* to length *n_slots* using cyclic indexing.

    Parameters
    ----------
    x : np.ndarray
        Feature vector of shape (n_features,).
    n_slots : int
        Desired output length (number of encoding slots).

    Returns
    -------
    np.ndarray
        Array of shape (n_slots,).
    """
    n = len(x)
    indices = np.arange(n_slots) % n
    return x[indices]


def normalize_features(X: np.ndarray, feature_range: tuple[float, float] = (-np.pi, np.pi)) -> np.ndarray:
    """Min-max scale *X* into *feature_range*.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (n_samples, n_features).
    feature_range : tuple[float, float]
        Target value range.

    Returns
    -------
    np.ndarray
        Scaled matrix of shape (n_samples, n_features).
    """
    lo, hi = feature_range
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    return lo + (X - x_min) / denom * (hi - lo)
