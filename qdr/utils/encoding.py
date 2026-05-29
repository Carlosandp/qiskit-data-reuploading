"""Encoding helpers: gate vocabularies and feature-scaling utilities."""

from __future__ import annotations

from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

EncodingName = Literal["rx", "ry", "rz", "rx_ry_rz"]
ConstantStrategy = Literal["midpoint", "lower"]

ENCODING_GATES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "rx": ("rx",),
        "ry": ("ry",),
        "rz": ("rz",),
        "rx_ry_rz": ("rx", "ry", "rz"),
    }
)
"""Immutable mapping from encoding name to rotation-gate sequence."""

N_ROTATIONS: Mapping[str, int] = MappingProxyType(
    {name: len(gates) for name, gates in ENCODING_GATES.items()}
)
"""Immutable mapping from encoding name to number of rotations per qubit/layer."""


def _validate_positive_int(name: str, value: int) -> int:
    """Validate that a parameter is a positive integer.

    Args:
        name: Parameter name used in error messages.
        value: Candidate value to validate.

    Returns:
        The value cast to a plain Python int.

    Raises:
        ValueError: If value is a bool, not an integer, or less than 1.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def cycle_features(x: np.ndarray, n_slots: int) -> np.ndarray:
    """Repeat or truncate a feature vector to ``n_slots`` via cyclic indexing.

    Parameters
    ----------
    x : np.ndarray
        Feature vector of shape ``(n_features,)``.
    n_slots : int
        Desired output length, usually the number of encoding slots.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_slots,)`` with ``out[j] = x[j % n_features]``.

    Raises
    ------
    ValueError
        If ``x`` is not one-dimensional, empty, non-finite, or if ``n_slots`` is
        not a positive integer.

    Notes
    -----
    This helper mirrors the deterministic DRU feature assignment used by
    :class:`qdr.circuits.DataReuploadingCircuit`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array, got x.ndim={x.ndim}.")
    if x.size == 0:
        raise ValueError("x must contain at least one feature.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains NaN or Inf values; all features must be finite.")
    n_slots = _validate_positive_int("n_slots", n_slots)
    indices = np.arange(n_slots) % x.size
    return x[indices]


def normalize_features(
    X: np.ndarray,
    feature_range: tuple[float, float] = (-np.pi, np.pi),
    *,
    constant_strategy: ConstantStrategy = "midpoint",
) -> np.ndarray:
    """Min-max scale features into an angular range.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape ``(n_samples, n_features)``.
    feature_range : tuple[float, float], optional
        Target value range ``(lo, hi)``. Default ``(-pi, pi)``.
    constant_strategy : {"midpoint", "lower"}, optional
        How to map columns with zero variance. ``"midpoint"`` maps them to
        ``(lo + hi) / 2`` so a non-informative feature does not become an
        arbitrary extreme angle. ``"lower"`` maps them to ``lo`` for classic
        min-max compatibility.

    Returns
    -------
    np.ndarray
        Scaled matrix of shape ``(n_samples, n_features)``.

    Raises
    ------
    ValueError
        If ``X`` is not a non-empty 2D finite array, ``feature_range`` is
        malformed, or ``constant_strategy`` is unknown.

    Notes
    -----
    This function is stateless and intended for quick examples. For production
    pipelines and cross-validation, prefer a fitted scaler from scikit-learn so
    train/test leakage is controlled explicitly.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got X.ndim={X.ndim}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values; all features must be finite.")

    try:
        lo, hi = feature_range
    except (TypeError, ValueError):
        raise ValueError("feature_range must contain exactly two values: (lo, hi).")
    if not isinstance(constant_strategy, str) or constant_strategy not in {"midpoint", "lower"}:
        raise ValueError(
            "constant_strategy must be one of {'midpoint', 'lower'}, "
            f"got {constant_strategy!r}."
        )
    if (
        isinstance(lo, bool)
        or isinstance(hi, bool)
        or not isinstance(lo, Real)
        or not isinstance(hi, Real)
        or not np.isfinite(lo)
        or not np.isfinite(hi)
    ):
        raise ValueError(f"feature_range values must be finite real numbers, got {feature_range!r}.")
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        raise ValueError(f"feature_range must satisfy hi > lo, got {feature_range!r}.")

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    span = x_max - x_min
    constant_columns = span == 0.0
    safe_span = span.copy()
    safe_span[constant_columns] = 1.0

    scaled = lo + (X - x_min) / safe_span * (hi - lo)
    if np.any(constant_columns):
        fill = (lo + hi) / 2.0 if constant_strategy == "midpoint" else lo
        scaled[:, constant_columns] = fill
    return scaled
