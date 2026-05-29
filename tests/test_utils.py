"""Tests for encoding utilities."""

import numpy as np
import pytest

from qdr.utils import ENCODING_GATES, N_ROTATIONS, cycle_features, normalize_features


class TestEncodingConstants:
    """Tests for the ENCODING_GATES and N_ROTATIONS module-level constants."""

    def test_encoding_gates_are_immutable(self):
        """ENCODING_GATES mapping raises TypeError on mutation attempts."""
        with pytest.raises(TypeError):
            ENCODING_GATES["rx"] = ("ry",)
        with pytest.raises(TypeError):
            ENCODING_GATES["rx"] += ("rz",)

    def test_n_rotations_matches_encoding_gates(self):
        """N_ROTATIONS values match the length of each ENCODING_GATES tuple."""
        assert dict(N_ROTATIONS) == {name: len(gates) for name, gates in ENCODING_GATES.items()}
        assert N_ROTATIONS["rx_ry_rz"] == 3


class TestCycleFeatures:
    """Tests for the cycle_features utility function."""

    def test_cycles_to_requested_slots(self):
        """Features are repeated cyclically to fill the requested slot count."""
        out = cycle_features(np.array([10.0, 20.0, 30.0]), 8)
        np.testing.assert_allclose(out, np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0]))

    def test_truncates_when_slots_are_fewer_than_features(self):
        """When n_slots < n_features only the first n_slots features are returned."""
        out = cycle_features(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        np.testing.assert_allclose(out, np.array([1.0, 2.0]))

    def test_rejects_invalid_inputs(self):
        """cycle_features raises ValueError for invalid x or n_slots arguments."""
        with pytest.raises(ValueError, match="x must be a 1D array"):
            cycle_features(np.zeros((2, 2)), 3)
        with pytest.raises(ValueError, match="x must contain at least one feature"):
            cycle_features(np.array([]), 3)
        with pytest.raises(ValueError, match="x contains NaN or Inf"):
            cycle_features(np.array([1.0, np.inf]), 3)
        with pytest.raises(ValueError, match="n_slots must be >= 1"):
            cycle_features(np.array([1.0]), 0)
        with pytest.raises(ValueError, match="n_slots must be a positive integer"):
            cycle_features(np.array([1.0]), True)


class TestNormalizeFeatures:
    """Tests for the normalize_features utility function."""

    def test_scales_each_feature_to_target_range(self):
        """Each feature column is linearly scaled to [lo, hi]."""
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        scaled = normalize_features(X, feature_range=(-1.0, 1.0))
        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(scaled, expected)

    def test_constant_columns_default_to_midpoint(self):
        """Zero-variance columns are mapped to the midpoint of the range."""
        X = np.array([[2.0, 1.0], [2.0, 3.0]])
        scaled = normalize_features(X, feature_range=(-np.pi, np.pi))
        np.testing.assert_allclose(scaled[:, 0], np.zeros(2))
        np.testing.assert_allclose(scaled[:, 1], np.array([-np.pi, np.pi]))

    def test_constant_columns_can_use_lower_bound_for_compatibility(self):
        """constant_strategy='lower' maps zero-variance columns to lo."""
        X = np.array([[2.0], [2.0]])
        scaled = normalize_features(X, feature_range=(-1.0, 1.0), constant_strategy="lower")
        np.testing.assert_allclose(scaled, np.full((2, 1), -1.0))

    def test_rejects_invalid_X(self):
        """normalize_features raises ValueError for malformed X arrays."""
        with pytest.raises(ValueError, match="X must be a 2D array"):
            normalize_features(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="X must contain at least one sample"):
            normalize_features(np.empty((0, 2)))
        with pytest.raises(ValueError, match="X must contain at least one feature"):
            normalize_features(np.empty((2, 0)))
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            normalize_features(np.array([[1.0, np.nan]]))

    def test_rejects_invalid_feature_range(self):
        """normalize_features raises ValueError for malformed feature_range."""
        with pytest.raises(ValueError, match="feature_range must contain exactly two values"):
            normalize_features(np.ones((2, 1)), feature_range=1.0)
        with pytest.raises(ValueError, match="feature_range must contain exactly two values"):
            normalize_features(np.ones((2, 1)), feature_range=(0.0, 1.0, 2.0))
        with pytest.raises(ValueError, match="feature_range values must be finite"):
            normalize_features(np.ones((2, 1)), feature_range=(0.0, np.inf))
        with pytest.raises(ValueError, match="feature_range must satisfy hi > lo"):
            normalize_features(np.ones((2, 1)), feature_range=(1.0, 1.0))

    def test_rejects_invalid_constant_strategy(self):
        """normalize_features raises ValueError for unknown constant_strategy."""
        with pytest.raises(ValueError, match="constant_strategy must be one of"):
            normalize_features(np.ones((2, 1)), constant_strategy="bad")
        with pytest.raises(ValueError, match="constant_strategy must be one of"):
            normalize_features(np.ones((2, 1)), constant_strategy=[])
