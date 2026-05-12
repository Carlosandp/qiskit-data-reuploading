"""Parameter-shift-rule gradient computation for Qiskit 2.x primitives."""

from __future__ import annotations

from numbers import Real
from typing import Any, Callable

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qdr.circuits.data_reuploading import DataReuploadingCircuit


class ParameterShiftGradient:
    """Parameter-shift gradient of expectation-value circuits.

    For a circuit :math:`U(\\theta)` and observable :math:`O`, the derivative is:

    .. math::

        \\frac{\\partial}{\\partial \\theta_i}\\langle O \\rangle =
        \\frac{\\langle O \\rangle_{\\theta_i + s}
        - \\langle O \\rangle_{\\theta_i - s}}{2\\sin(s)}

    The default ``s = pi/2`` recovers the usual ``0.5 * (f+ - f-)`` rule.
    With an exact estimator this produces analytic gradients for the Pauli
    rotations used by :class:`qdr.circuits.DataReuploadingCircuit`. With
    finite-shot estimators it produces a stochastic gradient estimate.

    For an MSE gradient call this implementation evaluates the current
    predictions once and then performs two shifted evaluations per trainable
    parameter, i.e. ``2 * n_weights + 1`` batched estimator calls.

    Parameters
    ----------
    circuit_obj : DataReuploadingCircuit
        The parameterized circuit (already built).
    observable : SparsePauliOp
        Observable to differentiate.
    estimator : object or None, optional
        Qiskit V2 estimator. A fresh :class:`~qiskit.primitives.StatevectorEstimator`
        is created when ``None``.
    shift : float, optional
        Shift value ``s`` in radians. Default ``pi/2``.

    Raises
    ------
    ValueError
        If ``shift`` is non-finite or makes ``sin(shift)`` numerically zero.
    """

    def __init__(
        self,
        circuit_obj: DataReuploadingCircuit,
        observable: SparsePauliOp,
        estimator: Any | None = None,
        shift: float = np.pi / 2,
    ) -> None:
        if isinstance(shift, bool) or not isinstance(shift, Real) or not np.isfinite(shift):
            raise ValueError(f"shift must be a finite real number, got {shift!r}.")
        scale_denominator = 2.0 * np.sin(float(shift))
        if np.isclose(scale_denominator, 0.0):
            raise ValueError(
                f"shift={shift!r} is invalid for parameter-shift because sin(shift) is zero."
            )
        self.circuit_obj = circuit_obj
        self.observable = observable
        self.estimator = estimator if estimator is not None else StatevectorEstimator()
        self.shift = float(shift)
        self._shift_scale = 1.0 / scale_denominator

    def _validate_weights_X(
        self,
        weights: np.ndarray,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(weights, dtype=float)
        X = np.asarray(X, dtype=float)
        if weights.ndim != 1:
            raise ValueError(f"weights must be a 1D array, got weights.ndim={weights.ndim}.")
        expected_weights = (self.circuit_obj.n_weights,)
        if weights.shape != expected_weights:
            raise ValueError(f"weights must have shape {expected_weights}, got {weights.shape}.")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weights contains NaN or Inf values; all weights must be finite.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got X.ndim={X.ndim}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        expected_features = self.circuit_obj.n_features
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X must have {expected_features} features, got X.shape[1]={X.shape[1]}."
            )
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values; all features must be finite.")
        return weights, X

    @staticmethod
    def _validate_targets(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got y.ndim={y.ndim}.")
        if y.shape != (n_samples,):
            raise ValueError(f"y must have shape ({n_samples},), got {y.shape}.")
        if np.any(~np.isfinite(y)):
            raise ValueError("y contains NaN or Inf values; all targets must be finite.")
        return y

    def _eval(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        *,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate expectation values for a batch of samples."""
        if validate:
            weights, X = self._validate_weights_X(weights, X)
        param_values = self.circuit_obj.make_param_batch(weights, X)
        pub = (self.circuit_obj.circuit, self.observable, param_values)
        job = self.estimator.run([pub])
        evs = np.asarray(job.result()[0].data.evs, dtype=float)
        expected_shape = (X.shape[0],)
        if evs.shape != expected_shape:
            raise ValueError(
                f"estimator returned evs with shape {evs.shape}, expected {expected_shape}."
            )
        if np.any(~np.isfinite(evs)):
            raise ValueError("estimator returned NaN or Inf expectation values.")
        return evs

    def compute(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: Callable[..., float] | None = None,
    ) -> np.ndarray:
        """Compute the gradient of the MSE loss with respect to ``weights``.

        .. math::

            \\nabla_\\theta \\mathcal{L} =
            \\frac{2}{N} \\sum_i (\\hat{y}_i - y_i) \\nabla_\\theta \\hat{y}_i

        Parameters
        ----------
        weights : np.ndarray
            Current weight vector, shape ``(n_weights,)``.
        X : np.ndarray
            Input matrix, shape ``(n_samples, n_features)``.
        y : np.ndarray
            Target values in the same scale as the measured expectation values,
            shape ``(n_samples,)``.
        loss_fn : callable or None
            Ignored; kept for API compatibility with possible future gradient
            engines.

        Returns
        -------
        np.ndarray
            Gradient vector, shape ``(n_weights,)``.

        Raises
        ------
        TypeError
            If ``loss_fn`` is provided and is not callable.
        ValueError
            If ``weights``, ``X``, ``y``, or estimator outputs have invalid
            shape or non-finite values.
        """
        if loss_fn is not None and not callable(loss_fn):
            raise TypeError("loss_fn must be callable or None.")
        weights, X = self._validate_weights_X(weights, X)
        y = self._validate_targets(y, X.shape[0])
        n_weights = weights.shape[0]
        n_samples = X.shape[0]
        grad = np.zeros(n_weights)
        evs_0 = self._eval(weights, X, validate=False)

        for i in range(n_weights):
            w_plus = weights.copy()
            w_minus = weights.copy()
            w_plus[i] += self.shift
            w_minus[i] -= self.shift

            evs_plus = self._eval(w_plus, X, validate=False)
            evs_minus = self._eval(w_minus, X, validate=False)

            # The generalized prefactor keeps non-default shifts mathematically correct.
            dO_dtheta_i = self._shift_scale * (evs_plus - evs_minus)

            # Chain rule for MSE: dL/dtheta_i = (2/N) sum_j residual_j * d ev_j/dtheta_i.
            grad[i] = (2.0 / n_samples) * np.dot(evs_0 - y, dO_dtheta_i)

        return grad

    def compute_jacobian(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix ``d y_hat_i / d theta_j``.

        Parameters
        ----------
        weights : np.ndarray
            Shape ``(n_weights,)``.
        X : np.ndarray
            Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Jacobian of shape ``(n_samples, n_weights)``.

        Raises
        ------
        ValueError
            If ``weights``, ``X``, or estimator outputs have invalid shape or
            non-finite values.
        """
        weights, X = self._validate_weights_X(weights, X)
        n_weights = weights.shape[0]
        n_samples = X.shape[0]
        jacobian = np.zeros((n_samples, n_weights))

        for i in range(n_weights):
            w_plus = weights.copy()
            w_minus = weights.copy()
            w_plus[i] += self.shift
            w_minus[i] -= self.shift
            jacobian[:, i] = self._shift_scale * (
                self._eval(w_plus, X, validate=False)
                - self._eval(w_minus, X, validate=False)
            )

        return jacobian
