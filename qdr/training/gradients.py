"""Parameter-shift-rule gradient computation for Qiskit 2.x primitives."""

from __future__ import annotations

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class ParameterShiftGradient:
    """Exact gradient of expectation-value circuits via the parameter-shift rule.

    For a circuit :math:`U(\\theta)` and observable :math:`O`, the gradient is:

    .. math::

        \\frac{\\partial}{\\partial \\theta_i}\\langle O \\rangle =
        \\frac{1}{2}\\left[\\langle O \\rangle_{\\theta_i + \\pi/2}
        - \\langle O \\rangle_{\\theta_i - \\pi/2}\\right]

    This requires exactly ``2 × n_params × n_samples`` circuit evaluations per
    gradient call.

    Parameters
    ----------
    circuit_obj : DataReuploadingCircuit
        The parameterized circuit (already built).
    observable : SparsePauliOp
        Observable to differentiate.
    estimator : StatevectorEstimator or None, optional
        Qiskit V2 estimator.  A fresh :class:`~qiskit.primitives.StatevectorEstimator`
        is created when ``None``.
    shift : float, optional
        Shift value.  Default ``π/2``.
    """

    def __init__(
        self,
        circuit_obj,
        observable: SparsePauliOp,
        estimator=None,
        shift: float = np.pi / 2,
    ) -> None:
        self.circuit_obj = circuit_obj
        self.observable = observable
        self.estimator = estimator if estimator is not None else StatevectorEstimator()
        self.shift = shift

    def _eval(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate expectation values for a batch of samples."""
        param_values = self.circuit_obj.make_param_batch(weights, X)
        pub = (self.circuit_obj.circuit, self.observable, param_values)
        job = self.estimator.run([pub])
        return job.result()[0].data.evs  # shape (n_samples,)

    def compute(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn=None,
    ) -> np.ndarray:
        """Compute the gradient of the MSE loss w.r.t. *weights*.

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
            Target labels in ``{-1, +1}``, shape ``(n_samples,)``.
        loss_fn : callable or None
            Ignored (kept for API consistency with other gradient engines).

        Returns
        -------
        np.ndarray
            Gradient vector, shape ``(n_weights,)``.
        """
        n_weights = len(weights)
        n_samples = X.shape[0]
        grad = np.zeros(n_weights)
        evs_0 = self._eval(weights, X)  # current predictions

        for i in range(n_weights):
            w_plus = weights.copy()
            w_minus = weights.copy()
            w_plus[i] += self.shift
            w_minus[i] -= self.shift

            evs_plus = self._eval(w_plus, X)
            evs_minus = self._eval(w_minus, X)

            # d<O>/dθ_i for each sample
            dO_dtheta_i = 0.5 * (evs_plus - evs_minus)

            # Chain rule: dL/dθ_i = (2/N) Σ_j (evs_0_j - y_j) * dO_dtheta_i_j
            grad[i] = (2.0 / n_samples) * np.dot(evs_0 - y, dO_dtheta_i)

        return grad

    def compute_jacobian(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix :math:`\\partial \\hat{y}_i / \\partial \\theta_j`.

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
        """
        n_weights = len(weights)
        n_samples = X.shape[0]
        jacobian = np.zeros((n_samples, n_weights))

        for i in range(n_weights):
            w_plus = weights.copy()
            w_minus = weights.copy()
            w_plus[i] += self.shift
            w_minus[i] -= self.shift
            jacobian[:, i] = 0.5 * (self._eval(w_plus, X) - self._eval(w_minus, X))

        return jacobian
