"""Optimizer wrappers: SPSA, COBYLA, and ADAM for quantum circuit training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize


@dataclass
class OptimizeResult:
    """Minimal result container compatible with ``scipy.optimize.OptimizeResult``."""

    x: np.ndarray
    fun: float
    nit: int
    loss_history: list[float] = field(default_factory=list)
    success: bool = True


# ---------------------------------------------------------------------------
# SPSA
# ---------------------------------------------------------------------------


class SPSA:
    """Simultaneous Perturbation Stochastic Approximation.

    Gradient-free optimizer that requires only **two** function evaluations per
    iteration regardless of the number of parameters — ideal for noisy quantum
    circuits.

    Parameters
    ----------
    maxiter : int
        Maximum number of iterations.
    a : float
        Numerator of the learning-rate schedule ``a_k = a/(A+k+1)^alpha``.
    c : float
        Perturbation-size numerator ``c_k = c/(k+1)^gamma``.
    A : float
        Stability constant (large ``A`` → slower early updates).
    alpha : float
        Learning-rate decay exponent (SPSA theory: 0.602).
    gamma : float
        Perturbation-size decay exponent (SPSA theory: 0.101).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        maxiter: int = 100,
        a: float = 0.6,
        c: float = 0.1,
        A: float = 10.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: int | None = None,
    ) -> None:
        self.maxiter = maxiter
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> OptimizeResult:
        """Run SPSA optimization.

        Parameters
        ----------
        fun : callable
            Objective function ``f(theta) -> float``.
        x0 : np.ndarray
            Initial parameter vector.
        callback : callable or None
            Called after each iteration with the current parameter vector.

        Returns
        -------
        OptimizeResult
        """
        theta = np.array(x0, dtype=float)
        n = len(theta)
        rng = np.random.default_rng(self.seed)
        loss_history: list[float] = []

        for k in range(self.maxiter):
            a_k = self.a / (self.A + k + 1) ** self.alpha
            c_k = self.c / (k + 1) ** self.gamma

            # Rademacher ±1 perturbation vector
            delta = 2 * rng.integers(0, 2, n).astype(float) - 1

            f_plus = fun(theta + c_k * delta)
            f_minus = fun(theta - c_k * delta)

            # Gradient estimate: g_k = (f+ - f-) / (2*c_k) / delta (element-wise)
            grad = (f_plus - f_minus) / (2.0 * c_k * delta)
            theta -= a_k * grad

            loss = float(fun(theta))
            loss_history.append(loss)
            if callback is not None:
                callback(theta)

        return OptimizeResult(
            x=theta,
            fun=float(fun(theta)),
            nit=self.maxiter,
            loss_history=loss_history,
        )


# ---------------------------------------------------------------------------
# COBYLA wrapper
# ---------------------------------------------------------------------------


class COBYLA:
    """Constrained Optimization BY Linear Approximations (gradient-free).

    Thin wrapper around :func:`scipy.optimize.minimize` with
    ``method="COBYLA"``.

    Parameters
    ----------
    maxiter : int
        Maximum number of function evaluations.
    rhobeg : float
        Initial size of the simplex.  Smaller values → finer search.
    """

    def __init__(self, maxiter: int = 200, rhobeg: float = 0.1) -> None:
        self.maxiter = maxiter
        self.rhobeg = rhobeg

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> OptimizeResult:
        """Run COBYLA optimization.

        Parameters
        ----------
        fun : callable
            Objective function.
        x0 : np.ndarray
            Initial parameter vector.
        callback : callable or None
            Called after each successful iteration.

        Returns
        -------
        OptimizeResult
        """
        loss_history: list[float] = []

        def _wrapped(theta: np.ndarray) -> float:
            val = float(fun(theta))
            loss_history.append(val)
            return val

        result = scipy_minimize(
            _wrapped,
            x0,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": self.rhobeg},
            callback=callback,
        )
        return OptimizeResult(
            x=result.x,
            fun=float(result.fun),
            nit=getattr(result, "nit", len(loss_history)),
            loss_history=loss_history,
            success=result.success,
        )


# ---------------------------------------------------------------------------
# ADAM
# ---------------------------------------------------------------------------


class ADAM:
    """Adaptive Moment Estimation (ADAM) optimizer.

    Requires a gradient function (e.g., from
    :class:`~qdr.training.gradients.ParameterShiftGradient`).

    Parameters
    ----------
    maxiter : int
        Maximum number of gradient steps.
    lr : float
        Learning rate (step size).
    beta1 : float
        Exponential decay rate for the first moment.
    beta2 : float
        Exponential decay rate for the second moment.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.maxiter = maxiter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> OptimizeResult:
        """Run ADAM optimization.

        Parameters
        ----------
        fun : callable
            Objective function (used only for loss tracking).
        x0 : np.ndarray
            Initial parameter vector.
        gradient_fn : callable
            Function returning the gradient ``∇L(theta)``.
        callback : callable or None
            Called after each iteration.

        Returns
        -------
        OptimizeResult
        """
        theta = np.array(x0, dtype=float)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        loss_history: list[float] = []

        for t in range(1, self.maxiter + 1):
            grad = gradient_fn(theta)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            loss = float(fun(theta))
            loss_history.append(loss)
            if callback is not None:
                callback(theta)

        return OptimizeResult(
            x=theta,
            fun=float(fun(theta)),
            nit=self.maxiter,
            loss_history=loss_history,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OPTIMIZERS: dict[str, type] = {
    "SPSA": SPSA,
    "COBYLA": COBYLA,
    "ADAM": ADAM,
}


def get_optimizer(name: str, **kwargs) -> SPSA | COBYLA | ADAM:
    """Instantiate an optimizer by name.

    Parameters
    ----------
    name : str
        One of ``"SPSA"``, ``"COBYLA"``, or ``"ADAM"`` (case-sensitive).
    **kwargs
        Forwarded to the optimizer constructor.

    Returns
    -------
    SPSA | COBYLA | ADAM
    """
    if name not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {list(_OPTIMIZERS)}, got '{name}'")
    return _OPTIMIZERS[name](**kwargs)
