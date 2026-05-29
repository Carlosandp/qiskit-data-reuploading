"""Optimizer wrappers: SPSA, COBYLA, and ADAM for quantum circuit training."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize


ObjectiveFn = Callable[[np.ndarray], float]
CallbackFn = Callable[[np.ndarray], None]
GradientFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class OptimizeResult:
    """Minimal result container compatible with ``scipy.optimize.OptimizeResult``.

    Attributes
    ----------
    x : np.ndarray
        Final parameter vector.
    fun : float
        Final objective value.
    nit : int
        Number of optimizer iterations.
    loss_history : list[float]
        Tracked objective values. For COBYLA these are function-evaluation
        values because SciPy controls the internal iteration schedule.
    success : bool
        Whether the optimizer reported successful termination.
    nfev : int
        Number of objective-function evaluations performed by this wrapper.
    njev : int
        Number of gradient evaluations performed by this wrapper.
    message : str
        Optional optimizer status message.
    """

    x: np.ndarray
    fun: float
    nit: int
    loss_history: list[float] = field(default_factory=list)
    success: bool = True
    nfev: int = 0
    njev: int = 0
    message: str = ""


def _validate_positive_int(name: str, value: int) -> int:
    """Validate that a hyperparameter is a positive integer.

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


def _validate_positive_real(name: str, value: float) -> float:
    """Validate that a hyperparameter is a finite, strictly positive real number.

    Args:
        name: Parameter name used in error messages.
        value: Candidate value to validate.

    Returns:
        The value cast to a Python float.

    Raises:
        ValueError: If value is non-finite, not a real number, or not > 0.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _validate_nonnegative_real(name: str, value: float) -> float:
    """Validate that a hyperparameter is a finite, non-negative real number.

    Args:
        name: Parameter name used in error messages.
        value: Candidate value to validate.

    Returns:
        The value cast to a Python float.

    Raises:
        ValueError: If value is non-finite, not a real number, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def _validate_beta(name: str, value: float) -> float:
    """Validate that an ADAM moment-decay coefficient satisfies ``0 <= value < 1``.

    Args:
        name: Parameter name used in error messages.
        value: Candidate decay coefficient to validate.

    Returns:
        The value cast to a Python float.

    Raises:
        ValueError: If value is non-finite, not a real number, or outside [0, 1).
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and satisfy 0 <= {name} < 1, got {value!r}.")
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1, got {value}.")
    return value


def _validate_initial_point(x0: np.ndarray) -> np.ndarray:
    """Validate and copy the initial parameter vector.

    Args:
        x0: Starting point for the optimizer.

    Returns:
        A float64 copy of x0 as a 1D array.

    Raises:
        ValueError: If ``x0`` is not 1D, empty, or contains non-finite values.
    """
    theta = np.asarray(x0, dtype=float)
    if theta.ndim != 1:
        raise ValueError(f"x0 must be a 1D array, got x0.ndim={theta.ndim}.")
    if theta.size == 0:
        raise ValueError("x0 must contain at least one parameter.")
    if np.any(~np.isfinite(theta)):
        raise ValueError("x0 contains NaN or Inf values; all parameters must be finite.")
    return theta.copy()


def _evaluate_objective(fun: ObjectiveFn, theta: np.ndarray) -> float:
    """Safely call the objective function and validate the return value.

    Args:
        fun: Objective function ``f(theta) -> float``.
        theta: Current parameter vector passed to ``fun``.

    Returns:
        The scalar objective value.

    Raises:
        ValueError: If ``fun`` does not return a scalar or returns a non-finite
            value.
    """
    try:
        value = float(fun(theta.copy()))
    except (TypeError, ValueError) as exc:
        raise ValueError("Objective function must return a scalar float.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Objective function returned a non-finite value: {value!r}.")
    return value


def _evaluate_gradient(gradient_fn: GradientFn, theta: np.ndarray) -> np.ndarray:
    """Safely call the gradient function and validate the return array.

    Args:
        gradient_fn: Gradient function ``grad(theta) -> np.ndarray``.
        theta: Current parameter vector passed to ``gradient_fn``.

    Returns:
        Gradient array with the same shape as ``theta``.

    Raises:
        ValueError: If the gradient has a shape mismatch or contains non-finite
            values.
    """
    grad = np.asarray(gradient_fn(theta.copy()), dtype=float)
    if grad.shape != theta.shape:
        raise ValueError(f"gradient_fn returned shape {grad.shape}, expected {theta.shape}.")
    if np.any(~np.isfinite(grad)):
        raise ValueError("gradient_fn returned NaN or Inf values.")
    return grad


def _call_callback(callback: CallbackFn | None, theta: np.ndarray) -> None:
    """Invoke the optional callback with a copy of the current parameters.

    Args:
        callback: Callable to invoke, or ``None`` to skip.
        theta: Current parameter vector; a copy is passed to the callback.
    """
    if callback is not None:
        callback(theta.copy())


# ---------------------------------------------------------------------------
# SPSA
# ---------------------------------------------------------------------------


class SPSA:
    """Simultaneous Perturbation Stochastic Approximation.

    Gradient-free optimizer that requires two perturbation evaluations per
    iteration regardless of the number of parameters. This wrapper also
    evaluates the objective once at the updated point to record loss history,
    so the tracked training cost is three objective evaluations per iteration.

    Parameters
    ----------
    maxiter : int
        Maximum number of iterations.
    a : float
        Numerator of the learning-rate schedule ``a_k = a/(A+k+1)^alpha``.
    c : float
        Perturbation-size numerator ``c_k = c/(k+1)^gamma``.
    A : float
        Non-negative stability constant.
    alpha : float
        Learning-rate decay exponent. The classical SPSA recommendation is
        approximately ``0.602``.
    gamma : float
        Perturbation-size decay exponent. The classical SPSA recommendation is
        approximately ``0.101``.
    seed : int or None
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If any hyperparameter is outside its valid numerical range.
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
        self.maxiter = _validate_positive_int("maxiter", maxiter)
        self.a = _validate_positive_real("a", a)
        self.c = _validate_positive_real("c", c)
        self.A = _validate_nonnegative_real("A", A)
        self.alpha = _validate_positive_real("alpha", alpha)
        self.gamma = _validate_positive_real("gamma", gamma)
        self.seed = seed

    def minimize(
        self,
        fun: ObjectiveFn,
        x0: np.ndarray,
        callback: CallbackFn | None = None,
    ) -> OptimizeResult:
        """Run SPSA optimization.

        Parameters
        ----------
        fun : callable
            Objective function ``f(theta) -> float``.
        x0 : np.ndarray
            Initial parameter vector.
        callback : callable or None
            Called after each iteration with a copy of the current parameter vector.

        Returns
        -------
        OptimizeResult

        Raises
        ------
        ValueError
            If ``x0`` is invalid or the objective returns a non-finite value.
        """
        theta = _validate_initial_point(x0)
        n = theta.size
        rng = np.random.default_rng(self.seed)
        loss_history: list[float] = []
        nfev = 0

        for k in range(self.maxiter):
            a_k = self.a / (self.A + k + 1) ** self.alpha
            c_k = self.c / (k + 1) ** self.gamma

            delta = 2 * rng.integers(0, 2, n).astype(float) - 1

            f_plus = _evaluate_objective(fun, theta + c_k * delta)
            f_minus = _evaluate_objective(fun, theta - c_k * delta)
            nfev += 2

            grad = (f_plus - f_minus) / (2.0 * c_k * delta)
            theta -= a_k * grad

            loss = _evaluate_objective(fun, theta)
            nfev += 1
            loss_history.append(loss)
            _call_callback(callback, theta)

        return OptimizeResult(
            x=theta,
            fun=loss_history[-1],
            nit=self.maxiter,
            loss_history=loss_history,
            nfev=nfev,
            success=True,
        )


# ---------------------------------------------------------------------------
# COBYLA wrapper
# ---------------------------------------------------------------------------


class COBYLA:
    """Constrained Optimization BY Linear Approximations.

    Thin wrapper around :func:`scipy.optimize.minimize` with
    ``method="COBYLA"``.

    Parameters
    ----------
    maxiter : int
        Maximum number of function evaluations passed to SciPy.
    rhobeg : float
        Initial trust-region radius.

    Raises
    ------
    ValueError
        If ``maxiter`` or ``rhobeg`` is outside its valid numerical range.
    """

    def __init__(self, maxiter: int = 200, rhobeg: float = 0.1) -> None:
        self.maxiter = _validate_positive_int("maxiter", maxiter)
        self.rhobeg = _validate_positive_real("rhobeg", rhobeg)

    def minimize(
        self,
        fun: ObjectiveFn,
        x0: np.ndarray,
        callback: CallbackFn | None = None,
    ) -> OptimizeResult:
        """Run COBYLA optimization.

        Parameters
        ----------
        fun : callable
            Objective function.
        x0 : np.ndarray
            Initial parameter vector.
        callback : callable or None
            Called by SciPy with a copy of the current parameter vector.

        Returns
        -------
        OptimizeResult

        Raises
        ------
        ValueError
            If ``x0`` is invalid or the objective returns a non-finite value.
        """
        theta0 = _validate_initial_point(x0)
        loss_history: list[float] = []
        nfev = 0

        def _wrapped(theta: np.ndarray) -> float:
            nonlocal nfev
            val = _evaluate_objective(fun, theta)
            nfev += 1
            loss_history.append(val)
            return val

        def _wrapped_callback(theta: np.ndarray) -> None:
            _call_callback(callback, np.asarray(theta, dtype=float))

        result = scipy_minimize(
            _wrapped,
            theta0,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": self.rhobeg},
            callback=_wrapped_callback if callback is not None else None,
        )
        return OptimizeResult(
            x=np.asarray(result.x, dtype=float),
            fun=float(result.fun),
            nit=int(getattr(result, "nit", len(loss_history))),
            loss_history=loss_history,
            success=bool(result.success),
            nfev=nfev,
            message=str(getattr(result, "message", "")),
        )


# ---------------------------------------------------------------------------
# ADAM
# ---------------------------------------------------------------------------


class ADAM:
    """Adaptive Moment Estimation optimizer.

    Requires a gradient function, for example one built from
    :class:`~qdr.training.gradients.ParameterShiftGradient`.

    Parameters
    ----------
    maxiter : int
        Maximum number of gradient steps.
    lr : float
        Learning rate.
    beta1 : float
        Exponential decay rate for the first moment, ``0 <= beta1 < 1``.
    beta2 : float
        Exponential decay rate for the second moment, ``0 <= beta2 < 1``.
    eps : float
        Positive numerical stability constant.

    Raises
    ------
    ValueError
        If any hyperparameter is outside its valid numerical range.
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.maxiter = _validate_positive_int("maxiter", maxiter)
        self.lr = _validate_positive_real("lr", lr)
        self.beta1 = _validate_beta("beta1", beta1)
        self.beta2 = _validate_beta("beta2", beta2)
        self.eps = _validate_positive_real("eps", eps)

    def minimize(
        self,
        fun: ObjectiveFn,
        x0: np.ndarray,
        gradient_fn: GradientFn,
        callback: CallbackFn | None = None,
    ) -> OptimizeResult:
        """Run ADAM optimization.

        Parameters
        ----------
        fun : callable
            Objective function used for loss tracking.
        x0 : np.ndarray
            Initial parameter vector.
        gradient_fn : callable
            Function returning the gradient ``grad L(theta)``.
        callback : callable or None
            Called after each iteration with a copy of the current parameter vector.

        Returns
        -------
        OptimizeResult

        Raises
        ------
        ValueError
            If ``x0`` is invalid, the objective returns a non-finite value, or
            the gradient has invalid shape/non-finite values.
        """
        theta = _validate_initial_point(x0)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        loss_history: list[float] = []
        nfev = 0
        njev = 0

        for t in range(1, self.maxiter + 1):
            grad = _evaluate_gradient(gradient_fn, theta)
            njev += 1
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            loss = _evaluate_objective(fun, theta)
            nfev += 1
            loss_history.append(loss)
            _call_callback(callback, theta)

        return OptimizeResult(
            x=theta,
            fun=loss_history[-1],
            nit=self.maxiter,
            loss_history=loss_history,
            success=True,
            nfev=nfev,
            njev=njev,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OPTIMIZERS: dict[str, type[SPSA] | type[COBYLA] | type[ADAM]] = {
    "SPSA": SPSA,
    "COBYLA": COBYLA,
    "ADAM": ADAM,
}


def get_optimizer(name: str, **kwargs: Any) -> SPSA | COBYLA | ADAM:
    """Instantiate an optimizer by name.

    Parameters
    ----------
    name : str
        One of ``"SPSA"``, ``"COBYLA"``, or ``"ADAM"`` (case-sensitive).
    **kwargs : Any
        Forwarded to the optimizer constructor.

    Returns
    -------
    SPSA | COBYLA | ADAM

    Raises
    ------
    ValueError
        If ``name`` is unknown or forwarded hyperparameters are invalid.
    """
    if name not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {list(_OPTIMIZERS)}, got '{name}'")
    return _OPTIMIZERS[name](**kwargs)
