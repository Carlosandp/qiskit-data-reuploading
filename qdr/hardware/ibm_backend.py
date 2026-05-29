"""Hardware integration: run data re-uploading circuits on IBM Quantum backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp


def _validate_nonempty_string(name: str, value: str) -> str:
    """Validate that a value is a non-empty, non-blank string.

    Args:
        name: Parameter name used in error messages.
        value: Candidate string to validate.

    Returns:
        The original string value.

    Raises:
        ValueError: If value is not a string or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
    return value


def _validate_optional_token(token: str | None) -> str | None:
    """Validate an optional IBM Quantum API token string.

    Args:
        token: API token or ``None`` to use the saved account.

    Returns:
        The token unchanged, or ``None``.

    Raises:
        ValueError: If token is provided but blank.
    """
    if token is None:
        return None
    return _validate_nonempty_string("token", token)


def _validate_optimization_level(optimization_level: int) -> int:
    """Validate that the Qiskit transpiler optimization level is in [0, 3].

    Args:
        optimization_level: Candidate level value.

    Returns:
        The level cast to a plain Python int.

    Raises:
        ValueError: If the value is not an integer in [0, 3].
    """
    if isinstance(optimization_level, bool) or not isinstance(optimization_level, Integral):
        raise ValueError(
            f"optimization_level must be an integer in [0, 3], got {optimization_level!r}."
        )
    optimization_level = int(optimization_level)
    if not 0 <= optimization_level <= 3:
        raise ValueError(
            f"optimization_level must be an integer in [0, 3], got {optimization_level}."
        )
    return optimization_level


def _validate_resilience_level(resilience_level: int | None) -> int | None:
    """Validate that the IBM Runtime resilience level is 0, 1, 2, or None.

    Args:
        resilience_level: Candidate resilience level or ``None``.

    Returns:
        The level cast to int, or ``None``.

    Raises:
        ValueError: If the value is not one of {0, 1, 2, None}.
    """
    if resilience_level is None:
        return None
    if isinstance(resilience_level, bool) or not isinstance(resilience_level, Integral):
        raise ValueError(f"resilience_level must be one of 0, 1, 2, or None; got {resilience_level!r}.")
    resilience_level = int(resilience_level)
    if resilience_level not in {0, 1, 2}:
        raise ValueError(f"resilience_level must be one of 0, 1, 2, or None; got {resilience_level}.")
    return resilience_level


def _validate_positive_int(name: str, value: int | None) -> int | None:
    """Validate that an optional integer parameter is positive.

    Args:
        name: Parameter name used in error messages.
        value: Candidate value or ``None``.

    Returns:
        The value cast to int, or ``None``.

    Raises:
        ValueError: If value is provided but is not a positive integer.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _validate_nonnegative_int(name: str, value: int | None) -> int | None:
    """Validate that an optional integer parameter is non-negative.

    Args:
        name: Parameter name used in error messages.
        value: Candidate value or ``None``.

    Returns:
        The value cast to int, or ``None``.

    Raises:
        ValueError: If value is provided but is not a non-negative integer.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def _validate_precision(precision: float | None) -> float | None:
    """Validate that the optional precision argument is a finite positive number.

    Args:
        precision: Target precision value or ``None``.

    Returns:
        The value cast to float, or ``None``.

    Raises:
        ValueError: If precision is provided but is not a finite positive number.
    """
    if precision is None:
        return None
    if isinstance(precision, bool) or not isinstance(precision, Real) or not np.isfinite(precision):
        raise ValueError(f"precision must be a finite positive number or None, got {precision!r}.")
    precision = float(precision)
    if precision <= 0.0:
        raise ValueError(f"precision must be > 0, got {precision}.")
    return precision


def _num_parameters(circuit: "QuantumCircuit") -> int:
    """Return the number of free parameters in a Qiskit circuit.

    Args:
        circuit: A Qiskit ``QuantumCircuit``.

    Returns:
        The number of unbound parameters as a plain Python int.
    """
    num_parameters = getattr(circuit, "num_parameters", None)
    if num_parameters is not None:
        return int(num_parameters)
    return len(getattr(circuit, "parameters"))


def _validate_parameter_values(
    circuit: "QuantumCircuit",
    parameter_values: np.ndarray,
) -> np.ndarray:
    """Validate and reshape the parameter value array against a circuit.

    Args:
        circuit: Circuit whose parameter count sets the expected column count.
        parameter_values: 1D (single sample) or 2D (batch) parameter array.

    Returns:
        A float64 array of shape ``(n_samples, n_params)``.

    Raises:
        ValueError: If the array has the wrong number of dimensions, wrong
            column count, zero rows, or contains non-finite values.
    """
    values = np.asarray(parameter_values, dtype=float)
    expected_params = _num_parameters(circuit)
    if values.ndim == 1:
        if values.shape[0] != expected_params:
            raise ValueError(
                "1D parameter_values represents one sample and must have "
                f"{expected_params} entries, got {values.shape[0]}."
            )
        values = values.reshape(1, expected_params)
    elif values.ndim != 2:
        raise ValueError(
            f"parameter_values must be a 1D or 2D array, got parameter_values.ndim={values.ndim}."
        )
    if values.shape[0] < 1:
        raise ValueError("parameter_values must contain at least one parameter set.")
    if values.shape[1] != expected_params:
        raise ValueError(
            f"parameter_values must have {expected_params} columns, got {values.shape[1]}."
        )
    if np.any(~np.isfinite(values)):
        raise ValueError("parameter_values contains NaN or Inf values; all values must be finite.")
    return values


def _validate_run_options(
    circuit: "QuantumCircuit",
    parameter_values: np.ndarray,
    backend_name: str,
    token: str | None,
    channel: str,
    optimization_level: int,
    resilience_level: int | None,
    default_shots: int | None,
    seed_estimator: int | None,
    precision: float | None,
) -> tuple[np.ndarray, str, str | None, str, int, int | None, int | None, int | None, float | None]:
    """Validate all arguments of :func:`run_on_ibm_backend` before any I/O.

    Args:
        circuit: Parameterized circuit without measurements.
        parameter_values: Parameter array to validate against the circuit.
        backend_name: IBM Quantum backend name.
        token: Optional API token.
        channel: Runtime channel name.
        optimization_level: Qiskit transpiler optimization level.
        resilience_level: IBM Runtime resilience level or ``None``.
        default_shots: Total shots per circuit or ``None``.
        seed_estimator: Estimator random seed or ``None``.
        precision: Target precision or ``None``.

    Returns:
        A tuple of validated, coerced argument values in the same order.

    Raises:
        ValueError: If any argument is invalid or ``default_shots`` and
            ``precision`` are both specified.
    """
    values = _validate_parameter_values(circuit, parameter_values)
    backend_name = _validate_nonempty_string("backend_name", backend_name)
    token = _validate_optional_token(token)
    channel = _validate_nonempty_string("channel", channel)
    optimization_level = _validate_optimization_level(optimization_level)
    resilience_level = _validate_resilience_level(resilience_level)
    default_shots = _validate_nonnegative_int("default_shots", default_shots)
    seed_estimator = _validate_nonnegative_int("seed_estimator", seed_estimator)
    precision = _validate_precision(precision)
    if default_shots is not None and precision is not None:
        raise ValueError("Use either default_shots or precision, not both; default_shots overrides precision.")
    return (
        values,
        backend_name,
        token,
        channel,
        optimization_level,
        resilience_level,
        default_shots,
        seed_estimator,
        precision,
    )


def _load_runtime():
    """Import qiskit-ibm-runtime classes or raise a helpful ImportError.

    Returns:
        A tuple ``(EstimatorV2, QiskitRuntimeService, EstimatorOptions)``.

    Raises:
        ImportError: If ``qiskit-ibm-runtime`` is not installed.
    """
    try:
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
        from qiskit_ibm_runtime.options import EstimatorOptions
    except ImportError as exc:
        raise ImportError(
            "Hardware execution requires qiskit-ibm-runtime. "
            "Install it with: pip install qiskit-data-reuploading[hardware]"
        ) from exc
    return EstimatorV2, QiskitRuntimeService, EstimatorOptions


def _estimator_options(
    EstimatorOptions: type,
    resilience_level: int | None,
    default_shots: int | None,
    seed_estimator: int | None,
) -> Any:
    """Build an ``EstimatorOptions`` object from optional keyword arguments.

    Args:
        EstimatorOptions: The options class from ``qiskit_ibm_runtime``.
        resilience_level: Resilience level to set, or ``None`` to leave unset.
        default_shots: Shot count to set, or ``None`` to leave unset.
        seed_estimator: Estimator seed to set, or ``None`` to leave unset.

    Returns:
        A configured ``EstimatorOptions`` instance.
    """
    options = EstimatorOptions()
    if resilience_level is not None:
        options.resilience_level = resilience_level
    if default_shots is not None:
        options.default_shots = default_shots
    if seed_estimator is not None:
        options.seed_estimator = seed_estimator
    return options


def _backend_name(backend: Any) -> str:
    """Extract the name string from an IBM backend object.

    Args:
        backend: An IBM Quantum backend object (callable or attribute ``.name``).

    Returns:
        The backend name as a plain string.
    """
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    return str(name)


def run_on_ibm_backend(
    circuit: "QuantumCircuit",
    observable: "SparsePauliOp",
    parameter_values: np.ndarray,
    backend_name: str,
    token: str | None = None,
    channel: str = "ibm_quantum",
    optimization_level: int = 1,
    resilience_level: int | None = 1,
    default_shots: int | None = None,
    seed_estimator: int | None = None,
    precision: float | None = None,
) -> np.ndarray:
    """Evaluate expectation values on an IBM Quantum backend.

    Requires the optional ``hardware`` dependency group::

        pip install qiskit-data-reuploading[hardware]

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit without measurements.
    observable : SparsePauliOp
        Observable to estimate.
    parameter_values : np.ndarray
        Parameter values. Shape ``(n_samples, n_params)`` for a batch, or
        ``(n_params,)`` for one sample.
    backend_name : str
        IBM Quantum backend name, e.g. ``"ibm_brisbane"``.
    token : str or None, optional
        IBM Quantum API token. If ``None``, the token is read from the saved
        account configured with ``QiskitRuntimeService.save_account(...)``.
    channel : str, optional
        Runtime channel. Default ``"ibm_quantum"``.
    optimization_level : int, optional
        Transpiler optimization level, from 0 to 3. Default 1.
    resilience_level : int or None, optional
        Estimator resilience level. IBM Runtime supports 0, 1, and 2; ``None``
        leaves the server default unset. Default 1.
    default_shots : int or None, optional
        Non-negative total shots per circuit/configuration. Mutually exclusive with
        ``precision``.
    seed_estimator : int or None, optional
        Runtime estimator seed.
    precision : float or None, optional
        Target precision passed to ``EstimatorV2.run``. Mutually exclusive with
        ``default_shots``.

    Returns
    -------
    np.ndarray
        Expectation values of shape ``(n_samples,)``.

    Raises
    ------
    ImportError
        If ``qiskit-ibm-runtime`` is not installed.
    ValueError
        If inputs are malformed or estimator output has an unexpected shape.

    Notes
    -----
    This function submits real or runtime-managed backend jobs and can incur
    queue time and account cost. It intentionally does not train models; use it
    for explicit hardware evaluation of already-built circuits/observables.
    """
    (
        values,
        backend_name,
        token,
        channel,
        optimization_level,
        resilience_level,
        default_shots,
        seed_estimator,
        precision,
    ) = _validate_run_options(
        circuit,
        parameter_values,
        backend_name,
        token,
        channel,
        optimization_level,
        resilience_level,
        default_shots,
        seed_estimator,
        precision,
    )
    EstimatorV2, QiskitRuntimeService, EstimatorOptions = _load_runtime()

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(channel=channel, token=token)
    backend = service.backend(backend_name)

    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    isa_circuit = pm.run(circuit)
    if _num_parameters(isa_circuit) != values.shape[1]:
        raise ValueError(
            "Transpilation changed the number of circuit parameters: "
            f"expected {values.shape[1]}, got {_num_parameters(isa_circuit)}."
        )
    isa_observable = observable.apply_layout(isa_circuit.layout)

    options = _estimator_options(
        EstimatorOptions,
        resilience_level=resilience_level,
        default_shots=default_shots,
        seed_estimator=seed_estimator,
    )

    estimator = EstimatorV2(mode=backend, options=options)
    pub = (isa_circuit, isa_observable, values)
    job = estimator.run([pub], precision=precision) if precision is not None else estimator.run([pub])

    result = job.result()
    evs = np.asarray(result[0].data.evs, dtype=float).reshape(-1)
    if evs.shape != (values.shape[0],):
        raise ValueError(
            f"Estimator returned evs with shape {evs.shape}, expected {(values.shape[0],)}."
        )
    if np.any(~np.isfinite(evs)):
        raise ValueError("Estimator returned NaN or Inf expectation values.")
    return evs


def list_available_backends(
    token: str | None = None,
    channel: str = "ibm_quantum",
    min_qubits: int = 2,
    operational: bool = True,
) -> list[dict[str, Any]]:
    """Return accessible IBM Quantum backends.

    Parameters
    ----------
    token : str or None, optional
        IBM Quantum API token. If ``None``, uses the saved account.
    channel : str, optional
        Runtime channel. Default ``"ibm_quantum"``.
    min_qubits : int, optional
        Filter backends with fewer than this many qubits. Default 2.
    operational : bool, optional
        Only include currently operational backends. Default ``True``.

    Returns
    -------
    list[dict[str, Any]]
        Each dict has ``name``, ``n_qubits``, ``status``, and ``pending_jobs``.

    Raises
    ------
    ImportError
        If ``qiskit-ibm-runtime`` is not installed.
    ValueError
        If inputs are malformed.
    """
    token = _validate_optional_token(token)
    channel = _validate_nonempty_string("channel", channel)
    min_qubits = _validate_positive_int("min_qubits", min_qubits)
    if not isinstance(operational, bool):
        raise ValueError(f"operational must be a bool, got {operational!r}.")

    _, QiskitRuntimeService, _ = _load_runtime()
    service = QiskitRuntimeService(channel=channel, token=token)
    backends = service.backends(
        operational=operational,
        min_num_qubits=min_qubits,
    )
    rows: list[dict[str, Any]] = []
    for backend in backends:
        status = backend.status()
        rows.append(
            {
                "name": _backend_name(backend),
                "n_qubits": int(getattr(backend, "num_qubits")),
                "status": str(getattr(status, "status_msg", status)),
                "pending_jobs": getattr(status, "pending_jobs", None),
            }
        )
    return rows
