"""Hardware integration: run data re-uploading circuits on IBM Quantum backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp


def run_on_ibm_backend(
    circuit: "QuantumCircuit",
    observable: "SparsePauliOp",
    parameter_values: np.ndarray,
    backend_name: str,
    token: str | None = None,
    channel: str = "ibm_quantum",
    optimization_level: int = 1,
) -> np.ndarray:
    """Evaluate expectation values on a real IBM Quantum backend.

    Requires the optional ``hardware`` dependency group::

        pip install qiskit-data-reuploading[hardware]

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit (no measurements).
    observable : SparsePauliOp
        Observable to measure.
    parameter_values : np.ndarray
        Shape ``(n_samples, n_params)`` — one parameter set per sample.
    backend_name : str
        IBM Quantum backend name, e.g. ``"ibm_brisbane"``.
    token : str or None
        IBM Quantum API token.  If ``None``, the token is read from the saved
        account (set up with ``QiskitRuntimeService.save_account(...)``).
    channel : str, optional
        ``"ibm_quantum"`` (default) or ``"ibm_cloud"``.
    optimization_level : int, optional
        Transpiler optimisation level 0–3.  Default 1.

    Returns
    -------
    np.ndarray
        Expectation values of shape ``(n_samples,)``.

    Raises
    ------
    ImportError
        If ``qiskit-ibm-runtime`` is not installed.
    """
    try:
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
        from qiskit_ibm_runtime.options import EstimatorOptions
    except ImportError as exc:
        raise ImportError(
            "Hardware execution requires qiskit-ibm-runtime. "
            "Install it with: pip install qiskit-data-reuploading[hardware]"
        ) from exc

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(channel=channel, token=token)
    backend = service.backend(backend_name)

    # Transpile to hardware topology
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    isa_circuit = pm.run(circuit)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    options = EstimatorOptions()
    options.resilience_level = 1  # zero-noise extrapolation

    estimator = EstimatorV2(backend=backend, options=options)
    pub = (isa_circuit, isa_observable, parameter_values)
    job = estimator.run([pub])

    result = job.result()
    return result[0].data.evs


def list_available_backends(
    token: str | None = None,
    channel: str = "ibm_quantum",
    min_qubits: int = 2,
    operational: bool = True,
) -> list[dict[str, Any]]:
    """Return a list of accessible IBM Quantum backends.

    Parameters
    ----------
    token : str or None
    channel : str
    min_qubits : int
        Filter backends with fewer than this many qubits.
    operational : bool
        Only include backends currently operational.

    Returns
    -------
    list[dict]
        Each dict has keys ``name``, ``n_qubits``, ``status``.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise ImportError(
            "Requires qiskit-ibm-runtime: pip install qiskit-data-reuploading[hardware]"
        ) from exc

    service = QiskitRuntimeService(channel=channel, token=token)
    backends = service.backends(
        operational=operational,
        min_num_qubits=min_qubits,
    )
    return [
        {
            "name": b.name,
            "n_qubits": b.num_qubits,
            "status": b.status().status_msg,
        }
        for b in backends
    ]
