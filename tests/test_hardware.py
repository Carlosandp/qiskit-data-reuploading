"""Tests for IBM Runtime hardware helpers."""

import sys
import types

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from qdr.hardware import list_available_backends, run_on_ibm_backend


class FakeOptions:
    def __init__(self):
        self.resilience_level = None
        self.default_shots = None
        self.seed_estimator = None


class FakeStatus:
    status_msg = "active"
    pending_jobs = 7


class FakeBackend:
    name = "ibm_fake"
    num_qubits = 5

    def status(self):
        return FakeStatus()


class FakeService:
    init_kwargs = None
    backend_name = None
    backends_kwargs = None

    def __init__(self, **kwargs):
        FakeService.init_kwargs = kwargs

    def backend(self, name):
        FakeService.backend_name = name
        return FakeBackend()

    def backends(self, **kwargs):
        FakeService.backends_kwargs = kwargs
        return [FakeBackend()]


class FakeJob:
    def __init__(self, evs):
        self._evs = evs

    def result(self):
        return [types.SimpleNamespace(data=types.SimpleNamespace(evs=self._evs))]


class FakeEstimator:
    last = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.mode = kwargs.get("mode")
        self.options = kwargs.get("options")
        self.pubs = None
        self.run_kwargs = None
        FakeEstimator.last = self

    def run(self, pubs, **kwargs):
        self.pubs = pubs
        self.run_kwargs = kwargs
        n_samples = pubs[0][2].shape[0]
        return FakeJob(np.linspace(0.0, 1.0, n_samples))


class FakePassManager:
    def run(self, circuit):
        return circuit


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = types.ModuleType("qiskit_ibm_runtime")
    runtime.EstimatorV2 = FakeEstimator
    runtime.QiskitRuntimeService = FakeService
    options = types.ModuleType("qiskit_ibm_runtime.options")
    options.EstimatorOptions = FakeOptions

    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", runtime)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime.options", options)

    import qiskit.transpiler.preset_passmanagers as preset_passmanagers

    monkeypatch.setattr(
        preset_passmanagers,
        "generate_preset_pass_manager",
        lambda optimization_level, backend: FakePassManager(),
    )
    FakeEstimator.last = None
    FakeService.init_kwargs = None
    FakeService.backend_name = None
    FakeService.backends_kwargs = None
    return runtime


@pytest.fixture
def one_parameter_problem():
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observable = SparsePauliOp("Z")
    return circuit, observable


def test_run_on_ibm_backend_uses_estimator_mode_and_options(fake_runtime, one_parameter_problem):
    circuit, observable = one_parameter_problem
    values = np.array([[0.1], [0.2]])

    evs = run_on_ibm_backend(
        circuit,
        observable,
        values,
        backend_name="ibm_fake",
        token="token",
        channel="ibm_quantum",
        optimization_level=2,
        resilience_level=2,
        default_shots=4096,
        seed_estimator=0,
    )

    np.testing.assert_allclose(evs, np.array([0.0, 1.0]))
    assert FakeService.init_kwargs == {"channel": "ibm_quantum", "token": "token"}
    assert FakeService.backend_name == "ibm_fake"
    assert FakeEstimator.last.args == ()
    assert FakeEstimator.last.mode is not None
    assert FakeEstimator.last.kwargs["mode"].name == "ibm_fake"
    assert FakeEstimator.last.options.resilience_level == 2
    assert FakeEstimator.last.options.default_shots == 4096
    assert FakeEstimator.last.options.seed_estimator == 0
    assert FakeEstimator.last.pubs[0][2].shape == (2, 1)
    assert FakeEstimator.last.run_kwargs == {}


def test_run_on_ibm_backend_accepts_single_parameter_vector(fake_runtime, one_parameter_problem):
    circuit, observable = one_parameter_problem

    evs = run_on_ibm_backend(
        circuit,
        observable,
        np.array([0.25]),
        backend_name="ibm_fake",
        precision=0.01,
        resilience_level=None,
    )

    np.testing.assert_allclose(evs, np.array([0.0]))
    assert FakeEstimator.last.options.resilience_level is None
    assert FakeEstimator.last.run_kwargs == {"precision": 0.01}


def test_run_on_ibm_backend_validates_inputs_before_runtime_import(one_parameter_problem, monkeypatch):
    circuit, observable = one_parameter_problem
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)

    with pytest.raises(ValueError, match="parameter_values must have 1 columns"):
        run_on_ibm_backend(
            circuit,
            observable,
            np.zeros((2, 2)),
            backend_name="ibm_fake",
        )


def test_run_on_ibm_backend_rejects_conflicting_precision_and_shots(one_parameter_problem):
    circuit, observable = one_parameter_problem

    with pytest.raises(ValueError, match="Use either default_shots or precision"):
        run_on_ibm_backend(
            circuit,
            observable,
            np.array([[0.1]]),
            backend_name="ibm_fake",
            default_shots=1024,
            precision=0.01,
        )


def test_run_on_ibm_backend_missing_runtime_raises(one_parameter_problem, monkeypatch):
    circuit, observable = one_parameter_problem
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)

    with pytest.raises(ImportError, match="qiskit-ibm-runtime"):
        run_on_ibm_backend(circuit, observable, np.array([[0.1]]), backend_name="ibm_fake")


def test_run_on_ibm_backend_rejects_bad_optimization_level(one_parameter_problem):
    circuit, observable = one_parameter_problem

    with pytest.raises(ValueError, match="optimization_level must be an integer in \\[0, 3\\]"):
        run_on_ibm_backend(
            circuit,
            observable,
            np.array([[0.1]]),
            backend_name="ibm_fake",
            optimization_level=4,
        )


def test_run_on_ibm_backend_rejects_bad_output_shape(fake_runtime, one_parameter_problem, monkeypatch):
    circuit, observable = one_parameter_problem

    def bad_run(self, pubs, **kwargs):
        self.pubs = pubs
        self.run_kwargs = kwargs
        return FakeJob(np.array([0.0, 1.0, 2.0]))

    monkeypatch.setattr(FakeEstimator, "run", bad_run)

    with pytest.raises(ValueError, match="Estimator returned evs with shape"):
        run_on_ibm_backend(circuit, observable, np.array([[0.1]]), backend_name="ibm_fake")


def test_list_available_backends(fake_runtime):
    rows = list_available_backends(
        token="token",
        channel="ibm_quantum",
        min_qubits=3,
        operational=False,
    )

    assert FakeService.init_kwargs == {"channel": "ibm_quantum", "token": "token"}
    assert FakeService.backends_kwargs == {"operational": False, "min_num_qubits": 3}
    assert rows == [
        {
            "name": "ibm_fake",
            "n_qubits": 5,
            "status": "active",
            "pending_jobs": 7,
        }
    ]


def test_list_available_backends_validates_inputs():
    with pytest.raises(ValueError, match="min_qubits must be >= 1"):
        list_available_backends(min_qubits=0)
    with pytest.raises(ValueError, match="operational must be a bool"):
        list_available_backends(operational="yes")
