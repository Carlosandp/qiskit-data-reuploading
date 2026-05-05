"""Tests for DataReuploadingCircuit and ReuploadingFeatureMap."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qdr.circuits import DataReuploadingCircuit, ReuploadingFeatureMap


class TestDataReuploadingCircuit:
    def test_build_returns_quantum_circuit(self):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2)
        qc = drc.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_n_qubits(self):
        drc = DataReuploadingCircuit(n_qubits=3, n_layers=1, n_features=2)
        assert drc.circuit.num_qubits == 3

    def test_weight_count_rx_ry_rz(self):
        # n_layers=2, n_qubits=2, 3 rotations → 12 weights
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, encoding="rx_ry_rz")
        assert drc.n_weights == 12
        assert len(list(drc.weight_params)) == 12

    def test_weight_count_rx(self):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2, encoding="rx")
        assert drc.n_weights == 6

    def test_total_parameters_in_circuit(self):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=3, encoding="rx_ry_rz")
        drc.build_circuit()
        # 12 weights + 3 inputs = 15
        assert len(drc.circuit.parameters) == 15

    @pytest.mark.parametrize("encoding", ["rx", "ry", "rz", "rx_ry_rz"])
    def test_all_encodings_build(self, encoding):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, encoding=encoding)
        qc = drc.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    @pytest.mark.parametrize("entanglement", ["none", "linear", "circular", "full"])
    def test_all_entanglements_build(self, entanglement):
        drc = DataReuploadingCircuit(
            n_qubits=3, n_layers=2, n_features=2, entanglement=entanglement
        )
        qc = drc.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_invalid_encoding_raises(self):
        with pytest.raises(ValueError, match="encoding"):
            DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, encoding="bad")

    def test_invalid_entanglement_raises(self):
        with pytest.raises(ValueError, match="entanglement"):
            DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, entanglement="weird")

    def test_invalid_n_qubits_raises(self):
        with pytest.raises(ValueError):
            DataReuploadingCircuit(n_qubits=0, n_layers=2, n_features=2)

    def test_make_param_array_shape(self, tiny_weights_2q_3l):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        x = np.array([0.5, -0.3])
        arr = drc.make_param_array(tiny_weights_2q_3l, x)
        assert arr.shape == (len(drc.circuit.parameters),)

    def test_make_param_batch_shape(self, tiny_weights_2q_3l):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        X = np.random.default_rng(0).uniform(-1, 1, (5, 2))
        batch = drc.make_param_batch(tiny_weights_2q_3l, X)
        assert batch.shape == (5, len(drc.circuit.parameters))

    def test_circuit_property_builds_lazily(self):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2)
        assert drc._circuit is None
        _ = drc.circuit
        assert drc._circuit is not None

    def test_single_qubit_no_entanglement(self):
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=3, n_features=1)
        drc.build_circuit()
        assert drc.circuit.num_qubits == 1

    def test_repr(self):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        r = repr(drc)
        assert "DataReuploadingCircuit" in r
        assert "n_qubits=2" in r


class TestReuploadingFeatureMap:
    def test_build_circuit(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2)
        qc = fm.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_only_input_params_remain(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=3, seed=0)
        fm.build_circuit()
        # After binding fixed weights, only x[0..2] should remain
        param_names = {p.name for p in fm.circuit.parameters}
        assert all(name.startswith("x") for name in param_names)
        assert len(fm.circuit.parameters) == 3

    def test_bind_returns_bound_circuit(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        x = np.array([0.1, -0.2])
        bound = fm.bind(x)
        assert len(bound.parameters) == 0  # fully bound
