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

    def test_invalid_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            DataReuploadingCircuit(n_qubits=2, n_layers=0, n_features=2)

    def test_invalid_n_features_raises(self):
        with pytest.raises(ValueError, match="n_features"):
            DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=0)

    def test_too_many_features_for_encoding_slots_raises(self):
        with pytest.raises(ValueError, match="n_features=5 exceeds the number of encoding slots"):
            DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=5, encoding="rx")

    def test_non_integer_dimensions_raise(self):
        with pytest.raises(ValueError, match="n_qubits"):
            DataReuploadingCircuit(n_qubits=1.5, n_layers=2, n_features=2)

    def test_make_param_array_shape(self, tiny_weights_2q_3l):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        x = np.array([0.5, -0.3])
        arr = drc.make_param_array(tiny_weights_2q_3l, x)
        assert arr.shape == (len(drc.circuit.parameters),)

    def test_make_param_array_rejects_wrong_weight_shape(self):
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match=r"weights must have shape \(3,\), got \(2,\)"):
            drc.make_param_array(np.ones(2), np.zeros(2))

    def test_make_param_array_rejects_wrong_feature_shape(self):
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match=r"x must have shape \(2,\), got \(1,\)"):
            drc.make_param_array(np.ones(3), np.zeros(1))

    def test_make_param_array_rejects_non_finite_values(self):
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match="weights contains NaN or Inf"):
            drc.make_param_array(np.array([np.inf, 0.0, 0.0]), np.zeros(2))
        with pytest.raises(ValueError, match="x contains NaN or Inf"):
            drc.make_param_array(np.ones(3), np.array([0.0, np.nan]))

    def test_make_param_batch_shape(self, tiny_weights_2q_3l):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        X = np.random.default_rng(0).uniform(-1, 1, (5, 2))
        batch = drc.make_param_batch(tiny_weights_2q_3l, X)
        assert batch.shape == (5, len(drc.circuit.parameters))

    def test_make_param_batch_matches_single_sample_binding(self):
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=1, n_features=5, encoding="rx_ry_rz")
        weights = np.linspace(-0.5, 0.5, drc.n_weights)
        X = np.random.default_rng(0).uniform(-1, 1, (4, 5))
        batch = drc.make_param_batch(weights, X)
        expected = np.vstack([drc.make_param_array(weights, x) for x in X])
        np.testing.assert_allclose(batch, expected)

    def test_make_param_batch_rejects_wrong_feature_count(self):
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match="X must have 2 features"):
            drc.make_param_batch(np.ones(3), np.zeros((3, 3)))

    def test_make_param_batch_rejects_non_2d_input(self):
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match="X must be a 2D array"):
            drc.make_param_batch(np.ones(3), np.zeros(2))

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

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError, match="n_features"):
            ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=0)

    def test_too_many_features_for_feature_map_slots_raises(self):
        with pytest.raises(ValueError, match="n_features=5 exceeds the number of encoding slots"):
            ReuploadingFeatureMap(n_qubits=1, n_layers=1, n_features=5, encoding="rx")

    def test_only_input_params_remain(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=3, seed=0)
        fm.build_circuit()
        # After binding fixed weights, only x[0..2] should remain
        param_names = {p.name for p in fm.circuit.parameters}
        assert all(name.startswith("x") for name in param_names)
        assert len(fm.circuit.parameters) == 3

    def test_input_params_remains_parameter_vector_after_build(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=3, seed=0)
        fm.build_circuit()
        assert isinstance(fm.input_params, ParameterVector)

    def test_input_params_are_the_circuit_parameters(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=1, n_features=5, encoding="rx_ry_rz", seed=0)
        fm.build_circuit()
        input_param_set = set(fm.input_params)
        assert set(fm.circuit.parameters) == input_param_set

    def test_bind_returns_bound_circuit(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        x = np.array([0.1, -0.2])
        bound = fm.bind(x)
        assert len(bound.parameters) == 0  # fully bound

    def test_bind_accepts_full_feature_vector(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=1, n_features=5, encoding="rx_ry_rz", seed=1)
        bound = fm.bind(np.arange(5, dtype=float))
        assert len(bound.parameters) == 0

    def test_bind_rejects_wrong_shape(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        with pytest.raises(ValueError, match=r"x must have shape \(2,\), got \(1,\)"):
            fm.bind(np.array([0.1]))

    def test_bind_rejects_non_finite_values(self):
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        with pytest.raises(ValueError, match="x contains NaN or Inf"):
            fm.bind(np.array([0.1, np.inf]))
