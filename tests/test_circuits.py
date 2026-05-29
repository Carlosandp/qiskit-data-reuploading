"""Tests for DataReuploadingCircuit and ReuploadingFeatureMap."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qdr.circuits import DataReuploadingCircuit, ReuploadingFeatureMap


class TestDataReuploadingCircuit:
    """Tests for :class:`DataReuploadingCircuit` construction and parameter binding."""

    def test_build_returns_quantum_circuit(self):
        """build_circuit() returns a QuantumCircuit instance."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2)
        qc = drc.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_n_qubits(self):
        """Built circuit has the requested qubit count."""
        drc = DataReuploadingCircuit(n_qubits=3, n_layers=1, n_features=2)
        assert drc.circuit.num_qubits == 3

    def test_weight_count_rx_ry_rz(self):
        """rx_ry_rz encoding yields 3 weights per qubit per layer."""
        # n_layers=2, n_qubits=2, 3 rotations → 12 weights
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, encoding="rx_ry_rz")
        assert drc.n_weights == 12
        assert len(list(drc.weight_params)) == 12

    def test_weight_count_rx(self):
        """rx encoding yields 1 weight per qubit per layer."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2, encoding="rx")
        assert drc.n_weights == 6

    def test_total_parameters_in_circuit(self):
        """Circuit has n_weights + n_features total parameters."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=3, encoding="rx_ry_rz")
        drc.build_circuit()
        # 12 weights + 3 inputs = 15
        assert len(drc.circuit.parameters) == 15

    @pytest.mark.parametrize("encoding", ["rx", "ry", "rz", "rx_ry_rz"])
    def test_all_encodings_build(self, encoding):
        """All supported encoding schemes produce a valid circuit."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, encoding=encoding)
        qc = drc.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    @pytest.mark.parametrize("entanglement", ["none", "linear", "circular", "full"])
    def test_all_entanglements_build(self, entanglement):
        """All supported entanglement patterns produce a valid circuit."""
        drc = DataReuploadingCircuit(
            n_qubits=3, n_layers=2, n_features=2, entanglement=entanglement
        )
        qc = drc.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_invalid_encoding_raises(self):
        """An unknown encoding name raises ValueError."""
        with pytest.raises(ValueError, match="encoding"):
            DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, encoding="bad")

    def test_invalid_entanglement_raises(self):
        """An unknown entanglement name raises ValueError."""
        with pytest.raises(ValueError, match="entanglement"):
            DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2, entanglement="weird")

    def test_invalid_n_qubits_raises(self):
        """n_qubits=0 raises ValueError."""
        with pytest.raises(ValueError):
            DataReuploadingCircuit(n_qubits=0, n_layers=2, n_features=2)

    def test_invalid_n_layers_raises(self):
        """n_layers=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_layers"):
            DataReuploadingCircuit(n_qubits=2, n_layers=0, n_features=2)

    def test_invalid_n_features_raises(self):
        """n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features"):
            DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=0)

    def test_too_many_features_for_encoding_slots_raises(self):
        """More features than encoding slots raises ValueError."""
        with pytest.raises(ValueError, match="n_features=5 exceeds the number of encoding slots"):
            DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=5, encoding="rx")

    def test_non_integer_dimensions_raise(self):
        """Non-integer n_qubits raises ValueError."""
        with pytest.raises(ValueError, match="n_qubits"):
            DataReuploadingCircuit(n_qubits=1.5, n_layers=2, n_features=2)

    def test_make_param_array_shape(self, tiny_weights_2q_3l):
        """make_param_array returns an array of length n_params."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        x = np.array([0.5, -0.3])
        arr = drc.make_param_array(tiny_weights_2q_3l, x)
        assert arr.shape == (len(drc.circuit.parameters),)

    def test_make_param_array_rejects_wrong_weight_shape(self):
        """make_param_array raises ValueError for wrong weight shape."""
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match=r"weights must have shape \(3,\), got \(2,\)"):
            drc.make_param_array(np.ones(2), np.zeros(2))

    def test_make_param_array_rejects_wrong_feature_shape(self):
        """make_param_array raises ValueError for wrong feature vector shape."""
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match=r"x must have shape \(2,\), got \(1,\)"):
            drc.make_param_array(np.ones(3), np.zeros(1))

    def test_make_param_array_rejects_non_finite_values(self):
        """make_param_array raises ValueError for non-finite inputs."""
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match="weights contains NaN or Inf"):
            drc.make_param_array(np.array([np.inf, 0.0, 0.0]), np.zeros(2))
        with pytest.raises(ValueError, match="x contains NaN or Inf"):
            drc.make_param_array(np.ones(3), np.array([0.0, np.nan]))

    def test_make_param_batch_shape(self, tiny_weights_2q_3l):
        """make_param_batch returns an array of shape (n_samples, n_params)."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        X = np.random.default_rng(0).uniform(-1, 1, (5, 2))
        batch = drc.make_param_batch(tiny_weights_2q_3l, X)
        assert batch.shape == (5, len(drc.circuit.parameters))

    def test_make_param_batch_matches_single_sample_binding(self):
        """Batch binding matches stacking individual make_param_array calls."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=1, n_features=5, encoding="rx_ry_rz")
        weights = np.linspace(-0.5, 0.5, drc.n_weights)
        X = np.random.default_rng(0).uniform(-1, 1, (4, 5))
        batch = drc.make_param_batch(weights, X)
        expected = np.vstack([drc.make_param_array(weights, x) for x in X])
        np.testing.assert_allclose(batch, expected)

    def test_make_param_batch_rejects_wrong_feature_count(self):
        """make_param_batch raises ValueError when X has the wrong number of features."""
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match="X must have 2 features"):
            drc.make_param_batch(np.ones(3), np.zeros((3, 3)))

    def test_make_param_batch_rejects_non_2d_input(self):
        """make_param_batch raises ValueError for a 1D X array."""
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=2)
        with pytest.raises(ValueError, match="X must be a 2D array"):
            drc.make_param_batch(np.ones(3), np.zeros(2))

    def test_circuit_property_builds_lazily(self):
        """The circuit property triggers build_circuit on first access."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=2, n_features=2)
        assert drc._circuit is None
        _ = drc.circuit
        assert drc._circuit is not None

    def test_single_qubit_no_entanglement(self):
        """Single-qubit circuit builds without entanglement gates."""
        drc = DataReuploadingCircuit(n_qubits=1, n_layers=3, n_features=1)
        drc.build_circuit()
        assert drc.circuit.num_qubits == 1

    def test_repr(self):
        """repr includes class name and key hyperparameters."""
        drc = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
        r = repr(drc)
        assert "DataReuploadingCircuit" in r
        assert "n_qubits=2" in r


class TestReuploadingFeatureMap:
    """Tests for :class:`ReuploadingFeatureMap` circuit construction and binding."""

    def test_build_circuit(self):
        """build_circuit() returns a QuantumCircuit with only input parameters."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2)
        qc = fm.build_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_invalid_dimensions_raise(self):
        """n_features=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features"):
            ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=0)

    def test_too_many_features_for_feature_map_slots_raises(self):
        """More features than slots raises ValueError."""
        with pytest.raises(ValueError, match="n_features=5 exceeds the number of encoding slots"):
            ReuploadingFeatureMap(n_qubits=1, n_layers=1, n_features=5, encoding="rx")

    def test_only_input_params_remain(self):
        """After build, only x parameters remain in the circuit."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=3, seed=0)
        fm.build_circuit()
        # After binding fixed weights, only x[0..2] should remain
        param_names = {p.name for p in fm.circuit.parameters}
        assert all(name.startswith("x") for name in param_names)
        assert len(fm.circuit.parameters) == 3

    def test_input_params_remains_parameter_vector_after_build(self):
        """input_params attribute is still a ParameterVector after build."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=3, seed=0)
        fm.build_circuit()
        assert isinstance(fm.input_params, ParameterVector)

    def test_input_params_are_the_circuit_parameters(self):
        """input_params objects match the circuit's parameter set exactly."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=1, n_features=5, encoding="rx_ry_rz", seed=0)
        fm.build_circuit()
        input_param_set = set(fm.input_params)
        assert set(fm.circuit.parameters) == input_param_set

    def test_bind_returns_bound_circuit(self):
        """bind() returns a circuit with no free parameters."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        x = np.array([0.1, -0.2])
        bound = fm.bind(x)
        assert len(bound.parameters) == 0  # fully bound

    def test_bind_accepts_full_feature_vector(self):
        """bind() works when n_features matches the full vector length."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=1, n_features=5, encoding="rx_ry_rz", seed=1)
        bound = fm.bind(np.arange(5, dtype=float))
        assert len(bound.parameters) == 0

    def test_bind_rejects_wrong_shape(self):
        """bind() raises ValueError for a feature vector with the wrong length."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        with pytest.raises(ValueError, match=r"x must have shape \(2,\), got \(1,\)"):
            fm.bind(np.array([0.1]))

    def test_bind_rejects_non_finite_values(self):
        """bind() raises ValueError when the feature vector contains Inf."""
        fm = ReuploadingFeatureMap(n_qubits=2, n_layers=2, n_features=2, seed=1)
        with pytest.raises(ValueError, match="x contains NaN or Inf"):
            fm.bind(np.array([0.1, np.inf]))
