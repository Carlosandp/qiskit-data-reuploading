"""ReuploadingFeatureMap: drop-in feature map for QML pipelines."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qdr.utils.encoding import ENCODING_GATES


class ReuploadingFeatureMap:
    """Feature map based on data re-uploading, compatible with QML pipelines.

    Unlike :class:`~qdr.circuits.DataReuploadingCircuit` this class treats
    the weights as *fixed* random parameters (frozen at construction time) and
    exposes only the *input* parameters — matching the interface expected by
    kernel-based methods and :class:`qiskit_machine_learning.kernels.FidelityQuantumKernel`.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of data-reuploading layers.
    n_features : int
        Number of classical input features.
    encoding : str, optional
        Rotation scheme.  Default ``"rx_ry_rz"``.
    entanglement : str, optional
        Entanglement pattern.  Default ``"full"``.
    seed : int or None, optional
        Random seed for weight initialisation.
    """

    _VALID_ENCODINGS = frozenset(ENCODING_GATES)
    _VALID_ENTANGLEMENTS = frozenset({"none", "linear", "circular", "full"})

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        n_features: int,
        encoding: str = "rx_ry_rz",
        entanglement: str = "full",
        seed: int | None = None,
    ) -> None:
        if encoding not in self._VALID_ENCODINGS:
            raise ValueError(f"Invalid encoding '{encoding}'")
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(f"Invalid entanglement '{entanglement}'")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.encoding = encoding
        self.entanglement = entanglement

        self._rotations: list[str] = ENCODING_GATES[encoding]
        self._n_rots: int = len(self._rotations)
        self.n_weights: int = n_layers * n_qubits * self._n_rots

        rng = np.random.default_rng(seed)
        self._fixed_weights: np.ndarray = rng.uniform(-np.pi, np.pi, self.n_weights)

        self.input_params: ParameterVector = ParameterVector("x", n_features)
        self._circuit: QuantumCircuit | None = None

    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """Build and return the feature-map circuit with fixed weights.

        Returns
        -------
        QuantumCircuit
            Parameterized only by *x* (input parameters).
        """
        from qdr.circuits.data_reuploading import DataReuploadingCircuit

        base = DataReuploadingCircuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_features=self.n_features,
            encoding=self.encoding,
            entanglement=self.entanglement,
        )
        base.build_circuit()

        # Bind weight parameters to their fixed values
        binding = dict(zip(base.weight_params, self._fixed_weights))
        self._circuit = base.circuit.assign_parameters(binding)
        # After binding, self._circuit.parameters contains only the x params
        # Store them in sorted order for consistent binding in bind()
        self.input_params = list(self._circuit.parameters)
        return self._circuit

    @property
    def circuit(self) -> QuantumCircuit:
        """The feature-map circuit (built on first access)."""
        if self._circuit is None:
            self.build_circuit()
        return self._circuit

    def bind(self, x: np.ndarray) -> QuantumCircuit:
        """Return a fully bound (non-parameterized) circuit for a single input.

        Parameters
        ----------
        x : np.ndarray
            Feature vector of shape ``(n_features,)``.

        Returns
        -------
        QuantumCircuit
            Circuit with all parameters assigned.
        """
        # self.input_params is a list of Parameter objects in circuit.parameters order
        if self._circuit is None:
            self.build_circuit()
        sorted_params = list(self._circuit.parameters)
        binding = dict(zip(sorted_params, x))
        return self._circuit.assign_parameters(binding)

    def draw(self, **kwargs):
        """Draw the feature-map circuit."""
        return self.circuit.draw(**kwargs)
