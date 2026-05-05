"""DataReuploadingCircuit: core parameterized quantum circuit for data re-uploading."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qdr.utils.encoding import ENCODING_GATES


class DataReuploadingCircuit:
    """Parameterized quantum circuit implementing the data re-uploading technique.

    Each layer applies rotation gates whose angles are the sum of a trainable
    weight and a (cyclically indexed) input feature, followed by entanglement
    gates.  Stacking *n_layers* of such blocks allows the circuit to act as a
    universal quantum classifier (Pérez-Salinas et al., 2019).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of data-reuploading layers.
    n_features : int
        Number of classical input features.
    encoding : str, optional
        Rotation scheme — ``"rx"``, ``"ry"``, ``"rz"``, or ``"rx_ry_rz"``
        (default).
    entanglement : str, optional
        Entanglement pattern — ``"none"``, ``"linear"``, ``"circular"``, or
        ``"full"`` (default).
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
    ) -> None:
        if encoding not in self._VALID_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {sorted(self._VALID_ENCODINGS)}, got '{encoding}'"
            )
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENTS)}, got '{entanglement}'"
            )
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.encoding = encoding
        self.entanglement = entanglement

        self._rotations: list[str] = ENCODING_GATES[encoding]
        self._n_rots: int = len(self._rotations)

        # n_weights = layers × qubits × rotations-per-qubit
        self.n_weights: int = n_layers * n_qubits * self._n_rots

        # 'w' sorts before 'x', so weights precede inputs in circuit.parameters
        self.weight_params: ParameterVector = ParameterVector("w", self.n_weights)
        self.input_params: ParameterVector = ParameterVector("x", n_features)

        self._circuit: QuantumCircuit | None = None

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """Build and return the parameterized quantum circuit.

        Returns
        -------
        QuantumCircuit
            The data re-uploading circuit (no measurements).
        """
        qc = QuantumCircuit(self.n_qubits)
        weight_idx = 0

        for layer in range(self.n_layers):
            # --- encoding block ---
            for qubit in range(self.n_qubits):
                for rot_idx, rot in enumerate(self._rotations):
                    feat_idx = (
                        layer * self.n_qubits * self._n_rots
                        + qubit * self._n_rots
                        + rot_idx
                    ) % self.n_features
                    angle = self.weight_params[weight_idx] + self.input_params[feat_idx]
                    getattr(qc, rot)(angle, qubit)
                    weight_idx += 1

            # --- entanglement block ---
            if self.n_qubits > 1:
                self._add_entanglement(qc)

        self._circuit = qc
        return qc

    def _add_entanglement(self, qc: QuantumCircuit) -> None:
        n = self.n_qubits
        if self.entanglement == "none":
            return
        elif self.entanglement == "linear":
            for i in range(n - 1):
                qc.cx(i, i + 1)
        elif self.entanglement == "circular":
            for i in range(n - 1):
                qc.cx(i, i + 1)
            if n > 2:
                qc.cx(n - 1, 0)
        elif self.entanglement == "full":
            for i in range(n):
                for j in range(i + 1, n):
                    qc.cx(i, j)

    # ------------------------------------------------------------------
    # Parameter binding helpers
    # ------------------------------------------------------------------

    def make_param_array(self, weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return a parameter array for a *single* sample in ``circuit.parameters`` order.

        Parameters
        ----------
        weights : np.ndarray
            Shape ``(n_weights,)``.
        x : np.ndarray
            Shape ``(n_features,)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_params,)``.
        """
        if self._circuit is None:
            self.build_circuit()
        sorted_params = list(self._circuit.parameters)
        w_dict = dict(zip(self.weight_params, weights))
        x_dict = dict(zip(self.input_params, x))
        full_dict = {**w_dict, **x_dict}
        return np.array([full_dict[p] for p in sorted_params], dtype=float)

    def make_param_batch(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return a batched parameter array for *multiple* samples.

        Parameters
        ----------
        weights : np.ndarray
            Shape ``(n_weights,)``.
        X : np.ndarray
            Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_params)``.
        """
        if self._circuit is None:
            self.build_circuit()
        sorted_params = list(self._circuit.parameters)
        w_dict = dict(zip(self.weight_params, weights))
        n_samples, n_params = X.shape[0], len(sorted_params)
        param_values = np.empty((n_samples, n_params), dtype=float)
        for i, x_i in enumerate(X):
            full_dict = {**w_dict, **dict(zip(self.input_params, x_i))}
            param_values[i] = [full_dict[p] for p in sorted_params]
        return param_values

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def circuit(self) -> QuantumCircuit:
        """The built :class:`~qiskit.QuantumCircuit`, building it on first access."""
        if self._circuit is None:
            self.build_circuit()
        return self._circuit

    def draw(self, **kwargs):
        """Draw the circuit (delegates to :meth:`QuantumCircuit.draw`)."""
        return self.circuit.draw(**kwargs)

    def __repr__(self) -> str:
        return (
            f"DataReuploadingCircuit(n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, n_features={self.n_features}, "
            f"encoding='{self.encoding}', entanglement='{self.entanglement}')"
        )
