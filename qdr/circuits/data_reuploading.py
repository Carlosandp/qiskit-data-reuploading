"""DataReuploadingCircuit: core parameterized quantum circuit for data re-uploading."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qdr.utils.encoding import ENCODING_GATES


class DataReuploadingCircuit:
    """Parameterized quantum circuit implementing the data re-uploading technique.

    Each layer applies rotation gates whose angles are the sum of a trainable
    weight and a (cyclically indexed) input feature, followed by entanglement
    gates.  Stacking *n_layers* of such blocks allows the circuit to act as a
    universal quantum classifier (Perez-Salinas et al., 2020).

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

    Notes
    -----
    The angle of each rotation gate in layer ``l``, qubit ``i``, and rotation
    axis ``r`` is

        ``angle = w[l, i, r] + x[feat_idx(l, i, r)]``

    where ``feat_idx`` cycles modulo ``n_features``; see
    :meth:`build_circuit` for the exact assignment rule.  The circuit requires
    ``n_features <= n_layers * n_qubits * n_rotations`` so every declared input
    feature is represented by at least one gate.
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
        n_qubits = self._validate_positive_int("n_qubits", n_qubits)
        n_layers = self._validate_positive_int("n_layers", n_layers)
        n_features = self._validate_positive_int("n_features", n_features)
        if encoding not in self._VALID_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {sorted(self._VALID_ENCODINGS)}, got '{encoding}'"
            )
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {sorted(self._VALID_ENTANGLEMENTS)}, got '{entanglement}'"
            )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.encoding = encoding
        self.entanglement = entanglement

        self._rotations: tuple[str, ...] = ENCODING_GATES[encoding]
        self._n_rots: int = len(self._rotations)

        # n_weights = layers × qubits × rotations-per-qubit
        self.n_weights: int = n_layers * n_qubits * self._n_rots
        if n_features > self.n_weights:
            raise ValueError(
                f"n_features={n_features} exceeds the number of encoding slots "
                f"({self.n_weights}). Increase n_layers, n_qubits, or encoding "
                "richness so every feature is uploaded at least once."
            )

        # 'w' sorts before 'x', so weights precede inputs in circuit.parameters
        self.weight_params: ParameterVector = ParameterVector("w", self.n_weights)
        self.input_params: ParameterVector = ParameterVector("x", n_features)

        self._circuit: QuantumCircuit | None = None

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")
        return value

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """Build and return the parameterized quantum circuit.

        Features are assigned deterministically and cyclically to the encoding
        slots:

            ``feat_idx = slot_idx % n_features``

        where ``slot_idx = layer * n_qubits * n_rots + qubit * n_rots + rot_idx``.
        If ``n_features < n_slots``, features are reused. For example,
        ``n_features=4`` and ``n_slots=18`` uses each feature 4 or 5 times.
        With ``n_features=30`` and ``n_slots=36``, features 0-5 are used twice
        and features 6-29 once, so the first features receive one extra slot.
        Configurations with ``n_features > n_slots`` are rejected at
        construction time because they would leave trailing features outside
        the circuit.

        Returns
        -------
        QuantumCircuit
            The data re-uploading circuit (no measurements).

        Notes
        -----
        The angle of each rotation gate is ``angle = w + x``. Data and
        trainable weights are mixed inside the same gate, which is the defining
        feature of data re-uploading.
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

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        expected_shape = (self.n_weights,)
        if weights.shape != expected_shape:
            raise ValueError(f"weights must have shape {expected_shape}, got {weights.shape}.")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weights contains NaN or Inf values; all weights must be finite.")
        return weights

    def _validate_input_vector(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        expected_shape = (self.n_features,)
        if x.shape != expected_shape:
            raise ValueError(f"x must have shape {expected_shape}, got {x.shape}.")
        if np.any(~np.isfinite(x)):
            raise ValueError("x contains NaN or Inf values; all features must be finite.")
        return x

    def _validate_input_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got X.ndim={X.ndim}.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X must have {self.n_features} features, got X.shape[1]={X.shape[1]}."
            )
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values; all features must be finite.")
        return X

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

        Raises
        ------
        ValueError
            If ``weights`` or ``x`` has the wrong shape or contains non-finite values.
        """
        weights = self._validate_weights(weights)
        x = self._validate_input_vector(x)
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

        Raises
        ------
        ValueError
            If ``weights`` or ``X`` has the wrong shape or contains non-finite values.
        """
        weights = self._validate_weights(weights)
        X = self._validate_input_batch(X)
        if self._circuit is None:
            self.build_circuit()
        sorted_params = list(self._circuit.parameters)
        n_samples, n_params = X.shape[0], len(sorted_params)
        param_values = np.empty((n_samples, n_params), dtype=float)

        param_to_col = {param: col for col, param in enumerate(sorted_params)}
        weight_cols = [param_to_col[param] for param in self.weight_params]
        input_items = [
            (feature_idx, param_to_col[param])
            for feature_idx, param in enumerate(self.input_params)
            if param in param_to_col
        ]

        param_values[:, weight_cols] = np.asarray(weights, dtype=float)
        if input_items:
            feature_indices, input_cols = zip(*input_items)
            param_values[:, input_cols] = X[:, feature_indices]
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

    def draw(self, **kwargs) -> Any:
        """Draw the circuit (delegates to :meth:`QuantumCircuit.draw`)."""
        return self.circuit.draw(**kwargs)

    def __repr__(self) -> str:
        return (
            f"DataReuploadingCircuit(n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, n_features={self.n_features}, "
            f"encoding='{self.encoding}', entanglement='{self.entanglement}')"
        )
