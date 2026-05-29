"""ReuploadingFeatureMap: drop-in feature map for QML pipelines."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

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

    Notes
    -----
    The feature map keeps the original data re-uploading angle structure,

        ``angle = fixed_w[l, i, r] + x[feat_idx(l, i, r)]``

    where ``fixed_w`` is sampled once at construction time and then frozen.
    Only the input parameters ``x`` remain symbolic in the returned circuit.
    The feature map requires enough encoding slots to upload every declared
    feature at least once.
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
        n_qubits = self._validate_positive_int("n_qubits", n_qubits)
        n_layers = self._validate_positive_int("n_layers", n_layers)
        n_features = self._validate_positive_int("n_features", n_features)
        if encoding not in self._VALID_ENCODINGS:
            raise ValueError(f"Invalid encoding '{encoding}'")
        if entanglement not in self._VALID_ENTANGLEMENTS:
            raise ValueError(f"Invalid entanglement '{entanglement}'")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.encoding = encoding
        self.entanglement = entanglement

        self._rotations: tuple[str, ...] = ENCODING_GATES[encoding]
        self._n_rots: int = len(self._rotations)
        self.n_weights: int = n_layers * n_qubits * self._n_rots
        if n_features > self.n_weights:
            raise ValueError(
                f"n_features={n_features} exceeds the number of encoding slots "
                f"({self.n_weights}). Increase n_layers, n_qubits, or encoding "
                "richness so every feature is uploaded at least once."
            )

        rng = np.random.default_rng(seed)
        self._fixed_weights: np.ndarray = rng.uniform(-np.pi, np.pi, self.n_weights)

        self.input_params: ParameterVector = ParameterVector("x", n_features)
        self._sorted_input_params_: list[Parameter] = []
        self._sorted_input_indices_: list[int] = []
        self._circuit: QuantumCircuit | None = None

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        """Validate that a constructor argument is a positive integer.

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

    def _validate_input_vector(self, x: np.ndarray) -> np.ndarray:
        """Validate a single input feature vector against the expected shape.

        Args:
            x: Candidate feature array to validate.

        Returns:
            The array cast to float64 with shape ``(n_features,)``.

        Raises:
            ValueError: If the shape is wrong or any value is non-finite.
        """
        x = np.asarray(x, dtype=float)
        expected_shape = (self.n_features,)
        if x.shape != expected_shape:
            raise ValueError(f"x must have shape {expected_shape}, got {x.shape}.")
        if np.any(~np.isfinite(x)):
            raise ValueError("x contains NaN or Inf values; all features must be finite.")
        return x

    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """Build and return the feature-map circuit with fixed weights.

        Returns
        -------
        QuantumCircuit
            Parameterized only by *x* (input parameters).

        Notes
        -----
        The circuit parameters are the same objects stored in
        :attr:`input_params`; this keeps external QML code from seeing a
        different ParameterVector than the one actually used by the circuit.
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

        # Bind weights to constants and remap base x parameters to this
        # object's ParameterVector so public input_params matches the circuit.
        binding = dict(zip(base.weight_params, self._fixed_weights))
        base_circuit_params = set(base.circuit.parameters)
        binding.update(
            {
                base_param: self.input_params[idx]
                for idx, base_param in enumerate(base.input_params)
                if base_param in base_circuit_params
            }
        )
        self._circuit = base.circuit.assign_parameters(binding)
        # After binding, self._circuit.parameters contains only the x params
        # Store them in sorted order for consistent binding in bind()
        self._sorted_input_params_ = list(self._circuit.parameters)
        param_to_feature = {param: idx for idx, param in enumerate(self.input_params)}
        self._sorted_input_indices_ = [
            param_to_feature[param] for param in self._sorted_input_params_
        ]
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

        Raises
        ------
        ValueError
            If ``x`` has the wrong shape or contains non-finite values.
        """
        x = self._validate_input_vector(x)
        if self._circuit is None:
            self.build_circuit()
        binding = dict(zip(self._sorted_input_params_, x[self._sorted_input_indices_]))
        return self._circuit.assign_parameters(binding)

    def draw(self, **kwargs) -> Any:
        """Draw the feature-map circuit."""
        return self.circuit.draw(**kwargs)
