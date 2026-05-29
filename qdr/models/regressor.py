"""DataReuploadingRegressor: sklearn-compatible quantum regressor."""

from __future__ import annotations

import pickle
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from qdr.circuits.data_reuploading import DataReuploadingCircuit
from qdr.training.optimizers import ADAM, COBYLA, SPSA
from qdr.utils.encoding import N_ROTATIONS


class DataReuploadingRegressor(BaseEstimator, RegressorMixin):
    """Quantum regressor based on the data re-uploading technique.

    Predicts a continuous scalar output as the expectation value of a Z
    observable, scaled to the target range observed during training.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits.  Default 2.
    n_layers : int, optional
        Number of data-reuploading layers.  Default 5.
    encoding : str, optional
        Rotation encoding.  Default ``"rx_ry_rz"``.
    entanglement : str, optional
        Entanglement pattern.  Default ``"full"``.
    optimizer : str, optional
        ``"SPSA"``, ``"COBYLA"`` (default), or ``"ADAM"``.
    backend : None, optional
        Must be ``None`` during ``fit``. Hardware execution is exposed through
        :func:`qdr.hardware.run_on_ibm_backend` so that real-backend cost,
        authentication, transpilation, and batching are explicit.
    shots : int or None, optional
        ``None`` = exact statevector simulation.  Positive int uses
        :class:`~qiskit_aer.primitives.EstimatorV2` with that shot count.
    max_iter : int, optional
        Maximum optimiser iterations.  Default 100.
    learning_rate : float, optional
        Learning rate for ADAM.  Default 0.01.
    seed : int or None, optional
        Random seed.

    Attributes
    ----------
    weights_ : np.ndarray
        Trained parameters.
    loss_history_ : list[float]
        Per-iteration loss values.
    n_features_in_ : int
        Number of features seen during fit.
    y_min_ : float
        Minimum target value seen during training.
    y_max_ : float
        Maximum target value seen during training.

    Notes
    -----
    The underlying circuit uses gates whose angles are

        ``angle = w[l, i, r] + x[feat_idx(l, i, r)]``

    and predicts one scalar from a single ``<Z_0>`` observable. Outputs are
    rescaled from ``[-1, 1]`` to the target range observed during ``fit``. This
    estimator is intentionally single-output; multi-output regression requires
    a separate observable design and is not implemented.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 5,
        encoding: str = "rx_ry_rz",
        entanglement: str = "full",
        optimizer: str = "COBYLA",
        backend: Any = None,
        shots: int | None = None,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.entanglement = entanglement
        self.optimizer = optimizer
        self.backend = backend
        self.shots = shots
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.seed = seed

    # ------------------------------------------------------------------

    def _build_estimator(self):
        """Instantiate the Qiskit V2 estimator appropriate for the current settings.

        Returns:
            A ``StatevectorEstimator`` when ``shots`` is ``None``, or an
            ``AerEstimatorV2`` configured with the requested shot count.

        Raises:
            ValueError: If ``backend`` is not ``None`` or ``shots`` is invalid.
            ImportError: If ``shots`` is set but ``qiskit-aer`` is not installed.
        """
        if self.backend is not None:
            raise ValueError(
                "backend is not supported by fit(); got "
                f"backend={self.backend!r}. Use qdr.hardware.run_on_ibm_backend() "
                "for IBM Quantum execution."
            )
        if self.shots is None:
            return StatevectorEstimator()
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots < 1:
            raise ValueError(f"shots must be None or a positive integer, got {self.shots!r}.")
        try:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
        except ImportError:
            raise ImportError(
                "Finite-shot simulation requires qiskit-aer. "
                "Install it with: pip install qiskit-data-reuploading"
            ) from None
        run_options: dict[str, int] = {"shots": self.shots}
        if self.seed is not None:
            run_options["seed_simulator"] = self.seed
        return AerEstimatorV2(options={"run_options": run_options})

    def _validate_X(self, X: np.ndarray, *, reset: bool) -> np.ndarray:
        """Validate the feature matrix and optionally record feature count.

        Args:
            X: Input feature matrix to validate.
            reset: When ``True``, accept any feature count (used during
                ``fit``). When ``False``, verify the count matches the stored
                ``n_features_in_`` (used during ``predict``).

        Returns:
            X cast to float64.

        Raises:
            ValueError: If ``X`` is not 2D, contains non-finite values, or the
                feature count mismatches when ``reset=False``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got X.ndim={X.ndim}.")
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values; all features must be finite.")
        if not reset and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this regressor was fitted with "
                f"n_features_in_={self.n_features_in_}."
            )
        return X

    def _validate_y(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        """Validate the target value vector.

        Args:
            y: Target array to validate.
            n_samples: Expected number of samples (must match ``X.shape[0]``).

        Returns:
            y cast to float64.

        Raises:
            ValueError: If ``y`` is not 1D, has a length mismatch, or contains
                non-finite values.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got y.ndim={y.ndim}.")
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y have inconsistent lengths: X has {n_samples} samples, "
                f"y has {y.shape[0]}."
            )
        if np.any(~np.isfinite(y)):
            raise ValueError("y contains NaN or Inf values; all targets must be finite.")
        return y

    def _validate_feature_capacity(self, n_features: int) -> None:
        """Raise if the feature count exceeds the available encoding slots.

        Args:
            n_features: Number of input features from the training data.

        Raises:
            ValueError: If ``n_features`` exceeds the number of rotation-gate
                slots implied by the current ``n_qubits``, ``n_layers``, and
                ``encoding`` settings.
        """
        if self.encoding not in N_ROTATIONS:
            return
        if (
            isinstance(self.n_layers, bool)
            or isinstance(self.n_qubits, bool)
            or not isinstance(self.n_layers, Integral)
            or not isinstance(self.n_qubits, Integral)
            or self.n_layers < 1
            or self.n_qubits < 1
        ):
            return
        n_slots = int(self.n_layers) * int(self.n_qubits) * N_ROTATIONS[self.encoding]
        if n_features > n_slots:
            raise ValueError(
                f"n_features_in_={n_features} exceeds the number of encoding slots "
                f"({n_slots}) for n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
                f"encoding='{self.encoding}'. Increase model capacity or reduce features."
            )

    def _observable(self) -> SparsePauliOp:
        """Build the Z observable on qubit 0 for the current circuit size.

        Returns:
            A ``SparsePauliOp`` representing ``<Z_0>`` using Qiskit's
            little-endian convention (rightmost character = qubit 0).
        """
        n = self.n_qubits
        return SparsePauliOp("I" * (n - 1) + "Z")

    def _evaluate_batch(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate the Z-observable expectation for a batch of samples.

        Args:
            weights: Current parameter vector, shape ``(n_weights,)``.
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            Expectation values in ``[-1, 1]``, shape ``(n_samples,)``.
        """
        param_values = self._circuit_.make_param_batch(weights, X)
        pub = (self._circuit_.circuit, self._obs_, param_values)
        job = self._estimator_.run([pub])
        return job.result()[0].data.evs  # shape (n_samples,) ∈ [-1,1]

    def _scale_to_target(self, evs: np.ndarray) -> np.ndarray:
        """Map [-1,1] → [y_min_, y_max_]."""
        evs = np.clip(evs, -1.0, 1.0)
        return self.y_min_ + (evs + 1.0) / 2.0 * (self.y_max_ - self.y_min_)

    def _scale_from_target(self, y: np.ndarray) -> np.ndarray:
        """Map [y_min_, y_max_] → [-1,1]."""
        denom = self.y_max_ - self.y_min_
        if denom == 0:
            return np.zeros_like(y)
        return 2.0 * (y - self.y_min_) / denom - 1.0

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataReuploadingRegressor":
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional, contains non-finite values, ``y``
            has the wrong shape or length, targets contain non-finite values,
            the feature count exceeds the available data-uploading slots, or
            the optimizer or circuit configuration is invalid.
        ImportError
            If ``shots`` is a positive integer and ``qiskit-aer`` is not
            installed.
        """
        X = self._validate_X(X, reset=True)
        y = self._validate_y(y, X.shape[0])

        self.n_features_in_ = X.shape[1]
        self._validate_feature_capacity(self.n_features_in_)
        self.y_min_ = float(y.min())
        self.y_max_ = float(y.max())
        y_scaled = self._scale_from_target(y)  # ∈ [-1, 1]

        self._circuit_ = DataReuploadingCircuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_features=self.n_features_in_,
            encoding=self.encoding,
            entanglement=self.entanglement,
        )
        self._circuit_.build_circuit()
        self._estimator_ = self._build_estimator()
        self._obs_ = self._observable()

        rng = np.random.default_rng(self.seed)
        init_weights = rng.uniform(-np.pi, np.pi, self._circuit_.n_weights)

        self._last_loss_: float = 0.0

        def loss_fn(w: np.ndarray) -> float:
            evs = self._evaluate_batch(w, X)
            val = float(np.mean((evs - y_scaled) ** 2))
            self._last_loss_ = val
            return val

        self.loss_history_: list[float] = []

        def _record(w: np.ndarray) -> None:
            self.loss_history_.append(self._last_loss_)

        if self.optimizer == "COBYLA":
            result = COBYLA(maxiter=self.max_iter).minimize(loss_fn, init_weights, callback=_record)
        elif self.optimizer == "SPSA":
            result = SPSA(maxiter=self.max_iter, seed=self.seed).minimize(
                loss_fn, init_weights, callback=_record
            )
        elif self.optimizer == "ADAM":
            from qdr.training.gradients import ParameterShiftGradient

            psr = ParameterShiftGradient(self._circuit_, self._obs_, estimator=self._estimator_)

            def grad_fn(w: np.ndarray) -> np.ndarray:
                return psr.compute(w, X, y_scaled)

            result = ADAM(maxiter=self.max_iter, lr=self.learning_rate).minimize(
                loss_fn, init_weights, gradient_fn=grad_fn, callback=_record
            )
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")

        self.weights_ = result.x
        if result.loss_history:
            self.loss_history_ = result.loss_history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)

        Raises
        ------
        ValueError
            If ``X`` contains non-finite values or its feature count differs
            from the data used in ``fit``.
        sklearn.exceptions.NotFittedError
            If the regressor has not been fitted.
        """
        check_is_fitted(self, "weights_")
        X = self._validate_X(X, reset=False)
        evs = self._evaluate_batch(self.weights_, X)
        return self._scale_to_target(evs)

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the regressor has not been fitted.
        """
        check_is_fitted(self, "weights_")
        payload = {
            "params": self.get_params(),
            "weights_": self.weights_,
            "n_features_in_": self.n_features_in_,
            "y_min_": self.y_min_,
            "y_max_": self.y_max_,
            "loss_history_": self.loss_history_,
        }
        Path(path).write_bytes(pickle.dumps(payload))

    @classmethod
    def load(
        cls: type["DataReuploadingRegressor"],
        path: str | Path,
    ) -> "DataReuploadingRegressor":
        """Load a saved model.

        Parameters
        ----------
        path : str or Path
            Path to a file created by :meth:`save`.

        Returns
        -------
        DataReuploadingRegressor

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the saved parameters are inconsistent with the circuit
            constraints.
        """
        payload = pickle.loads(Path(path).read_bytes())
        model = cls(**payload["params"])
        model.weights_ = payload["weights_"]
        model.n_features_in_ = payload["n_features_in_"]
        model.y_min_ = payload["y_min_"]
        model.y_max_ = payload["y_max_"]
        model.loss_history_ = payload["loss_history_"]
        model._circuit_ = DataReuploadingCircuit(
            n_qubits=model.n_qubits,
            n_layers=model.n_layers,
            n_features=model.n_features_in_,
            encoding=model.encoding,
            entanglement=model.entanglement,
        )
        model._circuit_.build_circuit()
        model._estimator_ = model._build_estimator()
        model._obs_ = model._observable()
        return model
