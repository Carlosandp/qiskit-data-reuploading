"""DataReuploadingRegressor: sklearn-compatible quantum regressor."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from qdr.circuits.data_reuploading import DataReuploadingCircuit
from qdr.training.optimizers import ADAM, COBYLA, SPSA


class DataReuploadingRegressor(BaseEstimator, RegressorMixin):
    """Quantum regressor based on the data re-uploading technique.

    Predicts a continuous scalar output as the expectation value of a Z
    observable, optionally scaled to the target range observed during training.

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
    backend : None
        Reserved for hardware integration.
    shots : int or None, optional
        ``None`` = exact statevector simulation.
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
        if self.shots is None:
            return StatevectorEstimator()
        try:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
            return AerEstimatorV2()
        except ImportError:
            return StatevectorEstimator()

    def _observable(self) -> SparsePauliOp:
        n = self.n_qubits
        return SparsePauliOp("I" * (n - 1) + "Z")

    def _evaluate_batch(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        param_values = self._circuit_.make_param_batch(weights, X)
        pub = (self._circuit_.circuit, self._obs_, param_values)
        job = self._estimator_.run([pub])
        return job.result()[0].data.evs  # shape (n_samples,) ∈ [-1,1]

    def _scale_to_target(self, evs: np.ndarray) -> np.ndarray:
        """Map [-1,1] → [y_min_, y_max_]."""
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
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]
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

        def loss_fn(w: np.ndarray) -> float:
            evs = self._evaluate_batch(w, X)
            return float(np.mean((evs - y_scaled) ** 2))

        self.loss_history_: list[float] = []

        def _record(w: np.ndarray) -> None:
            self.loss_history_.append(loss_fn(w))

        if self.optimizer == "COBYLA":
            result = COBYLA(maxiter=self.max_iter).minimize(loss_fn, init_weights, callback=_record)
        elif self.optimizer == "SPSA":
            result = SPSA(maxiter=self.max_iter, seed=self.seed).minimize(
                loss_fn, init_weights, callback=_record
            )
        elif self.optimizer == "ADAM":
            from qdr.training.gradients import ParameterShiftGradient

            psr = ParameterShiftGradient(self._circuit_, self._obs_, estimator=self._estimator_)
            grad_fn = lambda w: psr.compute(w, X, y_scaled)
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
        """
        check_is_fitted(self, "weights_")
        X = np.asarray(X, dtype=float)
        evs = self._evaluate_batch(self.weights_, X)
        return self._scale_to_target(evs)

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model."""
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
    def load(cls, path: str | Path) -> "DataReuploadingRegressor":
        """Load a saved model."""
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
