"""DataReuploadingClassifier: sklearn-compatible quantum classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from qdr.circuits.data_reuploading import DataReuploadingCircuit
from qdr.training.optimizers import ADAM, COBYLA, SPSA, get_optimizer


class DataReuploadingClassifier(BaseEstimator, ClassifierMixin):
    """Quantum classifier based on the data re-uploading technique.

    Implements a full sklearn-compatible estimator (``fit``, ``predict``,
    ``predict_proba``, ``score``, ``save`` / ``load``) backed by Qiskit 2.x
    V2 primitives.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits.  Default 2.
    n_layers : int, optional
        Number of data-reuploading layers.  Default 5.
    encoding : str, optional
        Rotation encoding — ``"rx"``, ``"ry"``, ``"rz"``, or ``"rx_ry_rz"``
        (default).
    entanglement : str, optional
        Entanglement pattern — ``"none"``, ``"linear"``, ``"circular"``,
        ``"full"`` (default).
    optimizer : str, optional
        Optimizer — ``"SPSA"``, ``"COBYLA"`` (default), or ``"ADAM"``.
    backend : None
        Reserved for future hardware integration.  Pass ``None`` to use the
        local :class:`~qiskit.primitives.StatevectorEstimator`.
    shots : int or None, optional
        ``None`` = exact statevector simulation.  Positive int uses the
        :class:`~qiskit_aer.primitives.SamplerV2` instead (noise aware).
    max_iter : int, optional
        Maximum optimiser iterations.  Default 100.
    learning_rate : float, optional
        Learning rate (ADAM only).  Default 0.01.
    seed : int or None, optional
        Random seed for weight initialisation and stochastic optimisers.

    Attributes
    ----------
    weights_ : np.ndarray
        Trained parameters, shape ``(n_weights,)``.
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    loss_history_ : list[float]
        Per-iteration loss values recorded during training.
    n_features_in_ : int
        Number of features seen during ``fit``.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_estimator(self):
        if self.shots is None:
            return StatevectorEstimator()
        # Noisy simulation via Aer sampler-backed estimator
        try:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
            return AerEstimatorV2()
        except ImportError:
            return StatevectorEstimator()

    def _make_observable(self, n_classes: int) -> list[SparsePauliOp]:
        """Return a list of Z observables (one per output class/qubit)."""
        n = self.n_qubits
        if n_classes == 2:
            # Measure qubit 0 in Z basis; Qiskit little-endian: rightmost char = qubit 0
            return [SparsePauliOp("I" * (n - 1) + "Z")]
        # Multiclass: one observable per class, cycling through qubits
        obs = []
        for k in range(n_classes):
            q = k % n
            s = list("I" * n)
            s[n - 1 - q] = "Z"
            obs.append(SparsePauliOp("".join(s)))
        return obs

    def _evaluate_batch(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        observables: list[SparsePauliOp],
    ) -> np.ndarray:
        """Return expectation values, shape ``(n_samples,)`` or ``(n_samples, n_obs)``."""
        param_values = self._circuit_.make_param_batch(weights, X)  # (n_samples, n_params)
        pubs = [(self._circuit_.circuit, obs, param_values) for obs in observables]
        job = self._estimator_.run(pubs)
        result = job.result()
        # Each pub result has evs of shape (n_samples,)
        evs = np.column_stack([result[i].data.evs for i in range(len(observables))])
        if evs.shape[1] == 1:
            return evs[:, 0]  # (n_samples,) for binary
        return evs  # (n_samples, n_classes) for multiclass

    def _loss(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y_mapped: np.ndarray,
        observables: list[SparsePauliOp],
    ) -> float:
        evs = self._evaluate_batch(weights, X, observables)
        return float(np.mean((evs - y_mapped) ** 2))

    def _make_grad_fn(
        self,
        X: np.ndarray,
        y_mapped: np.ndarray,
        observables: list[SparsePauliOp],
    ):
        from qdr.training.gradients import ParameterShiftGradient

        # For multiclass we sum gradients across all observables
        grads = [
            ParameterShiftGradient(self._circuit_, obs, estimator=self._estimator_)
            for obs in observables
        ]

        def gradient_fn(weights: np.ndarray) -> np.ndarray:
            if len(observables) == 1:
                evs = self._evaluate_batch(weights, X, observables)
                return grads[0].compute(weights, X, y_mapped)
            # Multiclass: average gradient across observables
            total_grad = np.zeros_like(weights)
            for k, g in enumerate(grads):
                total_grad += g.compute(weights, X, y_mapped[:, k])
            return total_grad / len(grads)

        return gradient_fn

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataReuploadingClassifier":
        """Fit the classifier to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._label_encoder_ = LabelEncoder()
        y_int = self._label_encoder_.fit_transform(y)
        self.classes_ = self._label_encoder_.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        # Map to {-1, +1} for binary, or one-hot-like for multiclass
        if n_classes == 2:
            y_mapped: np.ndarray = 2.0 * y_int - 1.0  # {0,1} → {-1,+1}
        else:
            # One-hot mapped to {-1, +1}
            y_one_hot = np.full((len(y_int), n_classes), -1.0)
            for i, label in enumerate(y_int):
                y_one_hot[i, label] = 1.0
            y_mapped = y_one_hot

        # Build circuit and estimator
        self._circuit_ = DataReuploadingCircuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_features=self.n_features_in_,
            encoding=self.encoding,
            entanglement=self.entanglement,
        )
        self._circuit_.build_circuit()
        self._estimator_ = self._build_estimator()
        self._observables_ = self._make_observable(n_classes)

        # Initialise weights
        rng = np.random.default_rng(self.seed)
        init_weights = rng.uniform(-np.pi, np.pi, self._circuit_.n_weights)

        def loss_fn(w: np.ndarray) -> float:
            return self._loss(w, X, y_mapped, self._observables_)

        # Run optimisation
        self.loss_history_: list[float] = []

        def _record(w: np.ndarray) -> None:
            self.loss_history_.append(loss_fn(w))

        if self.optimizer == "COBYLA":
            opt = COBYLA(maxiter=self.max_iter)
            result = opt.minimize(loss_fn, init_weights, callback=_record)

        elif self.optimizer == "SPSA":
            opt = SPSA(maxiter=self.max_iter, seed=self.seed)
            result = opt.minimize(loss_fn, init_weights, callback=_record)

        elif self.optimizer == "ADAM":
            grad_fn = self._make_grad_fn(X, y_mapped, self._observables_)
            opt = ADAM(maxiter=self.max_iter, lr=self.learning_rate)
            result = opt.minimize(loss_fn, init_weights, gradient_fn=grad_fn, callback=_record)

        else:
            raise ValueError(
                f"optimizer must be one of ['COBYLA', 'SPSA', 'ADAM'], got '{self.optimizer}'"
            )

        self.weights_ = result.x
        if result.loss_history:
            self.loss_history_ = result.loss_history

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, "weights_")
        X = np.asarray(X, dtype=float)
        evs = self._evaluate_batch(self.weights_, X, self._observables_)

        if evs.ndim == 1:
            # Binary: map [-1,1] → [0,1]
            p1 = (evs + 1.0) / 2.0
            return np.column_stack([1.0 - p1, p1])

        # Multiclass: softmax of expectations
        def softmax(z: np.ndarray) -> np.ndarray:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        return softmax(evs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to a file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"model.pkl"``).
        """
        check_is_fitted(self, "weights_")
        payload = {
            "params": self.get_params(),
            "weights_": self.weights_,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
            "loss_history_": self.loss_history_,
        }
        Path(path).write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "DataReuploadingClassifier":
        """Load a previously saved model.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pkl`` file created by :meth:`save`.

        Returns
        -------
        DataReuploadingClassifier
        """
        payload = pickle.loads(Path(path).read_bytes())
        model = cls(**payload["params"])
        # Reconstruct internal state without re-fitting
        model.weights_ = payload["weights_"]
        model.classes_ = payload["classes_"]
        model.n_features_in_ = payload["n_features_in_"]
        model.loss_history_ = payload["loss_history_"]
        model._label_encoder_ = LabelEncoder()
        model._label_encoder_.classes_ = model.classes_

        n_classes = len(model.classes_)
        model._circuit_ = DataReuploadingCircuit(
            n_qubits=model.n_qubits,
            n_layers=model.n_layers,
            n_features=model.n_features_in_,
            encoding=model.encoding,
            entanglement=model.entanglement,
        )
        model._circuit_.build_circuit()
        model._estimator_ = model._build_estimator()
        model._observables_ = model._make_observable(n_classes)
        return model

    def __repr__(self) -> str:
        return (
            f"DataReuploadingClassifier(n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, encoding='{self.encoding}', "
            f"entanglement='{self.entanglement}', optimizer='{self.optimizer}')"
        )
