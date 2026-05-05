"""BenchmarkRunner: compare DataReuploadingClassifier against classical baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class BenchmarkResult:
    """Container for a single model's benchmark outcome."""

    model_name: str
    accuracy: float
    f1: float
    train_time_s: float
    predict_time_s: float
    cv_mean: float = 0.0
    cv_std: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Compare a :class:`~qdr.models.DataReuploadingClassifier` against MLP and SVM.

    Parameters
    ----------
    test_size : float, optional
        Fraction of data held out for evaluation.  Default 0.2.
    cv_folds : int, optional
        Number of cross-validation folds.  Default 5.  Set to 0 to skip CV.
    random_state : int or None, optional
        Controls train/test split and classical model seeds.
    verbose : bool, optional
        Print progress updates.  Default ``True``.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int | None = 42,
        verbose: bool = True,
    ) -> None:
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self._results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _benchmark_one(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_full: np.ndarray,
        y_full: np.ndarray,
    ) -> BenchmarkResult:
        self._log(f"  → Training {name} …")

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(X_test)
        predict_time = time.perf_counter() - t0

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        cv_mean, cv_std = 0.0, 0.0
        if self.cv_folds > 1:
            self._log(f"    running {self.cv_folds}-fold CV …")
            cv_scores = cross_val_score(model, X_full, y_full, cv=self.cv_folds, scoring="accuracy")
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

        result = BenchmarkResult(
            model_name=name,
            accuracy=acc,
            f1=f1,
            train_time_s=train_time,
            predict_time_s=predict_time,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )
        self._log(
            f"    acc={acc:.3f}  f1={f1:.3f}  "
            f"train={train_time:.1f}s  "
            + (f"cv={cv_mean:.3f}±{cv_std:.3f}" if self.cv_folds > 1 else "")
        )
        return result

    # ------------------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        qdr_model: Any | None = None,
        include_svm: bool = True,
        include_mlp: bool = True,
    ) -> "BenchmarkRunner":
        """Run the full benchmark suite.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Labels.
        qdr_model : DataReuploadingClassifier or None
            A configured (but not yet fitted) quantum model.  If ``None`` a
            default 2-qubit, 3-layer COBYLA classifier is used.
        include_svm : bool
            Whether to benchmark a scaled RBF-SVM.  Default ``True``.
        include_mlp : bool
            Whether to benchmark a small MLP.  Default ``True``.

        Returns
        -------
        self
            Allows chaining: ``runner.run(...).summary()``.
        """
        from qdr.models.classifier import DataReuploadingClassifier

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self._log(f"Benchmark — {len(X_train)} train / {len(X_test)} test samples")

        self._results = []

        # --- Quantum model ---
        if qdr_model is None:
            qdr_model = DataReuploadingClassifier(
                n_qubits=2,
                n_layers=3,
                optimizer="COBYLA",
                max_iter=50,
                seed=self.random_state,
            )
        self._results.append(
            self._benchmark_one(
                "DataReuploadingClassifier",
                qdr_model,
                X_train, X_test, y_train, y_test, X, y,
            )
        )
        # Attach loss history as extra info
        if hasattr(qdr_model, "loss_history_"):
            self._results[-1].extra["loss_history"] = qdr_model.loss_history_

        # --- SVM ---
        if include_svm:
            svm = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", random_state=self.random_state, probability=True)),
                ]
            )
            self._results.append(
                self._benchmark_one(
                    "SVM (RBF)",
                    svm,
                    X_train, X_test, y_train, y_test, X, y,
                )
            )

        # --- MLP ---
        if include_mlp:
            mlp = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(32, 16),
                            max_iter=300,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            self._results.append(
                self._benchmark_one(
                    "MLP (32-16)",
                    mlp,
                    X_train, X_test, y_train, y_test, X, y,
                )
            )

        return self

    def summary(self) -> pd.DataFrame:
        """Return benchmark results as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            Columns: model_name, accuracy, f1, train_time_s, predict_time_s,
            cv_mean, cv_std.
        """
        rows = [
            {
                "model": r.model_name,
                "accuracy": round(r.accuracy, 4),
                "f1": round(r.f1, 4),
                "train_time_s": round(r.train_time_s, 2),
                "predict_time_s": round(r.predict_time_s, 4),
                "cv_mean": round(r.cv_mean, 4),
                "cv_std": round(r.cv_std, 4),
            }
            for r in self._results
        ]
        df = pd.DataFrame(rows).set_index("model")
        if self.verbose:
            print("\n" + df.to_string())
        return df

    @property
    def results(self) -> list[BenchmarkResult]:
        """Raw list of :class:`BenchmarkResult` objects."""
        return self._results
