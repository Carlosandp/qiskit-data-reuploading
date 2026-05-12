"""BenchmarkRunner: compare DataReuploadingClassifier against classical baselines."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass
class BenchmarkResult:
    """Container for a single model's benchmark outcome.

    Parameters
    ----------
    model_name : str
        Display name of the model.
    accuracy : float
        Holdout accuracy.
    f1 : float
        Holdout weighted F1-score.
    train_time_s : float
        Wall-clock fitting time in seconds.
    predict_time_s : float
        Wall-clock prediction time in seconds.
    cv_mean : float, optional
        Cross-validation accuracy mean. ``0.0`` when CV is skipped.
    cv_std : float, optional
        Cross-validation accuracy standard deviation. ``0.0`` when CV is skipped.
    extra : dict[str, Any], optional
        Additional model-specific metadata.
    """

    model_name: str
    accuracy: float
    f1: float
    train_time_s: float
    predict_time_s: float
    cv_mean: float = 0.0
    cv_std: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Compare a :class:`~qdr.models.DataReuploadingClassifier` against baselines.

    The runner performs one stratified train/test split and, when ``cv_folds > 1``,
    stratified cross-validation on the full dataset. Labels are encoded once with
    :class:`~sklearn.preprocessing.LabelEncoder` so all baselines, including
    XGBoost, receive the same target representation.

    Parameters
    ----------
    test_size : float, optional
        Fraction of data held out for evaluation. Default ``0.2``.
    cv_folds : int, optional
        Number of stratified cross-validation folds. Default ``5``. Set to ``0``
        or ``1`` to skip CV.
    random_state : int or None, optional
        Controls train/test split and classical model seeds.
    verbose : bool, optional
        Print progress updates. Default ``True``.

    Raises
    ------
    ValueError
        If constructor parameters are outside their valid ranges.

    Notes
    -----
    ``BenchmarkRunner`` does not perform angle scaling for the quantum model.
    Pass the feature representation intended for the circuit, usually scaled to
    a compact angular range such as ``[-pi, pi]``.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int | None = 42,
        verbose: bool = True,
    ) -> None:
        self.test_size = self._validate_test_size(test_size)
        self.cv_folds = self._validate_cv_folds(cv_folds)
        self.random_state = self._validate_random_state(random_state)
        self.verbose = self._validate_include_flag("verbose", verbose)
        self._results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_test_size(test_size: float) -> float:
        if isinstance(test_size, bool) or not isinstance(test_size, Real):
            raise ValueError(f"test_size must be a float in (0, 1), got {test_size!r}.")
        test_size = float(test_size)
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must satisfy 0 < test_size < 1, got {test_size}.")
        return test_size

    @staticmethod
    def _validate_cv_folds(cv_folds: int) -> int:
        if isinstance(cv_folds, bool) or not isinstance(cv_folds, Integral):
            raise ValueError(f"cv_folds must be a non-negative integer, got {cv_folds!r}.")
        cv_folds = int(cv_folds)
        if cv_folds < 0:
            raise ValueError(f"cv_folds must be >= 0, got {cv_folds}.")
        return cv_folds

    @staticmethod
    def _validate_random_state(random_state: int | None) -> int | None:
        if random_state is None:
            return None
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValueError(f"random_state must be an integer or None, got {random_state!r}.")
        return int(random_state)

    @staticmethod
    def _validate_include_flag(name: str, value: bool) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {value!r}.")
        return value

    @staticmethod
    def _validate_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got X.ndim={X.ndim}.")
        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values; all features must be finite.")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got y.ndim={y.ndim}.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: X has {X.shape[0]} samples, "
                f"y has {y.shape[0]}."
            )
        for label in y:
            if label is None or (
                isinstance(label, (float, np.floating)) and not np.isfinite(label)
            ):
                raise ValueError("y contains NaN, Inf, or None labels; class labels must be valid.")

        label_encoder = LabelEncoder()
        try:
            y_encoded = label_encoder.fit_transform(y)
        except TypeError as exc:
            raise ValueError("y labels must be mutually comparable by LabelEncoder.") from exc
        if len(label_encoder.classes_) < 2:
            raise ValueError(
                f"BenchmarkRunner requires at least 2 classes, got "
                f"n_classes={len(label_encoder.classes_)}."
            )
        return X, y_encoded, label_encoder.classes_

    def _validate_resampling(self, y_encoded: np.ndarray) -> None:
        classes, counts = np.unique(y_encoded, return_counts=True)
        n_classes = len(classes)
        min_count = int(counts.min())
        if min_count < 2:
            raise ValueError(
                "Each class needs at least 2 samples for stratified train/test split; "
                f"minimum class count is {min_count}."
            )

        n_samples = len(y_encoded)
        n_test = int(math.ceil(self.test_size * n_samples))
        n_train = n_samples - n_test
        if n_test < n_classes or n_train < n_classes:
            raise ValueError(
                "test_size leaves too few samples for stratified splitting: "
                f"n_train={n_train}, n_test={n_test}, n_classes={n_classes}."
            )

        if self.cv_folds > 1 and min_count < self.cv_folds:
            raise ValueError(
                f"cv_folds={self.cv_folds} exceeds the minimum class count "
                f"({min_count}). Use cv_folds <= {min_count} or disable CV."
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _cv_splitter(self) -> StratifiedKFold | None:
        if self.cv_folds <= 1:
            return None
        return StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

    @staticmethod
    def _default_qdr_model(n_features: int, n_classes: int, random_state: int | None) -> Any:
        from qdr.models.classifier import DataReuploadingClassifier

        n_qubits = max(2, n_classes)
        n_layers = max(3, math.ceil(n_features / (n_qubits * 3)))
        return DataReuploadingClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            optimizer="COBYLA",
            max_iter=50,
            seed=random_state,
        )

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
        cv: StratifiedKFold | None,
    ) -> BenchmarkResult:
        self._log(f"  -> Training {name} ...")

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(X_test)
        predict_time = time.perf_counter() - t0

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))

        cv_mean, cv_std = 0.0, 0.0
        if cv is not None:
            self._log(f"    running {self.cv_folds}-fold CV ...")
            cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="accuracy")
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

        result = BenchmarkResult(
            model_name=name,
            accuracy=acc,
            f1=f1,
            train_time_s=float(train_time),
            predict_time_s=float(predict_time),
            cv_mean=cv_mean,
            cv_std=cv_std,
        )
        self._log(
            f"    acc={acc:.3f}  f1={f1:.3f}  "
            f"train={train_time:.1f}s  "
            + (f"cv={cv_mean:.3f}+/-{cv_std:.3f}" if cv is not None else "")
        )
        return result

    # ------------------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        qdr_model: Any | None = None,
        include_logreg: bool = True,
        include_svm: bool = True,
        include_mlp: bool = True,
        include_rf: bool = True,
        include_xgboost: bool = False,
    ) -> "BenchmarkRunner":
        """Run the full benchmark suite.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Class labels, shape ``(n_samples,)``.
        qdr_model : DataReuploadingClassifier or None
            Configured but unfitted quantum model. If ``None``, a default model
            is created with enough qubits for the number of classes and enough
            layers to upload all features at least once.
        include_logreg : bool, optional
            Whether to benchmark scaled logistic regression. Default ``True``.
        include_svm : bool, optional
            Whether to benchmark a scaled RBF-SVM. Default ``True``.
        include_mlp : bool, optional
            Whether to benchmark a small scaled MLP. Default ``True``.
        include_rf : bool, optional
            Whether to benchmark a Random Forest. Default ``True``.
        include_xgboost : bool, optional
            Whether to benchmark XGBoost. Default ``False`` because it requires
            the optional ``xgboost`` package.

        Returns
        -------
        BenchmarkRunner
            Allows chaining: ``runner.run(...).summary()``.

        Raises
        ------
        ValueError
            If data, resampling settings, or include flags are invalid.
        ImportError
            If ``include_xgboost=True`` and ``xgboost`` is not installed.
        """
        include_logreg = self._validate_include_flag("include_logreg", include_logreg)
        include_svm = self._validate_include_flag("include_svm", include_svm)
        include_mlp = self._validate_include_flag("include_mlp", include_mlp)
        include_rf = self._validate_include_flag("include_rf", include_rf)
        include_xgboost = self._validate_include_flag("include_xgboost", include_xgboost)

        xgb_classifier_cls: Any | None = None
        if include_xgboost:
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "include_xgboost=True requires the optional xgboost package."
                ) from exc
            xgb_classifier_cls = XGBClassifier

        X, y_encoded, class_labels = self._validate_data(X, y)
        self._validate_resampling(y_encoded)
        self.classes_ = class_labels
        self.n_features_in_ = X.shape[1]
        cv = self._cv_splitter()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )
        self._log(f"Benchmark: {len(X_train)} train / {len(X_test)} test samples")

        results: list[BenchmarkResult] = []

        # --- Quantum model ---
        if qdr_model is None:
            qdr_model = self._default_qdr_model(
                n_features=X.shape[1],
                n_classes=len(class_labels),
                random_state=self.random_state,
            )
        qdr_result = self._benchmark_one(
            "DataReuploadingClassifier",
            qdr_model,
            X_train,
            X_test,
            y_train,
            y_test,
            X,
            y_encoded,
            cv,
        )
        if hasattr(qdr_model, "loss_history_"):
            qdr_result.extra["loss_history"] = qdr_model.loss_history_
        qdr_result.extra["classes"] = class_labels.tolist()
        results.append(qdr_result)

        # --- Logistic Regression ---
        if include_logreg:
            logreg = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            results.append(
                self._benchmark_one(
                    "Logistic Regression",
                    logreg,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X,
                    y_encoded,
                    cv,
                )
            )

        # --- SVM ---
        if include_svm:
            svm = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", random_state=self.random_state, probability=True)),
                ]
            )
            results.append(
                self._benchmark_one(
                    "SVM (RBF)",
                    svm,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X,
                    y_encoded,
                    cv,
                )
            )

        # --- Random Forest ---
        if include_rf:
            rf = RandomForestClassifier(
                n_estimators=300,
                random_state=self.random_state,
                # Keep benchmark execution portable in constrained CI/sandboxed
                # Windows environments where worker pools may be unavailable.
                n_jobs=1,
            )
            results.append(
                self._benchmark_one(
                    "Random Forest",
                    rf,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X,
                    y_encoded,
                    cv,
                )
            )

        # --- XGBoost ---
        if include_xgboost:
            xgb_kwargs: dict[str, Any] = {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": self.random_state,
                "n_jobs": 1,
                "verbosity": 0,
            }
            if len(class_labels) == 2:
                xgb_kwargs.update({"objective": "binary:logistic", "eval_metric": "logloss"})
            else:
                xgb_kwargs.update(
                    {
                        "objective": "multi:softprob",
                        "eval_metric": "mlogloss",
                        "num_class": len(class_labels),
                    }
                )
            xgb = xgb_classifier_cls(**xgb_kwargs)
            results.append(
                self._benchmark_one(
                    "XGBoost",
                    xgb,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X,
                    y_encoded,
                    cv,
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
            results.append(
                self._benchmark_one(
                    "MLP (32-16)",
                    mlp,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X,
                    y_encoded,
                    cv,
                )
            )

        self._results = results
        return self

    def summary(self) -> pd.DataFrame:
        """Return benchmark results as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            Indexed by model name with columns ``accuracy``, ``f1``,
            ``train_time_s``, ``predict_time_s``, ``cv_mean``, and ``cv_std``.
            Values are returned at full floating-point precision; callers can
            round only for presentation.

        Raises
        ------
        ValueError
            If called before :meth:`run` has produced results.
        """
        if not self._results:
            raise ValueError("No benchmark results available. Call run() before summary().")
        rows = [
            {
                "model": r.model_name,
                "accuracy": r.accuracy,
                "f1": r.f1,
                "train_time_s": r.train_time_s,
                "predict_time_s": r.predict_time_s,
                "cv_mean": r.cv_mean,
                "cv_std": r.cv_std,
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
