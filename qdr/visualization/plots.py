"""Visualization utilities for trained QDR models and benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _validate_finite_2d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {name}.ndim={array.ndim}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf values; all values must be finite.")
    return array


def _validate_resolution(resolution: int) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, Integral):
        raise ValueError(f"resolution must be an integer >= 2, got {resolution!r}.")
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}.")
    return resolution


def _validate_feature_indices(
    feature_indices: tuple[int, int],
    n_features: int,
) -> tuple[int, int]:
    try:
        n_indices = len(feature_indices)
    except TypeError:
        raise ValueError("feature_indices must contain exactly two feature indices.") from None
    if n_indices != 2:
        raise ValueError("feature_indices must contain exactly two feature indices.")
    i, j = feature_indices
    if (
        isinstance(i, bool)
        or isinstance(j, bool)
        or not isinstance(i, Integral)
        or not isinstance(j, Integral)
    ):
        raise ValueError(f"feature_indices must be integers, got {feature_indices!r}.")
    i, j = int(i), int(j)
    if i == j:
        raise ValueError(f"feature_indices must be distinct, got {feature_indices!r}.")
    if i < 0 or j < 0 or i >= n_features or j >= n_features:
        raise ValueError(
            f"feature_indices={feature_indices!r} is invalid for n_features={n_features}."
        )
    return i, j


def _ordered_unique(*arrays: np.ndarray) -> list[Any]:
    classes: list[Any] = []
    for array in arrays:
        for value in np.asarray(array, dtype=object).ravel():
            if not any(value == existing for existing in classes):
                classes.append(value)
    return classes


def _label_indices(values: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    flat_values = np.asarray(values, dtype=object).ravel()
    encoded = np.empty(flat_values.shape[0], dtype=int)
    for row, value in enumerate(flat_values):
        for class_idx, class_value in enumerate(classes):
            if value == class_value:
                encoded[row] = class_idx
                break
        else:  # pragma: no cover - defensive; classes are built from values.
            raise ValueError(f"Predicted label {value!r} is not present in the class palette.")
    return encoded.reshape(np.shape(values))


def _class_cmap(plt: Any, n_classes: int) -> Any:
    cmap_name = "tab10" if n_classes <= 10 else "tab20" if n_classes <= 20 else "hsv"
    colors = [plt.get_cmap(cmap_name, n_classes)(idx) for idx in range(n_classes)]
    from matplotlib.colors import ListedColormap

    return ListedColormap(colors)


def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 50,
    feature_indices: tuple[int, int] = (0, 1),
    ax: "Axes | None" = None,
    title: str = "Decision Boundary",
) -> "Figure":
    """Plot a 2D decision boundary for a fitted classifier.

    Parameters
    ----------
    model : fitted classifier
        Any object with a ``predict(X)`` method.
    X : np.ndarray
        Feature matrix with shape ``(n_samples, n_features)``.
    y : np.ndarray
        True labels with shape ``(n_samples,)``.
    resolution : int, optional
        Grid resolution, in points per axis. Default ``50``.
    feature_indices : tuple[int, int], optional
        Two feature indices to plot. Default ``(0, 1)``.
    ax : Axes or None, optional
        Existing axes to plot into. If ``None``, a new figure is created.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        Matplotlib figure containing the decision boundary.

    Raises
    ------
    ValueError
        If inputs have invalid shape, contain non-finite values, or if model
        predictions cannot be aligned with the plotted grid.

    Notes
    -----
    For ``n_features > 2``, the non-plotted features are fixed to their
    empirical mean in ``X``. This draws a well-defined 2D slice of the fitted
    model instead of silently moving selected features into the wrong columns.
    """
    import matplotlib.pyplot as plt

    X = _validate_finite_2d("X", X)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got y.ndim={y.ndim}.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y must have length {X.shape[0]}, got {y.shape[0]}.")
    resolution = _validate_resolution(resolution)
    i, j = _validate_feature_indices(feature_indices, X.shape[1])

    X2 = X[:, [i, j]]
    x_min, x_max = X2[:, 0].min() - 0.3, X2[:, 0].max() + 0.3
    y_min, y_max = X2[:, 1].min() - 0.3, X2[:, 1].max() + 0.3

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.tile(X.mean(axis=0), (xx.size, 1))
    grid[:, i] = xx.ravel()
    grid[:, j] = yy.ravel()

    Z = np.asarray(model.predict(grid))
    if Z.shape != (xx.size,):
        raise ValueError(
            f"model.predict must return shape ({xx.size},), got {Z.shape}."
        )
    Z = Z.reshape(xx.shape)

    classes = _ordered_unique(getattr(model, "classes_", np.array([], dtype=object)), y, Z)
    if len(classes) < 1:
        raise ValueError("At least one class is required to plot a decision boundary.")
    cmap = _class_cmap(plt, len(classes))
    Z_int = _label_indices(Z, classes)
    y_int = _label_indices(y, classes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    levels = np.arange(len(classes) + 1) - 0.5
    ax.contourf(xx, yy, Z_int, levels=levels, alpha=0.35, cmap=cmap)
    scatter = ax.scatter(
        X2[:, 0],
        X2[:, 1],
        c=y_int,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(classes) - 0.5,
        edgecolors="k",
        s=40,
    )
    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(classes)))
    cbar.ax.set_yticklabels([str(label) for label in classes])
    cbar.set_label("Class")
    fig.tight_layout()
    return fig


def plot_loss_curve(
    loss_history: Sequence[float],
    ax: "Axes | None" = None,
    title: str = "Training Loss",
    label: str = "loss",
) -> "Figure":
    """Plot an optimization loss curve.

    Parameters
    ----------
    loss_history : Sequence[float]
        Per-iteration loss values.
    ax : Axes or None, optional
        Existing axes to plot into. If ``None``, a new figure is created.
    title : str, optional
        Plot title.
    label : str, optional
        Legend label for the curve.

    Returns
    -------
    Figure
        Matplotlib figure containing the loss curve.

    Raises
    ------
    ValueError
        If ``loss_history`` is empty, not one-dimensional, or contains
        non-finite values.
    """
    import matplotlib.pyplot as plt

    losses = np.asarray(loss_history, dtype=float)
    if losses.ndim != 1:
        raise ValueError(f"loss_history must be a 1D sequence, got ndim={losses.ndim}.")
    if losses.size == 0:
        raise ValueError("loss_history must contain at least one value.")
    if np.any(~np.isfinite(losses)):
        raise ValueError("loss_history contains NaN or Inf values.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(np.arange(losses.size), losses, label=label, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_benchmark_comparison(
    summary_df: Any,
    metric: str = "accuracy",
    ax: "Axes | None" = None,
    title: str | None = None,
) -> "Figure":
    """Plot a bar chart comparing benchmark metrics across models.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of :meth:`qdr.benchmarks.BenchmarkRunner.summary`, indexed by
        model name.
    metric : str, optional
        Numeric column to plot. Default ``"accuracy"``.
    ax : Axes or None, optional
        Existing axes to plot into. If ``None``, a new figure is created.
    title : str or None, optional
        Plot title. If ``None``, a default title is used.

    Returns
    -------
    Figure
        Matplotlib figure containing the benchmark comparison.

    Raises
    ------
    ValueError
        If the summary is empty, the metric is missing, or metric values are
        non-numeric/non-finite.
    """
    import matplotlib.pyplot as plt

    if getattr(summary_df, "empty", False):
        raise ValueError("summary_df must contain at least one benchmark result.")
    if metric not in getattr(summary_df, "columns", []):
        raise ValueError(f"metric '{metric}' is not present in summary_df.")

    models = [str(model) for model in summary_df.index.tolist()]
    try:
        values = np.asarray(summary_df[metric].to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"metric '{metric}' must contain numeric values.") from None
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"metric '{metric}' must contain at least one value.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"metric '{metric}' contains NaN or Inf values.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    palette = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#937860",
        "#64B5CD",
    ]
    colors = [palette[i % len(palette)] for i in range(len(models))]

    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    if values.min() >= 0 and values.max() <= 1:
        ax.set_ylim(0, 1.05)
    else:
        margin = max(0.05 * (values.max() - values.min()), 1e-12)
        if values.min() >= 0:
            ax.set_ylim(0, values.max() + margin)
        elif values.max() <= 0:
            ax.set_ylim(values.min() - margin, 0)
        else:
            ax.set_ylim(values.min() - margin, values.max() + margin)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Model comparison - {metric}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bloch_sphere(
    statevector: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "Bloch Sphere",
) -> "Figure":
    """Render a normalized single-qubit state on a Bloch sphere.

    Parameters
    ----------
    statevector : np.ndarray
        Complex array with shape ``(2,)`` representing
        ``alpha |0> + beta |1>``.
    ax : Axes or None, optional
        Existing 3D axes created with ``projection="3d"``. If ``None``, a new
        3D axes is created.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        Matplotlib figure containing the Bloch sphere.

    Raises
    ------
    ValueError
        If ``statevector`` is not a finite normalized single-qubit state, or if
        ``ax`` is provided but is not a 3D axes.

    Notes
    -----
    The plotted vector is ``(<X>, <Y>, <Z>)`` for the supplied pure state. The
    state must be normalized because otherwise these expectation values are not
    physically valid Bloch coordinates.
    """
    import matplotlib.pyplot as plt

    state = np.asarray(statevector, dtype=complex)
    if state.shape != (2,):
        raise ValueError(f"statevector must have shape (2,), got {state.shape}.")
    if np.any(~np.isfinite(state.real)) or np.any(~np.isfinite(state.imag)):
        raise ValueError("statevector contains NaN or Inf values.")
    norm = float(np.linalg.norm(state))
    if not np.isclose(norm, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError(f"statevector must be normalized to norm 1, got norm={norm}.")

    alpha, beta = state[0], state[1]
    bx = 2.0 * np.real(np.conj(alpha) * beta)
    by = 2.0 * np.imag(np.conj(alpha) * beta)
    bz = float(np.abs(alpha) ** 2 - np.abs(beta) ** 2)

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        if not hasattr(ax, "get_zlim"):
            raise ValueError("ax must be a 3D axes created with projection='3d'.")
        fig = ax.figure

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgrey", alpha=0.3, linewidth=0.4)

    for start, end, label_axis in [
        ([0, 0, -1.3], [0, 0, 1.4], "Z"),
        ([-1.3, 0, 0], [1.4, 0, 0], "X"),
        ([0, -1.3, 0], [0, 1.4, 0], "Y"),
    ]:
        ax.plot(*zip(start, end), color="grey", linewidth=0.8)
        ax.text(*end, label_axis, fontsize=9, ha="center")

    ax.quiver(0, 0, 0, bx, by, bz, color="royalblue", linewidth=2, arrow_length_ratio=0.15)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()
    return fig
