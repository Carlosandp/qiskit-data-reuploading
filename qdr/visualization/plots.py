"""Visualization utilities: decision boundaries, loss curves, Bloch spheres."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 50,
    feature_indices: tuple[int, int] = (0, 1),
    ax: "Axes | None" = None,
    title: str = "Decision Boundary",
) -> "Figure":
    """Plot a 2-D decision boundary for a fitted classifier.

    Parameters
    ----------
    model : fitted classifier
        Any object with ``predict(X)`` and ``predict_proba(X)`` methods.
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        True labels.
    resolution : int, optional
        Grid resolution (points per axis).  Default 50.
    feature_indices : tuple[int, int], optional
        Which two features to plot.  Default ``(0, 1)``.
    ax : Axes or None
        Existing axes to plot into.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    i, j = feature_indices
    X2 = X[:, [i, j]]

    x_min, x_max = X2[:, 0].min() - 0.3, X2[:, 0].max() + 0.3
    y_min, y_max = X2[:, 1].min() - 0.3, X2[:, 1].max() + 0.3

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Pad with zeros for extra features
    if X.shape[1] > 2:
        pad = np.zeros((len(grid), X.shape[1] - 2))
        grid_full = np.c_[grid, pad]
    else:
        grid_full = grid

    Z = model.predict(grid_full).reshape(xx.shape)

    classes = np.unique(y)
    cmap_bg = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"][: len(classes)])
    cmap_pts = ListedColormap(["#CC0000", "#006600", "#0000CC"][: len(classes)])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Encode class labels to integers for colour mapping
    label_map = {c: k for k, c in enumerate(classes)}
    Z_int = np.vectorize(label_map.get)(Z)
    y_int = np.array([label_map[c] for c in y])

    ax.contourf(xx, yy, Z_int, alpha=0.4, cmap=cmap_bg)
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y_int, cmap=cmap_pts, edgecolors="k", s=40)
    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Class")
    fig.tight_layout()
    return fig


def plot_loss_curve(
    loss_history: list[float],
    ax: "Axes | None" = None,
    title: str = "Training Loss",
    label: str = "loss",
) -> "Figure":
    """Plot the optimisation loss curve.

    Parameters
    ----------
    loss_history : list[float]
        Per-iteration loss values.
    ax : Axes or None
    title : str
    label : str

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(loss_history, label=label, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_benchmark_comparison(
    summary_df,
    metric: str = "accuracy",
    ax: "Axes | None" = None,
    title: str | None = None,
) -> "Figure":
    """Bar chart comparing models from a :meth:`~qdr.benchmarks.BenchmarkRunner.summary` DataFrame.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``BenchmarkRunner.summary()``.
    metric : str, optional
        Column to plot.  Default ``"accuracy"``.
    ax : Axes or None
    title : str or None

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    models = summary_df.index.tolist()
    values = summary_df[metric].tolist()
    colors = ["#4C72B0", "#DD8452", "#55A868"][: len(models)]

    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title or f"Model comparison — {metric}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bloch_sphere(
    statevector: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "Bloch Sphere",
) -> "Figure":
    """Render a single-qubit state on a Bloch sphere (3-D projection).

    Parameters
    ----------
    statevector : np.ndarray
        Complex array of shape ``(2,)`` representing a single-qubit statevector.
    ax : Axes or None
        Must be a 3-D axes (created with ``projection='3d'``).
    title : str

    Returns
    -------
    Figure
    """
    import matplotlib.pyplot as plt

    alpha, beta = statevector[0], statevector[1]
    # Bloch vector components
    bx = 2 * np.real(np.conj(alpha) * beta)
    by = 2 * np.imag(np.conj(alpha) * beta)
    bz = float(np.abs(alpha) ** 2 - np.abs(beta) ** 2)

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgrey", alpha=0.3, linewidth=0.4)

    # Axes
    for (start, end, label) in [
        ([0, 0, -1.3], [0, 0, 1.4], "Z"),
        ([-1.3, 0, 0], [1.4, 0, 0], "X"),
        ([0, -1.3, 0], [0, 1.4, 0], "Y"),
    ]:
        ax.plot(*zip(start, end), color="grey", linewidth=0.8)
        ax.text(*end, label, fontsize=9, ha="center")

    ax.quiver(0, 0, 0, bx, by, bz, color="royalblue", linewidth=2, arrow_length_ratio=0.15)
    ax.set_title(title)
    ax.set_axis_off()
    return fig
