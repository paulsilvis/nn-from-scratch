"""Replay utilities: turn recorded Snapshot sequences into plots.

Stage-specific code decides *what* to record; this module only knows
how to turn generic snapshot fields into pictures, so the same
plotting logic can serve every stage in the roadmap, not just the
perceptron.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_boundary_evolution_2d(
    recorder,
    X,
    y,
    title="Decision boundary",
    class_labels=("false", "true"),
    save_path=None,
):
    """Plot 2D points colored by class, with every recorded decision
    line overlaid: faint lines for early snapshots, a solid black
    line for the final one. Expects each snapshot's data dict to
    contain 'weights' (length-2 array) and 'bias' (scalar).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    snapshots = recorder.snapshots
    n = len(snapshots)
    xs = np.linspace(-0.5, 1.5, 100)

    for i, snap in enumerate(snapshots):
        w = snap.data["weights"]
        b = snap.data["bias"]
        is_last = i == n - 1
        alpha = 0.15 + 0.75 * (i / max(n - 1, 1))
        color = "black" if is_last else "0.6"
        lw = 2 if is_last else 1
        if abs(w[1]) > 1e-9:
            ys_line = -(w[0] * xs + b) / w[1]
            ax.plot(xs, ys_line, color=color, alpha=alpha, linewidth=lw)
        elif abs(w[0]) > 1e-9:
            x_vert = -b / w[0]
            ax.axvline(x_vert, color=color, alpha=alpha, linewidth=lw)

    for xi, yi in zip(X, y):
        marker_color = "teal" if yi == 1 else "coral"
        ax.scatter(
            xi[0],
            xi[1],
            s=140,
            color=marker_color,
            edgecolor="black",
            zorder=5,
        )
        label = class_labels[1] if yi == 1 else class_labels[0]
        ax.annotate(
            f"({xi[0]:.0f},{xi[1]:.0f}) {label}",
            (xi[0], xi[1]),
            textcoords="offset points",
            xytext=(8, 8),
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_loss_curve(loss_history, title, save_path=None):
    """Plot continuous loss per epoch (as opposed to the discrete
    misclassification counts in plot_error_curve, which only makes
    sense for the hard-threshold perceptron of stage 1).
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_decision_region_2d(
    model,
    X,
    y,
    title="Decision region",
    class_labels=("false", "true"),
    save_path=None,
):
    """Plot a 2D input space colored by the model's continuous
    output (a filled contour, since the boundary need not be a
    straight line once there's a hidden layer), with a solid contour
    at the 0.5 decision threshold and the training points overlaid.
    Expects `model.predict_batch(X)` to exist.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    xs = np.linspace(-0.5, 1.5, 200)
    ys = np.linspace(-0.5, 1.5, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict_batch(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=20, cmap="RdYlGn_r", alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)

    for xi, yi in zip(X, y):
        marker_color = "teal" if yi == 1 else "coral"
        ax.scatter(
            xi[0],
            xi[1],
            s=140,
            color=marker_color,
            edgecolor="black",
            zorder=5,
        )
        label = class_labels[1] if yi == 1 else class_labels[0]
        ax.annotate(
            f"({xi[0]:.0f},{xi[1]:.0f}) {label}",
            (xi[0], xi[1]),
            textcoords="offset points",
            xytext=(8, 8),
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_surface_3d(grid_a, grid_b, surface, title, save_path=None):
    """3D view of a loss surface (see loss_surface.py) - a static
    picture of the bowl/saddle shape being descended.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        grid_a, grid_b, surface, cmap="viridis", alpha=0.9, linewidth=0
    )
    ax.set_xlabel("w2_1")
    ax.set_ylabel("w2_2")
    ax.set_zlabel("loss")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_contour_with_paths(
    grid_a,
    grid_b,
    surface,
    paths,
    labels,
    title,
    save_path=None,
):
    """Top-down contour of a loss surface with one or more gradient
    descent trajectories overlaid. `paths` is a list of (n_steps, 2)
    arrays; `labels` a same-length list of legend names.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cs = ax.contourf(grid_a, grid_b, surface, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="loss")

    colors = ["#E24B4A", "#378ADD", "#F2A623"]
    for i, (path, label) in enumerate(zip(paths, labels)):
        color = colors[i % len(colors)]
        ax.plot(
            path[:, 0], path[:, 1], color=color, linewidth=1.5, label=label
        )
        ax.scatter(
            path[0, 0], path[0, 1], color=color, marker="o", s=60, zorder=5
        )
        ax.scatter(
            path[-1, 0], path[-1, 1], color=color, marker="x", s=80, zorder=5
        )

    ax.set_xlabel("w2_1")
    ax.set_ylabel("w2_2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_error_curve(errors_per_epoch, title, save_path=None):
    """Plot misclassification count per epoch: a convergence curve
    for AND/OR, and a curve that never reaches zero for XOR.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    epochs = np.arange(1, len(errors_per_epoch) + 1)
    ax.plot(epochs, errors_per_epoch, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("misclassifications")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig
