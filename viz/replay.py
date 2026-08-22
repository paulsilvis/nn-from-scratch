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
