#!/usr/bin/env python3
"""Stage 3: visualize the loss landscape and gradient descent.

Trains stage 2's TwoLayerNetwork on XOR, freezes W1/b1/b2 at that
converged solution, and slices the true 9-parameter loss surface
down to the 2 output weights (w2_1, w2_2) - the only pair that can
be plotted directly. Then runs gradient descent on that 2D slice
twice: once with a learning rate that converges, once with one
large enough to diverge, to see both behaviors directly rather than
just being told about them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_backprop"))

from network import TwoLayerNetwork  # noqa: E402
from loss_surface import (  # noqa: E402
    compute_a1,
    gradient_descent_path,
    loss_given_output_weights,
    loss_surface,
)
from viz.replay import (  # noqa: E402
    plot_contour_with_paths,
    plot_surface_3d,
)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0])


def main():
    net = TwoLayerNetwork(n_inputs=2, n_hidden=2, learning_rate=5.0, seed=0)
    net.fit(X, Y, epochs=10000)
    print(f"converged W2: {net.W2}, b2: {net.b2:.4f}")

    grid_a, grid_b, surface = loss_surface(
        net.W1, net.b1, net.b2, X, Y, w2_range=(-8, 8), n=120
    )

    plot_surface_3d(
        grid_a,
        grid_b,
        surface,
        title="Loss surface: slice through (w2_1, w2_2)",
        save_path="plots/loss_surface_3d.png",
    )

    A1 = compute_a1(X, net.W1, net.b1)
    start = np.array([-6.0, 6.0])

    good_path = gradient_descent_path(
        start, net.b2, A1, Y, learning_rate=2.0, steps=150
    )
    bad_path = gradient_descent_path(
        start, net.b2, A1, Y, learning_rate=300.0, steps=60
    )

    good_loss = loss_given_output_weights(good_path[-1], net.b2, A1, Y)
    bad_loss = loss_given_output_weights(bad_path[-1], net.b2, A1, Y)
    print(
        f"good path (lr=2.0) final w2: {good_path[-1]}, "
        f"loss={good_loss:.4f}"
    )
    print(
        f"bad path (lr=300.0) final w2: {bad_path[-1]}, "
        f"loss={bad_loss:.4f}"
    )

    plot_contour_with_paths(
        grid_a,
        grid_b,
        surface,
        paths=[good_path],
        labels=["lr=2.0 (converges)"],
        title="Gradient descent: a learning rate that works",
        save_path="plots/descent_converging.png",
    )
    plot_contour_with_paths(
        grid_a,
        grid_b,
        surface,
        paths=[bad_path],
        labels=["lr=300.0 (overshoots, gets trapped)"],
        title="Gradient descent: a learning rate that doesn't",
        save_path="plots/descent_diverging.png",
    )


if __name__ == "__main__":
    main()
