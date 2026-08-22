"""Stage 3: the loss surface and gradient descent dynamics.

The full loss L is a function of all 9 network parameters - not
something anyone can plot directly. Here we freeze 7 of them (W1,
b1, b2) at a converged XOR solution from stage 2, and vary only the
2 output-layer weights (w2_1, w2_2). That gives a genuine 2D slice
through the true 9D surface: real, but not the whole picture.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_a1(X, W1, b1):
    """Hidden-layer activations for the frozen part of the network."""
    Z1 = X @ W1.T + b1
    return sigmoid(Z1)


def loss_given_output_weights(w2, b2, A1, y):
    """L(w2_1, w2_2) with W1, b1, b2 held fixed. w2 has shape (2,)."""
    z2 = A1 @ w2 + b2
    yhat = sigmoid(z2)
    return np.mean(0.5 * (y - yhat) ** 2)


def loss_surface(W1, b1, b2, X, y, w2_range=(-8, 8), n=120):
    """Evaluate the loss over a grid of (w2_1, w2_2) values."""
    A1 = compute_a1(X, W1, b1)
    ws = np.linspace(*w2_range, n)
    grid_a, grid_b = np.meshgrid(ws, ws)
    surface = np.zeros_like(grid_a)
    for i in range(n):
        for j in range(n):
            w2 = np.array([grid_a[i, j], grid_b[i, j]])
            surface[i, j] = loss_given_output_weights(w2, b2, A1, y)
    return grid_a, grid_b, surface


def gradient(w2, b2, A1, y):
    """dL/dw2 for the frozen-hidden-layer slice - the same delta2
    formula from stage 2, averaged over the training examples.
    """
    z2 = A1 @ w2 + b2
    yhat = sigmoid(z2)
    delta2 = -(y - yhat) * yhat * (1 - yhat)
    return (A1 * delta2[:, None]).mean(axis=0)


def gradient_descent_path(w2_init, b2, A1, y, learning_rate, steps):
    """Run gradient descent on w2 only, recording every point
    visited, so the trajectory can be drawn on top of the surface.
    """
    path = [w2_init.copy()]
    w2 = w2_init.copy()
    for _ in range(steps):
        g = gradient(w2, b2, A1, y)
        w2 = w2 - learning_rate * g
        path.append(w2.copy())
    return np.array(path)
