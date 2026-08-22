#!/usr/bin/env python3
"""Stage 1 experiments: train a Perceptron on AND, OR, and XOR.

AND and OR converge to a separating line. XOR does not converge,
ever, because no separating line exists (see notes.md and the
diagram discussed alongside it).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perceptron import Perceptron  # noqa: E402
from viz.snapshot import Recorder  # noqa: E402
from viz.replay import (  # noqa: E402
    plot_boundary_evolution_2d,
    plot_error_curve,
)

GATES = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def to_pm1(y01):
    return np.where(y01 == 1, 1, -1)


def run_gate(name, y01, max_epochs=50, seed=0):
    y = to_pm1(y01)
    p = Perceptron(n_inputs=2, learning_rate=1.0, seed=seed)
    recorder = Recorder()

    converged, epochs, errors_per_epoch = p.fit(
        X, y, max_epochs=max_epochs, recorder=recorder
    )

    print(f"--- {name} ---")
    print(f"converged: {converged} after {epochs} epoch(s)")
    print(f"final weights: {p.weights}, bias: {p.bias:.3f}")
    for xi, yi in zip(X, y):
        pred = p.predict(xi)
        status = "OK" if pred == yi else "WRONG"
        print(f"  x={xi}, target={yi:+d}, pred={pred:+d}  [{status}]")
    print()

    plot_boundary_evolution_2d(
        recorder,
        X,
        y,
        title=f"{name}: decision boundary",
        save_path=f"plots/{name.lower()}_boundary.png",
    )
    plot_error_curve(
        errors_per_epoch,
        title=f"{name}: errors per epoch",
        save_path=f"plots/{name.lower()}_errors.png",
    )
    return converged, epochs, errors_per_epoch


if __name__ == "__main__":
    for gate_name, y01 in GATES.items():
        run_gate(gate_name, y01)
