#!/usr/bin/env python3
"""Stage 2 experiments: train the two-layer network on AND, OR, XOR.

Unlike stage 1, this network has a hidden layer, so it should
succeed on XOR where the single perceptron provably could not.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network import TwoLayerNetwork  # noqa: E402
from viz.snapshot import Recorder  # noqa: E402
from viz.replay import (  # noqa: E402
    plot_decision_region_2d,
    plot_loss_curve,
)

GATES = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def run_gate(name, y, epochs=10000, learning_rate=5.0, seed=0):
    net = TwoLayerNetwork(
        n_inputs=2, n_hidden=2, learning_rate=learning_rate, seed=seed
    )
    recorder = Recorder()
    loss_history = net.fit(X, y, epochs=epochs, recorder=recorder)

    print(f"--- {name} ---")
    print(f"final loss: {loss_history[-1]:.6f}")
    all_correct = True
    for xi, yi in zip(X, y):
        pred = net.predict(xi)
        pred_class = 1 if pred >= 0.5 else 0
        status = "OK" if pred_class == yi else "WRONG"
        if pred_class != yi:
            all_correct = False
        print(
            f"  x={xi}, target={yi}, "
            f"pred={pred:.4f} (class {pred_class})  [{status}]"
        )
    print(f"all correct: {all_correct}\n")

    plot_loss_curve(
        loss_history,
        title=f"{name}: loss per epoch",
        save_path=f"plots/{name.lower()}_loss.png",
    )
    plot_decision_region_2d(
        net,
        X,
        y,
        title=f"{name}: decision region",
        save_path=f"plots/{name.lower()}_region.png",
    )
    return net, loss_history, all_correct


if __name__ == "__main__":
    for gate_name, y in GATES.items():
        run_gate(gate_name, y)
