"""Stage 1: a single perceptron, from scratch.

Implements the perceptron learning rule (not gradient descent) by
hand: for each misclassified point (x, y) with y in {-1, +1},

    w <- w + eta * y * x
    b <- b + eta * y

Trained here on AND, OR, and XOR in experiment.py to make the
Minsky-Papert failure concrete: XOR is not linearly separable, so
this algorithm cannot converge on it no matter how long it runs.
See notes.md for the geometric and algebraic derivation of why.
"""

import numpy as np


class Perceptron:
    """A single linear threshold unit.

    Decision rule: f(x) = sign(w . x + b), with sign(0) defined as
    +1 by convention (an arbitrary but necessary choice - see
    notes.md).
    """

    def __init__(self, n_inputs, learning_rate=1.0, seed=None):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-0.5, 0.5, size=n_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def net_input(self, x):
        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        return 1 if self.net_input(x) >= 0 else -1

    def fit(self, X, y, max_epochs=100, recorder=None):
        """Train using the perceptron learning rule.

        Returns (converged, epochs_run, errors_per_epoch).
        If `recorder` is given, a Snapshot is logged after every
        single weight update (not just every epoch), so replay.py
        can show the boundary moving update-by-update.
        """
        step = 0
        errors_per_epoch = []
        if recorder is not None:
            recorder.record(
                step,
                weights=self.weights.copy(),
                bias=self.bias,
            )

        for epoch in range(1, max_epochs + 1):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != yi:
                    self.weights = self.weights + (
                        self.learning_rate * yi * xi
                    )
                    self.bias = self.bias + self.learning_rate * yi
                    errors += 1
                    step += 1
                    if recorder is not None:
                        recorder.record(
                            step,
                            weights=self.weights.copy(),
                            bias=self.bias,
                        )
            errors_per_epoch.append(errors)
            if errors == 0:
                return True, epoch, errors_per_epoch

        return False, max_epochs, errors_per_epoch
