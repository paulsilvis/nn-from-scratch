"""Stage 2: a two-layer network, with backprop derived and coded by
hand (no autodiff).

Architecture: 2 inputs -> 2 hidden units (sigmoid) -> 1 output
(sigmoid). Notation matches the derivation in notes.md exactly:

    z1[j] = W1[j] . x + b1[j]        a1[j] = sigmoid(z1[j])
    z2    = W2 . a1 + b2             yhat  = sigmoid(z2)

    L = 0.5 * (y - yhat)^2

    delta2    = -(y - yhat) * yhat * (1 - yhat)
    delta1[j] = delta2 * W2[j] * a1[j] * (1 - a1[j])

    dL/dW2[j]    = delta2 * a1[j]        dL/db2    = delta2
    dL/dW1[j][i] = delta1[j] * x[i]      dL/db1[j] = delta1[j]

`forward` and `backward` operate on a single example at a time,
matching the derivation's notation one-to-one. `predict_batch` is a
vectorized restatement of the same forward-pass math for many
points at once - used only for plotting decision regions, not a
different algorithm.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TwoLayerNetwork:
    def __init__(self, n_inputs=2, n_hidden=2, learning_rate=1.0, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, size=(n_hidden, n_inputs))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1, size=n_hidden)
        self.b2 = 0.0
        self.learning_rate = learning_rate

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        yhat = sigmoid(z2)
        return z1, a1, z2, yhat

    def backward(self, x, y, a1, yhat):
        delta2 = -(y - yhat) * yhat * (1 - yhat)
        grad_W2 = delta2 * a1
        grad_b2 = delta2

        delta1 = delta2 * self.W2 * a1 * (1 - a1)
        grad_W1 = np.outer(delta1, x)
        grad_b1 = delta1

        return grad_W1, grad_b1, grad_W2, grad_b2

    def train_epoch(self, X, Y):
        """One full-batch gradient descent step: average the
        per-example gradients over all examples, then update once.
        """
        grad_W1_sum = np.zeros_like(self.W1)
        grad_b1_sum = np.zeros_like(self.b1)
        grad_W2_sum = np.zeros_like(self.W2)
        grad_b2_sum = 0.0
        total_loss = 0.0

        for x, y in zip(X, Y):
            _, a1, _, yhat = self.forward(x)
            total_loss += 0.5 * (y - yhat) ** 2
            gW1, gb1, gW2, gb2 = self.backward(x, y, a1, yhat)
            grad_W1_sum += gW1
            grad_b1_sum += gb1
            grad_W2_sum += gW2
            grad_b2_sum += gb2

        n = len(X)
        self.W1 -= self.learning_rate * grad_W1_sum / n
        self.b1 -= self.learning_rate * grad_b1_sum / n
        self.W2 -= self.learning_rate * grad_W2_sum / n
        self.b2 -= self.learning_rate * grad_b2_sum / n
        return total_loss / n

    def fit(self, X, Y, epochs=10000, recorder=None, record_every=50):
        loss_history = []
        for epoch in range(epochs):
            loss = self.train_epoch(X, Y)
            loss_history.append(loss)
            if recorder is not None and epoch % record_every == 0:
                recorder.record(
                    epoch,
                    loss=loss,
                    W1=self.W1.copy(),
                    b1=self.b1.copy(),
                    W2=self.W2.copy(),
                    b2=self.b2,
                )
        return loss_history

    def predict(self, x):
        _, _, _, yhat = self.forward(x)
        return yhat

    def predict_batch(self, X):
        """Vectorized forward pass for many points - same math as
        `forward`, restated for plotting decision regions.
        """
        Z1 = X @ self.W1.T + self.b1
        A1 = sigmoid(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return sigmoid(Z2)
