"""Stage 4: a fully-connected network for real MNIST digit
classification. Generalizes stage 2's TwoLayerNetwork in two ways:

1. Arbitrary hidden layer size (stage 2 fixed it at 2, sized for
   XOR; 784 pixel inputs need far more capacity).
2. A softmax + cross-entropy output layer for 10 classes, instead
   of a single sigmoid for one binary label. The output delta

       delta2_k = p_k - t_k

   is derived in notes.md and is the direct 10-class generalization
   of stage 2's delta2 = -(y-yhat)*yhat*(1-yhat) (that formula is
   the K=2 special case of this one). The hidden-layer delta is the
   same recursive idea as stage 2, just summed over all K outgoing
   paths instead of one:

       delta1_j = (sum_k delta2_k * W2[k,j]) * a1_j * (1 - a1_j)

Trained with minibatch stochastic gradient descent (stage 2 used
full-batch, workable there because the whole dataset was 4 points;
MNIST's 50000 examples make full-batch impractical per update).
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """Row-wise softmax, shifted by the row max for numerical
    stability (subtracting a constant from every logit in a row
    doesn't change the softmax output, since it cancels in the
    ratio - but it keeps exp() from overflowing).
    """
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class MLP:
    def __init__(
        self, n_inputs, n_hidden, n_classes, learning_rate=0.5, seed=None
    ):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(1.0 / n_inputs)
        scale2 = np.sqrt(1.0 / n_hidden)
        self.W1 = rng.normal(0, scale1, size=(n_hidden, n_inputs))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, scale2, size=(n_classes, n_hidden))
        self.b2 = np.zeros(n_classes)
        self.learning_rate = learning_rate

    def forward(self, X):
        z1 = X @ self.W1.T + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2.T + self.b2
        p = softmax(z2)
        return a1, p

    def backward(self, X, T, a1, p):
        batch = X.shape[0]
        delta2 = p - T
        grad_W2 = (delta2.T @ a1) / batch
        grad_b2 = delta2.mean(axis=0)

        delta1 = (delta2 @ self.W2) * a1 * (1 - a1)
        grad_W1 = (delta1.T @ X) / batch
        grad_b1 = delta1.mean(axis=0)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def step(self, X, T):
        a1, p = self.forward(X)
        grad_W1, grad_b1, grad_W2, grad_b2 = self.backward(X, T, a1, p)
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2

        eps = 1e-12
        loss = -np.mean(np.sum(T * np.log(p + eps), axis=1))
        return loss

    def predict(self, X):
        _, p = self.forward(X)
        return np.argmax(p, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(
        self,
        X,
        y,
        n_classes,
        epochs,
        batch_size=128,
        X_val=None,
        y_val=None,
        seed=None,
        recorder=None,
    ):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        T = np.eye(n_classes)[y]
        history = []

        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = start + batch_size
                idx = order[start:end]
                loss = self.step(X[idx], T[idx])
                epoch_loss += loss
                n_batches += 1

            train_acc = self.accuracy(X, y)
            record = {
                "epoch": epoch,
                "loss": epoch_loss / n_batches,
                "train_acc": train_acc,
            }
            if X_val is not None:
                record["val_acc"] = self.accuracy(X_val, y_val)
            history.append(record)
            if recorder is not None:
                recorder.record(epoch, **record)

        return history
