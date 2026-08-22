"""Stage 5: a small CNN - conv -> ReLU -> max-pool -> flatten ->
dense (sigmoid) -> dense (softmax) - built from conv.py's
from-scratch layers plus stage 4's dense-layer machinery (softmax +
cross-entropy delta = p - t, same derivation, unchanged here).

Architecture, for a 28x28 MNIST input, n_filters=8, k=3:
  conv:    (28,28) -> (26,26,8)   [8 filters, 3x3, no padding]
  relu:    elementwise
  pool:    (26,26,8) -> (13,13,8) [2x2 max-pool]
  flatten: (13,13,8) -> 1352
  dense1:  1352 -> hidden (sigmoid)
  dense2:  hidden -> 10 (softmax)
"""

import numpy as np

from conv import (
    conv_backward,
    conv_forward,
    maxpool_backward,
    maxpool_forward,
    relu,
    relu_backward,
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class SmallCNN:
    def __init__(
        self,
        image_size=28,
        n_filters=8,
        kernel_size=3,
        n_hidden=128,
        n_classes=10,
        learning_rate=0.5,
        seed=None,
    ):
        rng = np.random.default_rng(seed)
        k = kernel_size
        pooled_size = (image_size - k + 1) // 2
        flat_size = pooled_size * pooled_size * n_filters

        self.K = rng.normal(0, np.sqrt(1.0 / (k * k)), size=(n_filters, k, k))
        self.bK = np.zeros(n_filters)

        scale1 = np.sqrt(1.0 / flat_size)
        self.W1 = rng.normal(0, scale1, size=(n_hidden, flat_size))
        self.b1 = np.zeros(n_hidden)

        scale2 = np.sqrt(1.0 / n_hidden)
        self.W2 = rng.normal(0, scale2, size=(n_classes, n_hidden))
        self.b2 = np.zeros(n_classes)

        self.learning_rate = learning_rate
        self.flat_size = flat_size

    def forward(self, X):
        patches, z_conv = conv_forward(X, self.K, self.bK)
        a_conv = relu(z_conv)
        pooled, pool_cache = maxpool_forward(a_conv, size=2)

        batch = X.shape[0]
        flat = pooled.reshape(batch, -1)

        z1 = flat @ self.W1.T + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2.T + self.b2
        p = softmax(z2)

        cache = (patches, z_conv, pool_cache, flat, a1, p)
        return cache

    def backward(self, X, T, cache):
        patches, z_conv, pool_cache, flat, a1, p = cache
        batch = X.shape[0]

        delta2 = p - T
        grad_w2 = (delta2.T @ a1) / batch
        grad_b2 = delta2.mean(axis=0)

        delta1 = (delta2 @ self.W2) * a1 * (1 - a1)
        grad_w1 = (delta1.T @ flat) / batch
        grad_b1 = delta1.mean(axis=0)

        grad_flat = delta1 @ self.W1
        pooled_shape = pool_cache[1].shape
        grad_pooled = grad_flat.reshape(pooled_shape)

        grad_a_conv = maxpool_backward(grad_pooled, pool_cache)
        grad_z_conv = relu_backward(grad_a_conv, z_conv)

        grad_k, grad_bk, _ = conv_backward(
            patches, grad_z_conv, self.K, X.shape
        )

        return grad_k, grad_bk, grad_w1, grad_b1, grad_w2, grad_b2

    def step(self, X, T):
        cache = self.forward(X)
        p = cache[-1]
        grads = self.backward(X, T, cache)
        grad_k, grad_bk, grad_w1, grad_b1, grad_w2, grad_b2 = grads

        lr = self.learning_rate
        self.K -= lr * grad_k
        self.bK -= lr * grad_bk
        self.W1 -= lr * grad_w1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_w2
        self.b2 -= lr * grad_b2

        eps = 1e-12
        loss = -np.mean(np.sum(T * np.log(p + eps), axis=1))
        return loss

    def predict(self, X):
        cache = self.forward(X)
        return np.argmax(cache[-1], axis=1)

    def accuracy(self, X, y, batch_size=1000):
        """Batched to avoid materializing conv patches for the
        entire input at once - see notes.md for why this isn't
        optional at MNIST scale: an unbatched call on all 50,000
        training images allocates a (50000,26,26,3,3) patches array
        (~2.4 GB) and gets OOM-killed on a memory-constrained
        machine, exactly the failure this method predicts if you
        don't batch it.
        """
        correct = 0
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            correct += np.sum(self.predict(X[start:end]) == y[start:end])
        return correct / len(X)

    def n_params(self):
        return (
            self.K.size
            + self.bK.size
            + self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
        )

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
    ):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        t_onehot = np.eye(n_classes)[y]
        history = []

        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = start + batch_size
                idx = order[start:end]
                loss = self.step(X[idx], t_onehot[idx])
                epoch_loss += loss
                n_batches += 1

            record = {
                "epoch": epoch,
                "loss": epoch_loss / n_batches,
                "train_acc": self.accuracy(X, y),
            }
            if X_val is not None:
                record["val_acc"] = self.accuracy(X_val, y_val)
            history.append(record)

        return history
