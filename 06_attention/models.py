"""Two models for the associative-recall task, sharing the same
softmax + cross-entropy output head (unchanged from stage 4):

AttentionModel: embed -> single-head self-attention -> take the
    output at the query position -> dense -> softmax.
PlainMLP: flatten the whole sequence -> dense (sigmoid) -> dense ->
    softmax. No notion of "position" beyond raw concatenation order,
    and no content-based lookup mechanism at all - included as the
    baseline stage 1's XOR-style limitation demonstration.
"""

import numpy as np

from attention import attention_backward, attention_forward


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class AttentionModel:
    def __init__(self, d_model, d_k, n_classes, learning_rate=0.5, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / d_model)
        self.W_Q = rng.normal(0, scale, size=(d_model, d_k))
        self.W_K = rng.normal(0, scale, size=(d_model, d_k))
        self.W_V = rng.normal(0, scale, size=(d_model, d_k))

        scale_out = np.sqrt(1.0 / d_k)
        self.W_out = rng.normal(0, scale_out, size=(n_classes, d_k))
        self.b_out = np.zeros(n_classes)
        self.learning_rate = learning_rate

    def step(self, X, T, query_pos):
        out, attn_cache = attention_forward(X, self.W_Q, self.W_K, self.W_V)
        query_out = out[:, query_pos, :]

        z = query_out @ self.W_out.T + self.b_out
        p = softmax(z)

        batch = X.shape[0]
        delta = p - T
        grad_wout = (delta.T @ query_out) / batch
        grad_bout = delta.mean(axis=0)

        grad_query_out = delta @ self.W_out
        grad_out = np.zeros_like(out)
        grad_out[:, query_pos, :] = grad_query_out

        _, grad_wq, grad_wk, grad_wv = attention_backward(
            grad_out, attn_cache, self.W_Q, self.W_K, self.W_V
        )

        lr = self.learning_rate
        self.W_out -= lr * grad_wout
        self.b_out -= lr * grad_bout
        self.W_Q -= lr * grad_wq
        self.W_K -= lr * grad_wk
        self.W_V -= lr * grad_wv

        eps = 1e-12
        loss = -np.mean(np.sum(T * np.log(p + eps), axis=1))
        return loss

    def predict(self, X, query_pos):
        out, _ = attention_forward(X, self.W_Q, self.W_K, self.W_V)
        z = out[:, query_pos, :] @ self.W_out.T + self.b_out
        return np.argmax(z, axis=1)

    def accuracy(self, X, y, query_pos):
        return np.mean(self.predict(X, query_pos) == y)

    def n_params(self):
        return (
            self.W_Q.size
            + self.W_K.size
            + self.W_V.size
            + self.W_out.size
            + self.b_out.size
        )


class PlainMLP:
    def __init__(
        self, flat_size, n_hidden, n_classes, learning_rate=0.5, seed=None
    ):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(1.0 / flat_size)
        self.W1 = rng.normal(0, scale1, size=(n_hidden, flat_size))
        self.b1 = np.zeros(n_hidden)

        scale2 = np.sqrt(1.0 / n_hidden)
        self.W2 = rng.normal(0, scale2, size=(n_classes, n_hidden))
        self.b2 = np.zeros(n_classes)
        self.learning_rate = learning_rate

    def step(self, flat_X, T):
        z1 = flat_X @ self.W1.T + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2.T + self.b2
        p = softmax(z2)

        batch = flat_X.shape[0]
        delta2 = p - T
        grad_w2 = (delta2.T @ a1) / batch
        grad_b2 = delta2.mean(axis=0)

        delta1 = (delta2 @ self.W2) * a1 * (1 - a1)
        grad_w1 = (delta1.T @ flat_X) / batch
        grad_b1 = delta1.mean(axis=0)

        lr = self.learning_rate
        self.W2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_w1
        self.b1 -= lr * grad_b1

        eps = 1e-12
        loss = -np.mean(np.sum(T * np.log(p + eps), axis=1))
        return loss

    def predict(self, flat_X):
        z1 = flat_X @ self.W1.T + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2.T + self.b2
        return np.argmax(z2, axis=1)

    def accuracy(self, flat_X, y):
        return np.mean(self.predict(flat_X) == y)

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
