"""Stage 6: single-head self-attention, forward and backward,
derived by hand.

    Q = X @ W_Q      K = X @ W_K      V = X @ W_V
    S = Q @ K.T / sqrt(d_k)
    A = softmax_rows(S)
    O = A @ V

Backward, given G = dL/dO:

    dL/dV = A.T @ G
    dL/dA = G @ V.T
    dL/dS[i] = a_i * (g_i - (a_i . g_i))   -- softmax row Jacobian,
        where a_i, g_i are row i of A, dL/dA (the general softmax
        backward rule - stage 4's "p - t" shortcut was this formula
        specialized to being paired with cross-entropy loss, which
        isn't the case here)
    dL/dQ = (dL/dS) @ K / sqrt(d_k)
    dL/dK = (dL/dS).T @ Q / sqrt(d_k)
    dL/dW_Q = X.T @ dL/dQ   (same pattern for W_K, W_V)
    dL/dX   = dL/dQ @ W_Q.T + dL/dK @ W_K.T + dL/dV @ W_V.T
        (X feeds all three projections, so its gradient sums all
        three paths - same "sum every path used" principle as
        stage 4's hidden-layer delta)

Operates on a single sequence (seq_len, d_model) with no batch
dimension in the derivation above; batched versions
(batch, seq_len, d_model) are handled with an extra leading axis and
np.einsum, same underlying math.
"""

import numpy as np


def softmax_rows(s):
    s_shifted = s - s.max(axis=-1, keepdims=True)
    exp_s = np.exp(s_shifted)
    return exp_s / exp_s.sum(axis=-1, keepdims=True)


def attention_forward(X, W_Q, W_K, W_V):
    """X: (batch, seq_len, d_model). Returns cache with everything
    attention_backward needs.
    """
    d_k = W_Q.shape[1]
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    S = np.einsum("bid,bjd->bij", Q, K) / np.sqrt(d_k)
    A = softmax_rows(S)
    output = A @ V
    cache = (X, Q, K, V, A)
    return output, cache


def attention_backward(G, cache, W_Q, W_K, W_V):
    """G = dL/dO, shape (batch, seq_len, d_v). Returns
    (grad_X, grad_WQ, grad_WK, grad_WV).
    """
    X, Q, K, V, A = cache
    d_k = W_Q.shape[1]
    batch = X.shape[0]

    grad_V = np.einsum("bij,bik->bjk", A, G)
    grad_A = np.einsum("bik,bjk->bij", G, V)

    row_dot = np.sum(A * grad_A, axis=-1, keepdims=True)
    grad_S = A * (grad_A - row_dot)

    grad_Q = np.einsum("bij,bjd->bid", grad_S, K) / np.sqrt(d_k)
    grad_K = np.einsum("bij,bid->bjd", grad_S, Q) / np.sqrt(d_k)

    grad_WQ = np.einsum("bsd,bse->de", X, grad_Q) / batch
    grad_WK = np.einsum("bsd,bse->de", X, grad_K) / batch
    grad_WV = np.einsum("bsd,bse->de", X, grad_V) / batch

    grad_X = grad_Q @ W_Q.T + grad_K @ W_K.T + grad_V @ W_V.T

    return grad_X, grad_WQ, grad_WK, grad_WV
