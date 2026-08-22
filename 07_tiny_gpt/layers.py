"""Stage 7: the pieces needed beyond stage 6's attention - layer
normalization, a position-wise feedforward sublayer, and causal
masking - to assemble a real (tiny) transformer.

Layer normalization, per position, over the feature axis (last
axis, size d):

    mu    = mean(x)                    (per position)
    var   = mean((x - mu)^2)
    xhat  = (x - mu) / sqrt(var + eps)
    y     = gamma * xhat + beta

Backward, given dy = dL/dy (same shape as x, gamma/beta broadcast
over all but the last axis):

    dgamma = sum_{batch,pos}(dy * xhat)
    dbeta  = sum_{batch,pos}(dy)
    dxhat  = dy * gamma
    dvar   = sum_d(dxhat * (x-mu) * -0.5*(var+eps)^-1.5)
    dmu    = sum_d(dxhat * -1/sqrt(var+eps))
             + dvar * mean_d(-2*(x-mu))
    dx     = dxhat/sqrt(var+eps) + dvar*2*(x-mu)/d + dmu/d

This is the standard layer-norm backward derivation; verified
against finite differences in gradient_check.py rather than
re-derived line by line in prose here, same discipline as stages 5
and 6.
"""

import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x: (..., d). gamma, beta: (d,)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * std_inv
    y = gamma * xhat + beta
    cache = (x, mu, var, std_inv, xhat, gamma)
    return y, cache


def layernorm_backward(dy, cache):
    x, mu, var, std_inv, xhat, gamma = cache
    d = x.shape[-1]

    dgamma = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
    dbeta = np.sum(dy, axis=tuple(range(dy.ndim - 1)))

    dxhat = dy * gamma
    dvar = np.sum(dxhat * (x - mu) * -0.5 * std_inv**3, axis=-1, keepdims=True)
    dmu = np.sum(dxhat * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(
        -2.0 * (x - mu), axis=-1, keepdims=True
    )

    dx = dxhat * std_inv + dvar * 2.0 * (x - mu) / d + dmu / d
    return dx, dgamma, dbeta


def relu(z):
    return np.maximum(0, z)


def relu_backward(g, z):
    return g * (z > 0)


def feedforward_forward(x, W1, b1, W2, b2):
    """Position-wise feedforward: same 2 dense layers applied
    identically at every position (weight sharing across positions,
    same idea as attention's Q/K/V projections). x: (..., d_model).
    """
    z1 = x @ W1.T + b1
    a1 = relu(z1)
    z2 = a1 @ W2.T + b2
    cache = (x, z1, a1)
    return z2, cache


def feedforward_backward(dz2, cache, W1, W2):
    x, z1, a1 = cache
    axes = tuple(range(dz2.ndim - 1))

    grad_w2 = np.tensordot(dz2, a1, axes=(axes, axes))
    grad_b2 = dz2.sum(axis=axes)

    da1 = dz2 @ W2
    dz1 = relu_backward(da1, z1)

    grad_w1 = np.tensordot(dz1, x, axes=(axes, axes))
    grad_b1 = dz1.sum(axis=axes)

    grad_x = dz1 @ W1
    return grad_x, grad_w1, grad_b1, grad_w2, grad_b2


def causal_mask(seq_len):
    """Additive mask: 0 where attending to self/past, -inf where
    attending to the future. Add to raw attention scores before
    softmax so future positions get exactly zero weight.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask == 1, -np.inf, 0.0)
