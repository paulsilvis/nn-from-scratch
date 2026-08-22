#!/usr/bin/env python3
"""Numeric gradient check for the stage 7 additions: layer norm,
the position-wise feedforward, and masked attention - same
discipline as stages 5 and 6, verified before trusting them in
training.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_attention"))

from attention import attention_backward, attention_forward  # noqa: E402
from layers import (  # noqa: E402
    causal_mask,
    feedforward_backward,
    feedforward_forward,
    layernorm_backward,
    layernorm_forward,
)


def numeric_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f()
        x[idx] = orig - eps
        minus = f()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def report(name, analytic, numeric):
    max_abs_diff = np.max(np.abs(analytic - numeric))
    rel_diff = max_abs_diff / (np.max(np.abs(numeric)) + 1e-12)
    status = "PASS" if rel_diff < 1e-4 else "FAIL"
    print(
        f"{name}: max_abs_diff={max_abs_diff:.2e} "
        f"rel_diff={rel_diff:.2e}  [{status}]"
    )


def check_layernorm():
    rng = np.random.default_rng(0)
    batch, seq_len, d = 2, 3, 5
    x = rng.normal(size=(batch, seq_len, d))
    gamma = rng.normal(size=(d,)) + 1.0
    beta = rng.normal(size=(d,))
    upstream = rng.normal(size=(batch, seq_len, d))

    def loss():
        y, _ = layernorm_forward(x, gamma, beta)
        return float(np.sum(y * upstream))

    _, cache = layernorm_forward(x, gamma, beta)
    dx_a, dgamma_a, dbeta_a = layernorm_backward(upstream, cache)

    dx_n = numeric_gradient(loss, x)
    dgamma_n = numeric_gradient(loss, gamma)
    dbeta_n = numeric_gradient(loss, beta)

    report("layernorm dX", dx_a, dx_n)
    report("layernorm dgamma", dgamma_a, dgamma_n)
    report("layernorm dbeta", dbeta_a, dbeta_n)


def check_feedforward():
    rng = np.random.default_rng(1)
    batch, seq_len, d_model, d_ff = 2, 3, 4, 6
    x = rng.normal(size=(batch, seq_len, d_model))
    W1 = rng.normal(size=(d_ff, d_model))
    b1 = rng.normal(size=(d_ff,))
    W2 = rng.normal(size=(d_model, d_ff))
    b2 = rng.normal(size=(d_model,))
    upstream = rng.normal(size=(batch, seq_len, d_model))

    def loss():
        out, _ = feedforward_forward(x, W1, b1, W2, b2)
        return float(np.sum(out * upstream))

    _, cache = feedforward_forward(x, W1, b1, W2, b2)
    dx_a, dw1_a, db1_a, dw2_a, db2_a = feedforward_backward(
        upstream, cache, W1, W2
    )

    report("ff dX", dx_a, numeric_gradient(loss, x))
    report("ff dW1", dw1_a, numeric_gradient(loss, W1))
    report("ff db1", db1_a, numeric_gradient(loss, b1))
    report("ff dW2", dw2_a, numeric_gradient(loss, W2))
    report("ff db2", db2_a, numeric_gradient(loss, b2))


def check_masked_attention():
    rng = np.random.default_rng(2)
    batch, seq_len, d_model, d_k = 2, 4, 5, 3
    X = rng.normal(size=(batch, seq_len, d_model))
    W_Q = rng.normal(size=(d_model, d_k))
    W_K = rng.normal(size=(d_model, d_k))
    W_V = rng.normal(size=(d_model, d_k))
    mask = causal_mask(seq_len)
    upstream = rng.normal(size=(batch, seq_len, d_k))

    def loss():
        out, _ = attention_forward(X, W_Q, W_K, W_V, mask=mask)
        return float(np.sum(out * upstream))

    _, cache = attention_forward(X, W_Q, W_K, W_V, mask=mask)
    dx_a, _, _, _ = attention_backward(upstream, cache, W_Q, W_K, W_V)
    report("masked attention dX", dx_a, numeric_gradient(loss, X))


if __name__ == "__main__":
    check_layernorm()
    check_feedforward()
    check_masked_attention()
