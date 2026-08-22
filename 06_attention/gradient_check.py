#!/usr/bin/env python3
"""Numeric gradient check for attention_backward, on a tiny random
example - verifying the hand-derived formulas (including the
general softmax-backward rule, not the cross-entropy-specialized
shortcut) before trusting them in training.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention import attention_backward, attention_forward  # noqa: E402


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


def main():
    rng = np.random.default_rng(0)
    batch, seq_len, d_model, d_k = 2, 4, 5, 3

    X = rng.normal(size=(batch, seq_len, d_model))
    W_Q = rng.normal(size=(d_model, d_k))
    W_K = rng.normal(size=(d_model, d_k))
    W_V = rng.normal(size=(d_model, d_k))
    upstream = rng.normal(size=(batch, seq_len, d_k))

    def loss():
        out, _ = attention_forward(X, W_Q, W_K, W_V)
        return float(np.sum(out * upstream))

    _, cache = attention_forward(X, W_Q, W_K, W_V)
    grad_x_a, grad_wq_a, grad_wk_a, grad_wv_a = attention_backward(
        upstream, cache, W_Q, W_K, W_V
    )
    # loss = sum(O * upstream) => dL/dO = upstream exactly, with no
    # /batch averaging - undo attention_backward's /batch averaging
    # on the weight grads for a fair comparison.
    grad_wq_a = grad_wq_a * batch
    grad_wk_a = grad_wk_a * batch
    grad_wv_a = grad_wv_a * batch

    grad_x_n = numeric_gradient(loss, X)
    grad_wq_n = numeric_gradient(loss, W_Q)
    grad_wk_n = numeric_gradient(loss, W_K)
    grad_wv_n = numeric_gradient(loss, W_V)

    checks = [
        ("dX", grad_x_a, grad_x_n),
        ("dW_Q", grad_wq_a, grad_wq_n),
        ("dW_K", grad_wk_a, grad_wk_n),
        ("dW_V", grad_wv_a, grad_wv_n),
    ]
    for name, analytic, numeric in checks:
        max_abs_diff = np.max(np.abs(analytic - numeric))
        rel_diff = max_abs_diff / (np.max(np.abs(numeric)) + 1e-12)
        status = "PASS" if rel_diff < 1e-4 else "FAIL"
        print(
            f"{name}: max_abs_diff={max_abs_diff:.2e} "
            f"rel_diff={rel_diff:.2e}  [{status}]"
        )


if __name__ == "__main__":
    main()
