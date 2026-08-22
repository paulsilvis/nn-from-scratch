#!/usr/bin/env python3
"""Numeric gradient check for conv_backward, on a tiny random
example - verifying the hand-derived formulas against finite
differences before trusting them in a real training run.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conv import conv_backward, conv_forward  # noqa: E402


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
    batch, hgt, wdt, k, n_filters = 2, 6, 6, 3, 2

    X = rng.normal(size=(batch, hgt, wdt))
    K = rng.normal(size=(n_filters, k, k))
    b = rng.normal(size=(n_filters,))
    upstream = rng.normal(size=(batch, hgt - k + 1, wdt - k + 1, n_filters))

    def loss():
        _, z = conv_forward(X, K, b)
        return float(np.sum(z * upstream))

    patches, z = conv_forward(X, K, b)
    grad_k_analytic, grad_b_analytic, grad_x_analytic = conv_backward(
        patches, upstream, K, X.shape
    )
    # loss = sum(Z * upstream), so dL/dZ = upstream exactly (no /batch
    # averaging in this synthetic loss) - undo conv_backward's /batch
    # averaging for a fair comparison against finite differences.
    grad_k_analytic = grad_k_analytic * batch
    grad_b_analytic = grad_b_analytic * batch
    grad_x_analytic = grad_x_analytic * batch

    grad_k_numeric = numeric_gradient(loss, K)
    grad_b_numeric = numeric_gradient(loss, b)
    grad_x_numeric = numeric_gradient(loss, X)

    for name, analytic, numeric in [
        ("dK", grad_k_analytic, grad_k_numeric),
        ("db", grad_b_analytic, grad_b_numeric),
        ("dX", grad_x_analytic, grad_x_numeric),
    ]:
        max_abs_diff = np.max(np.abs(analytic - numeric))
        rel_diff = max_abs_diff / (np.max(np.abs(numeric)) + 1e-12)
        status = "PASS" if rel_diff < 1e-4 else "FAIL"
        print(
            f"{name}: max_abs_diff={max_abs_diff:.2e} "
            f"rel_diff={rel_diff:.2e}  [{status}]"
        )


if __name__ == "__main__":
    main()
