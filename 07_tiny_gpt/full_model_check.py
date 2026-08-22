#!/usr/bin/env python3
"""End-to-end numeric gradient check on the fully assembled TinyGPT
- every component passed its own isolated check, but bugs can still
hide in how they're wired together (a wrong cache threaded to the
wrong backward call, a missed residual gradient path). Checks a
few representative parameters across every part of the model:
embeddings, every layer's attention and feedforward weights, and
the output head.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinyGPT  # noqa: E402


def numeric_grad_scalar(f, arr, idx, eps=1e-4):
    orig = arr[idx]
    arr[idx] = orig + eps
    plus = f()
    arr[idx] = orig - eps
    minus = f()
    arr[idx] = orig
    return (plus - minus) / (2 * eps)


def loss_fn(net, tokens, targets):
    logits, _ = net.forward(tokens)
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    t_onehot = np.eye(net.vocab_size)[targets]
    eps = 1e-12
    return -np.mean(np.sum(t_onehot * np.log(p + eps), axis=-1))


def main():
    rng = np.random.default_rng(0)
    vocab_size, seq_len, batch = 8, 5, 3
    net = TinyGPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=6,
        d_k=6,
        d_ff=10,
        n_layers=2,
        learning_rate=0.0,
        seed=0,
    )
    tokens = rng.integers(0, vocab_size, size=(batch, seq_len))
    targets = rng.integers(0, vocab_size, size=(batch, seq_len))

    logits, caches = net.forward(tokens)
    grads = net.backward(tokens, targets, logits, caches)

    def loss():
        return loss_fn(net, tokens, targets)

    checks = [
        ("token_emb[2,3]", net.token_emb, (2, 3), grads["token_emb"]),
        ("pos_emb[1,0]", net.pos_emb, (1, 0), grads["pos_emb"]),
        ("W_head[0,2]", net.W_head, (0, 2), grads["W_head"]),
        ("b_head[3]", net.b_head, (3,), grads["b_head"]),
        (
            "ln_f_gamma[1]",
            net.ln_f_gamma,
            (1,),
            grads["ln_f_gamma"],
        ),
    ]
    for i, (layer, grad) in enumerate(zip(net.layers, grads["layers"])):
        checks.append(
            (f"layer{i}.W_Q[0,1]", layer["W_Q"], (0, 1), grad["W_Q"])
        )
        checks.append(
            (f"layer{i}.W_O[2,0]", layer["W_O"], (2, 0), grad["W_O"])
        )
        checks.append(
            (
                f"layer{i}.ff_W1[1,2]",
                layer["ff_W1"],
                (1, 2),
                grad["ff_W1"],
            )
        )
        checks.append(
            (
                f"layer{i}.ln1_gamma[0]",
                layer["ln1_gamma"],
                (0,),
                grad["ln1_gamma"],
            )
        )

    all_pass = True
    for name, arr, idx, grad_arr in checks:
        analytic = grad_arr[idx]
        numeric = numeric_grad_scalar(loss, arr, idx)
        rel_diff = abs(analytic - numeric) / (abs(numeric) + 1e-8)
        status = "PASS" if rel_diff < 1e-2 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"{name}: analytic={analytic:.6f} numeric={numeric:.6f} "
            f"rel_diff={rel_diff:.2e}  [{status}]"
        )

    print("\nALL PASS" if all_pass else "\nSOME FAILED")


if __name__ == "__main__":
    main()
