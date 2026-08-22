#!/usr/bin/env python3
"""Stage 6: attention vs. a plain MLP on associative recall.

Both models see the same input; the query position is always the
last one (n_pairs), fixed and known. The plain MLP gets the whole
flattened sequence as input - in principle it has access to every
bit of information the attention model does, it just has no
mechanism to search by content across positions, only fixed
per-position weights.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import AttentionModel, PlainMLP  # noqa: E402
from task import make_batch  # noqa: E402

N_PAIRS = 4
N_KEYS = 6
N_VALUES = 10
D_MODEL = N_KEYS + N_VALUES + 1
SEQ_LEN = N_PAIRS + 1
QUERY_POS = N_PAIRS

N_TRAIN = 20000
N_VAL = 2000
N_TEST = 2000
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.5


def train_attention(X_train, y_train, X_val, y_val):
    net = AttentionModel(
        d_model=D_MODEL,
        d_k=16,
        n_classes=N_VALUES,
        learning_rate=LEARNING_RATE,
        seed=0,
    )
    rng = np.random.default_rng(0)
    t_onehot = np.eye(N_VALUES)[y_train]
    n = X_train.shape[0]
    history = []

    t0 = time.time()
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = order[start:end]
            epoch_loss += net.step(X_train[idx], t_onehot[idx], QUERY_POS)
            n_batches += 1
        val_acc = net.accuracy(X_val, y_val, QUERY_POS)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / n_batches,
                "val_acc": val_acc,
            }
        )
    elapsed = time.time() - t0
    return net, history, elapsed


def train_mlp(X_train, y_train, X_val, y_val):
    flat_train = X_train.reshape(X_train.shape[0], -1)
    flat_val = X_val.reshape(X_val.shape[0], -1)

    net = PlainMLP(
        flat_size=flat_train.shape[1],
        n_hidden=64,
        n_classes=N_VALUES,
        learning_rate=LEARNING_RATE,
        seed=0,
    )
    rng = np.random.default_rng(0)
    t_onehot = np.eye(N_VALUES)[y_train]
    n = flat_train.shape[0]
    history = []

    t0 = time.time()
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = order[start:end]
            epoch_loss += net.step(flat_train[idx], t_onehot[idx])
            n_batches += 1
        val_acc = net.accuracy(flat_val, y_val)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / n_batches,
                "val_acc": val_acc,
            }
        )
    elapsed = time.time() - t0
    return net, history, elapsed


def plot_comparison(hist_attn, hist_mlp, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(
        [r["epoch"] for r in hist_attn],
        [r["loss"] for r in hist_attn],
        label="attention",
    )
    ax_loss.plot(
        [r["epoch"] for r in hist_mlp],
        [r["loss"] for r in hist_mlp],
        label="plain MLP",
    )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("Loss: attention vs. plain MLP")
    ax_loss.legend(fontsize=9)

    ax_acc.plot(
        [r["epoch"] for r in hist_attn],
        [r["val_acc"] for r in hist_attn],
        label="attention",
    )
    ax_acc.plot(
        [r["epoch"] for r in hist_mlp],
        [r["val_acc"] for r in hist_mlp],
        label="plain MLP",
    )
    ax_acc.axhline(
        1.0 / N_VALUES, color="gray", linestyle="--", label="chance"
    )
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_title("Associative recall: attention vs. plain MLP")
    ax_acc.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    rng = np.random.default_rng(1)
    X_train, y_train = make_batch(N_TRAIN, N_PAIRS, N_KEYS, N_VALUES, rng)
    X_val, y_val = make_batch(N_VAL, N_PAIRS, N_KEYS, N_VALUES, rng)
    X_test, y_test = make_batch(N_TEST, N_PAIRS, N_KEYS, N_VALUES, rng)

    print(
        f"task: {N_PAIRS} pairs, {N_KEYS} possible keys, "
        f"{N_VALUES} possible values, chance accuracy = "
        f"{1.0/N_VALUES:.3f}"
    )

    print("\n--- attention model ---")
    attn_net, attn_hist, attn_time = train_attention(
        X_train, y_train, X_val, y_val
    )
    attn_test_acc = attn_net.accuracy(X_test, y_test, QUERY_POS)
    print(f"params: {attn_net.n_params():,}")
    print(f"trained in {attn_time:.1f}s over {EPOCHS} epochs")
    print(f"final val acc:  {attn_hist[-1]['val_acc']:.4f}")
    print(f"test acc:       {attn_test_acc:.4f}")

    print("\n--- plain MLP baseline ---")
    mlp_net, mlp_hist, mlp_time = train_mlp(X_train, y_train, X_val, y_val)
    flat_test = X_test.reshape(X_test.shape[0], -1)
    mlp_test_acc = mlp_net.accuracy(flat_test, y_test)
    print(f"params: {mlp_net.n_params():,}")
    print(f"trained in {mlp_time:.1f}s over {EPOCHS} epochs")
    print(f"final val acc:  {mlp_hist[-1]['val_acc']:.4f}")
    print(f"test acc:       {mlp_test_acc:.4f}")

    plot_comparison(attn_hist, mlp_hist, "plots/attention_vs_mlp.png")


if __name__ == "__main__":
    main()
