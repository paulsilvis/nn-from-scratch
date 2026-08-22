#!/usr/bin/env python3
"""Stage 5: train the from-scratch CNN on real MNIST and compare
against stage 4's plain MLP at the same hidden-layer width (128),
so the comparison isolates what the conv+pool feature extractor
adds, not a difference in dense-layer capacity.

Per-epoch accuracy tracking uses a fixed subset (2000 examples) of
train/val rather than the full sets - stage 4's memory_scaling.py
addendum flagged exactly this cost (a full-dataset forward pass
every epoch) as the dominant inefficiency in that code; fixed here
by design rather than repeating it. Final reported numbers still
use the complete train/val/test sets, evaluated once at the end.
"""

import gzip
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn import SmallCNN  # noqa: E402

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "04_mnist" / "data" / "mnist.pkl.gz"
)
MNIST_URL = (
    "https://raw.githubusercontent.com/mnielsen/"
    "neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
)
EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.5
MONITOR_SUBSET = 2000


def load_mnist():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, DATA_PATH)
    with gzip.open(DATA_PATH, "rb") as f:
        return pickle.load(f, encoding="latin1")


def plot_curves(history, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    ax_loss.plot(epochs, [r["loss"] for r in history])
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("CNN: loss per epoch")

    ax_acc.plot(
        epochs, [r["train_acc"] for r in history], label="train (subset)"
    )
    ax_acc.plot(epochs, [r["val_acc"] for r in history], label="val (subset)")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("CNN: accuracy per epoch")
    ax_acc.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    train, val, test = load_mnist()
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    X_train_img = X_train.reshape(-1, 28, 28)
    X_val_img = X_val.reshape(-1, 28, 28)
    X_test_img = X_test.reshape(-1, 28, 28)

    net = SmallCNN(
        image_size=28,
        n_filters=8,
        kernel_size=3,
        n_hidden=128,
        n_classes=10,
        learning_rate=LEARNING_RATE,
        seed=0,
    )
    print(f"CNN parameters: {net.n_params():,}")

    rng = np.random.default_rng(0)
    train_sub = rng.choice(len(X_train_img), MONITOR_SUBSET, replace=False)
    val_sub = rng.choice(len(X_val_img), MONITOR_SUBSET, replace=False)

    history = []
    n = X_train_img.shape[0]
    t_onehot = np.eye(10)[y_train]
    t0 = time.time()
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = order[start:end]
            epoch_loss += net.step(X_train_img[idx], t_onehot[idx])
            n_batches += 1
        record = {
            "epoch": epoch,
            "loss": epoch_loss / n_batches,
            "train_acc": net.accuracy(
                X_train_img[train_sub], y_train[train_sub]
            ),
            "val_acc": net.accuracy(X_val_img[val_sub], y_val[val_sub]),
        }
        history.append(record)
        print(
            f"epoch {epoch}: loss={record['loss']:.4f} "
            f"train_acc(subset)={record['train_acc']:.4f} "
            f"val_acc(subset)={record['val_acc']:.4f}"
        )
    elapsed = time.time() - t0

    train_acc_full = net.accuracy(X_train_img, y_train)
    val_acc_full = net.accuracy(X_val_img, y_val)
    test_acc_full = net.accuracy(X_test_img, y_test)

    print(f"\ntrained in {elapsed:.1f}s over {EPOCHS} epochs")
    print(f"final train acc (full 50000): {train_acc_full:.4f}")
    print(f"final val acc (full 10000):   {val_acc_full:.4f}")
    print(f"final test acc (full 10000):  {test_acc_full:.4f}")

    plot_curves(history, "plots/cnn_training_curves.png")


if __name__ == "__main__":
    main()
