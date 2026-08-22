#!/usr/bin/env python3
"""Stage 4: real MNIST, comparing 3 hidden-layer sizes.

Data: the standard 50k/10k/10k train/val/test MNIST split, fetched
from Michael Nielsen's neural-networks-and-deep-learning repo (one
of the resources flagged as a good fit at project kickoff) via
raw.githubusercontent.com, one of this sandbox's allowed network
domains. See notes.md for why this is real MNIST and not a
substitute.
"""

import gzip
import pickle
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist_net import MLP  # noqa: E402
from viz.snapshot import Recorder  # noqa: E402
from viz.replay import plot_loss_curve  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "mnist.pkl.gz"
MNIST_URL = (
    "https://raw.githubusercontent.com/mnielsen/"
    "neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
)
HIDDEN_SIZES = [32, 128, 512]
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.5


def load_mnist():
    """Load the standard 50k/10k/10k MNIST split, downloading it
    from Michael Nielsen's repo if not already present locally (the
    data file is gitignored - see notes.md for why).
    """
    if not DATA_PATH.exists():
        print(f"MNIST data not found at {DATA_PATH}, fetching...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, DATA_PATH)
        print("done.")

    with gzip.open(DATA_PATH, "rb") as f:
        train, val, test = pickle.load(f, encoding="latin1")
    return train, val, test


def plot_accuracy_comparison(results, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for hidden, history in results.items():
        epochs = [r["epoch"] for r in history]
        losses = [r["loss"] for r in history]
        val_accs = [r["val_acc"] for r in history]
        ax_loss.plot(epochs, losses, label=f"hidden={hidden}")
        ax_acc.plot(epochs, val_accs, label=f"hidden={hidden}")

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("Loss by hidden layer size")
    ax_loss.legend(fontsize=9)

    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_title("Validation accuracy by hidden layer size")
    ax_acc.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    train, val, test = load_mnist()
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    results = {}
    final_nets = {}

    for hidden in HIDDEN_SIZES:
        net = MLP(
            n_inputs=784,
            n_hidden=hidden,
            n_classes=10,
            learning_rate=LEARNING_RATE,
            seed=0,
        )
        recorder = Recorder()
        t0 = time.time()
        history = net.fit(
            X_train,
            y_train,
            n_classes=10,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            X_val=X_val,
            y_val=y_val,
            seed=0,
            recorder=recorder,
        )
        elapsed = time.time() - t0

        test_acc = net.accuracy(X_test, y_test)
        n_params = net.W1.size + net.b1.size + net.W2.size + net.b2.size
        print(f"--- hidden={hidden} ({n_params} parameters) ---")
        print(f"  trained in {elapsed:.1f}s over {EPOCHS} epochs")
        print(f"  final train acc: {history[-1]['train_acc']:.4f}")
        print(f"  final val acc:   {history[-1]['val_acc']:.4f}")
        print(f"  test acc:        {test_acc:.4f}\n")

        results[hidden] = history
        final_nets[hidden] = net

        plot_loss_curve(
            [r["loss"] for r in history],
            title=f"MNIST (hidden={hidden}): loss per epoch",
            save_path=f"plots/loss_hidden{hidden}.png",
        )

    plot_accuracy_comparison(results, "plots/hidden_size_comparison.png")


if __name__ == "__main__":
    main()
