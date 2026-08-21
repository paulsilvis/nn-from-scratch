"""Chapter 3's dissecting-planes classifier, tested on real
handwriting, reproducing the shape of the book's own
reliability-vs-training-set-size curves (Ch. 3 sec. 2, Tables
XV-XVIII) -- there, "reliability" meant how often the trained
arrangement classified held-out representatives correctly as the
number of training examples per class grew from a handful upward.

The book built those tables from ~12-20 hand-prepared representatives
per class. Here the same question is asked of scikit-learn's real
`load_digits` corpus (1797 real handwritten 8x8 = 64-pixel images),
holding out a fixed test set and growing the training set from very
small to (almost) the whole remaining pool, repeating each size with
several random seeds since a single draw of a small training set is
noisy.

Two curves come out of this:

  1. reliability (held-out accuracy) vs. training-set size -- the
     direct analogue of Tables XV-XVIII.
  2. number of planes the algorithm actually drew vs. training-set
     size -- something the book reports per-run in its tables but
     doesn't plot as a curve; worth looking at here since it's a
     direct measure of how much the arrangement had to fragment
     receptor space to keep opponents apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from planes import fit_dissecting_planes  # noqa: E402

MAX_INTENSITY = 16.0
PLOTS_DIR = Path(__file__).parent / "plots"
TEST_SIZE = 300
TRAIN_SIZES = [10, 20, 40, 80, 160, 320, 640, 1000, 1497]
N_TRIALS = 5


def load_encoded_digits():
    """Real digits, grayscale-normalized to [0, 1] (the continuous
    receptor space Ch. 2 sec. 3 sketches), which is what we use here
    since Ch. 3's dissecting planes generalize naturally to
    continuous coordinates -- unlike Ch. 2's exact bit-flip
    definition, nothing about "which side of a hyperplane" requires
    binary codes.
    """
    data = load_digits()
    flat = data.images.reshape(len(data.images), -1)
    codes = flat / MAX_INTENSITY
    return codes, data.target


def run_curve(
    codes: np.ndarray, labels: np.ndarray, seed: int = 0
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Hold out a fixed test set, then for each training-set size in
    TRAIN_SIZES, fit N_TRIALS models on random draws from the
    remaining pool and record mean +/- std of (test accuracy,
    n_planes).
    """
    rng = np.random.default_rng(seed)
    n = len(codes)
    perm = rng.permutation(n)
    test_idx = perm[:TEST_SIZE]
    pool_idx = perm[TEST_SIZE:]

    test_codes, test_labels = codes[test_idx], labels[test_idx]

    acc_means, acc_stds = [], []
    plane_means, plane_stds = [], []

    for size in TRAIN_SIZES:
        size = min(size, len(pool_idx))
        trial_accs, trial_planes = [], []
        for trial in range(N_TRIALS):
            trial_rng = np.random.default_rng(seed * 1000 + trial)
            train_idx = trial_rng.choice(pool_idx, size=size, replace=False)
            model = fit_dissecting_planes(
                codes[train_idx],
                labels[train_idx],
                rng=trial_rng,
                max_planes=3000,
            )
            preds = model.predict(test_codes)
            trial_accs.append(float(np.mean(preds == test_labels)))
            trial_planes.append(model.n_planes)
        acc_means.append(float(np.mean(trial_accs)))
        acc_stds.append(float(np.std(trial_accs)))
        plane_means.append(float(np.mean(trial_planes)))
        plane_stds.append(float(np.std(trial_planes)))
        print(
            f"train_size={size:>5} "
            f"test_acc={acc_means[-1]:.3f}+/-{acc_stds[-1]:.3f} "
            f"n_planes={plane_means[-1]:.1f}+/-{plane_stds[-1]:.1f}"
        )

    return TRAIN_SIZES, acc_means, acc_stds, plane_means, plane_stds


def plot_curves(
    sizes: List[int],
    acc_means: List[float],
    acc_stds: List[float],
    plane_means: List[float],
    plane_stds: List[float],
) -> Path:
    """Two-panel plot: reliability curve (left, analogue of the
    book's Tables XV-XVIII) and planes-drawn curve (right, our own
    addition) vs. training-set size. Saves to
    plots/reliability_curve.png and returns the path.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_left.errorbar(
        sizes, acc_means, yerr=acc_stds, marker="o", color="#2b6cb0"
    )
    ax_left.set_xscale("log")
    ax_left.set_xlabel("training set size")
    ax_left.set_ylabel("held-out accuracy")
    ax_left.set_title("Reliability vs. training set size")
    ax_left.set_ylim(0, 1.02)

    ax_right.errorbar(
        sizes, plane_means, yerr=plane_stds, marker="o", color="#2f855a"
    )
    ax_right.set_xscale("log")
    ax_right.set_xlabel("training set size")
    ax_right.set_ylabel("planes drawn")
    ax_right.set_title("Arrangement size vs. training set size")

    fig.suptitle(
        "Dissecting-planes classifier on real digits "
        "(Arkadev & Braverman, Ch. 3)"
    )
    fig.tight_layout()

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / "reliability_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    codes, labels = load_encoded_digits()
    sizes, acc_means, acc_stds, plane_means, plane_stds = run_curve(
        codes, labels
    )
    plot_path = plot_curves(
        sizes, acc_means, acc_stds, plane_means, plane_stds
    )
    print(f"\nplot saved to: {plot_path}")


if __name__ == "__main__":
    main()
