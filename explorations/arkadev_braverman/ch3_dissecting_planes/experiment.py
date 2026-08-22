#!/usr/bin/env python3
"""Chapter 3's dissecting-planes classifier, tested on real
handwriting, reproducing the shape of the book's own
reliability-vs-training-set-size curves (Ch. 3 sec. 2/4, Tables
XV-XVIII) -- there, "reliability" meant how often the trained
arrangement classified held-out representatives correctly as the
number of training examples per class grew from a handful upward.
The book built those tables from ~200 hand-prepared numeral images
(5 classes, a 60-cell receptor field). Here the same question is
asked of scikit-learn's real `load_digits` corpus (1797 real
handwritten 8x8 = 64-pixel images, 10 classes), holding out a fixed
test set and growing the training set from very small to (almost)
the whole remaining pool, repeating each size with several random
seeds since a single draw of a small training set is noisy.

Three things come out of this:

  1. reliability (held-out accuracy) vs. training-set size, for the
     book's actual base construction (`construction="original"`:
     random +/-1/0-coefficient planes, sec. 2, p. 54) -- the direct
     analogue of Table XV.
  2. the same curve for `construction="bisecting"` (the deterministic
     limit of sec. 4's "improved algorithm") -- for comparison, since
     an earlier session mistakenly used this as the base algorithm.
  3. a small check of sec. 4's "method of parallel variants" (train
     several independent models, combine by majority vote) at one
     training-set size, for both constructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from planes import (  # noqa: E402
    fit_dissecting_planes,
    fit_parallel_variants,
    predict_parallel_variants,
)

MAX_INTENSITY = 16.0
PLOTS_DIR = Path(__file__).parent / "plots"
TEST_SIZE = 300
TRAIN_SIZES = [10, 20, 40, 80, 160, 320, 640, 1000, 1497]
N_TRIALS = 5
PARALLEL_VARIANTS_TRAIN_SIZE = 160
N_PARALLEL_VARIANTS = 7


def load_encoded_digits():
    """Real digits, grayscale-normalized to [0, 1] (the continuous
    receptor space Ch. 2 sec. 3 sketches), which both plane
    constructions handle natively -- nothing about "which side of a
    hyperplane" requires binary codes.
    """
    data = load_digits()
    flat = data.images.reshape(len(data.images), -1)
    codes = flat / MAX_INTENSITY
    return codes, data.target


def run_curve(
    codes: np.ndarray,
    labels: np.ndarray,
    construction: str,
    seed: int = 0,
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
                max_planes=5000,
                construction=construction,
            )
            preds = model.predict(test_codes)
            trial_accs.append(float(np.mean(preds == test_labels)))
            trial_planes.append(model.n_planes)
        acc_means.append(float(np.mean(trial_accs)))
        acc_stds.append(float(np.std(trial_accs)))
        plane_means.append(float(np.mean(trial_planes)))
        plane_stds.append(float(np.std(trial_planes)))
        print(
            f"[{construction}] train_size={size:>5} "
            f"test_acc={acc_means[-1]:.3f}+/-{acc_stds[-1]:.3f} "
            f"n_planes={plane_means[-1]:.1f}+/-{plane_stds[-1]:.1f}"
        )

    return TRAIN_SIZES, acc_means, acc_stds, plane_means, plane_stds


def run_parallel_variants_check(
    codes: np.ndarray, labels: np.ndarray, seed: int = 0
) -> None:
    """Sec. 4's "method of parallel variants" at one training-set
    size: does majority-voting several independent models beat a
    single model, for each construction?
    """
    rng = np.random.default_rng(seed)
    n = len(codes)
    perm = rng.permutation(n)
    test_idx = perm[:TEST_SIZE]
    pool_idx = perm[TEST_SIZE:]
    train_idx = rng.choice(
        pool_idx, size=PARALLEL_VARIANTS_TRAIN_SIZE, replace=False
    )

    test_codes, test_labels = codes[test_idx], labels[test_idx]
    train_codes, train_labels = codes[train_idx], labels[train_idx]

    print(
        f"\nparallel-variants check "
        f"(train_size={PARALLEL_VARIANTS_TRAIN_SIZE}, "
        f"n_variants={N_PARALLEL_VARIANTS}):"
    )
    for construction in ("original", "bisecting"):
        single_rng = np.random.default_rng(seed + 1)
        single_model = fit_dissecting_planes(
            train_codes,
            train_labels,
            rng=single_rng,
            max_planes=5000,
            construction=construction,
        )
        single_acc = float(
            np.mean(single_model.predict(test_codes) == test_labels)
        )

        variants_rng = np.random.default_rng(seed + 2)
        models = fit_parallel_variants(
            train_codes,
            train_labels,
            n_variants=N_PARALLEL_VARIANTS,
            rng=variants_rng,
            construction=construction,
            max_planes=5000,
        )
        ensemble_preds = predict_parallel_variants(models, test_codes)
        ensemble_acc = float(np.mean(ensemble_preds == test_labels))

        print(
            f"  {construction:>10}: single={single_acc:.3f}  "
            f"parallel_variants={ensemble_acc:.3f}"
        )


def plot_curves(
    original_curve: tuple,
    bisecting_curve: tuple,
) -> Path:
    """Two-panel plot: reliability curves (left) and planes-drawn
    curves (right) vs. training-set size, for both constructions.
    Saves to plots/reliability_curve.png and returns the path.
    """
    sizes, o_acc, o_acc_std, o_planes, o_planes_std = original_curve
    _, b_acc, b_acc_std, b_planes, b_planes_std = bisecting_curve

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_left.errorbar(
        sizes,
        o_acc,
        yerr=o_acc_std,
        marker="o",
        color="#2b6cb0",
        label="original (sec. 2, random +/-1/0 planes)",
    )
    ax_left.errorbar(
        sizes,
        b_acc,
        yerr=b_acc_std,
        marker="s",
        color="#c05621",
        label="bisecting (sec. 4, k=0 limit)",
    )
    ax_left.set_xscale("log")
    ax_left.set_xlabel("training set size")
    ax_left.set_ylabel("held-out accuracy")
    ax_left.set_title("Reliability vs. training set size")
    ax_left.set_ylim(0, 1.02)
    ax_left.legend(fontsize=8)

    ax_right.errorbar(
        sizes, o_planes, yerr=o_planes_std, marker="o", color="#2b6cb0"
    )
    ax_right.errorbar(
        sizes, b_planes, yerr=b_planes_std, marker="s", color="#c05621"
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

    original_curve = run_curve(codes, labels, construction="original")
    bisecting_curve = run_curve(codes, labels, construction="bisecting")
    run_parallel_variants_check(codes, labels)

    plot_path = plot_curves(original_curve, bisecting_curve)
    print(f"\nplot saved to: {plot_path}")


if __name__ == "__main__":
    main()
