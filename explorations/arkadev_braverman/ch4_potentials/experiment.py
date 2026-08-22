#!/usr/bin/env python3
"""Chapter 4's potentials-method classifier, tested on real
handwriting -- the direct analogue of the book's own Fig. 42 and
Tables XVII-XVIII experiments, but on scikit-learn's `load_digits`
corpus (1797 real handwritten 8x8 = 64-pixel images, 10 classes)
instead of the book's ~200 hand-prepared numerals.

Three things come out of this, matching the chapter's own structure:

  1. `run_reliability_curve` -- reliability (held-out accuracy) of the
     simple sec. 1 potentials classifier vs. N (training examples per
     class), reproducing the shape of Fig. 42: does it plateau the
     way the book's own curve does past N~13?
  2. `run_improved_comparison` -- sec. 4's iterative-reweighting
     improvement vs. the simple classifier, at two training sizes,
     with a per-class breakdown, reproducing Table XVII's comparison.
  3. `run_receptor_field_comparison` -- sec. 2's receptor-field
     potential encoding as a preprocessing step, vs. plain grayscale
     codes, with a per-class breakdown, reproducing Table XVIII's
     comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from potentials import (  # noqa: E402
    PotentialClassifier,
    encode_figures_potential,
)

MAX_INTENSITY = 16.0
PLOTS_DIR = Path(__file__).parent / "plots"
ALPHA = 1.0  # decay rate for phi(R) = 1/(1+alpha*R^2); book gives no
# specific value, so this is chosen (not tuned) to leave meaningful
# potential mass beyond R=0 for our (grayscale, [0,1]-normalized)
# receptor space -- see notes.md.
TEST_SIZE = 300
N_PER_CLASS_SIZES = [2, 4, 6, 8, 10, 13, 16, 20, 25, 30]
N_TRIALS = 5
TABLE_XVII_NS = [12, 21]  # matching the book's own Table XVII columns
TABLE_XVIII_N = 12  # matching the book's own Table XVIII


def load_encoded_digits():
    """Real digits as 8x8 images (for receptor-field encoding) plus
    their flat grayscale-normalized codes (for the plain classifier),
    matching ch3's [0, 1] continuous receptor space convention.
    """
    data = load_digits()
    images = data.images / MAX_INTENSITY
    flat_codes = images.reshape(len(images), -1)
    return images, flat_codes, data.target


def _sample_n_per_class(
    labels: np.ndarray, pool_idx: np.ndarray, n_per_class: int, rng
) -> np.ndarray:
    """Draw up to `n_per_class` indices per class from `pool_idx`."""
    chosen = []
    for label in np.unique(labels):
        class_pool = pool_idx[labels[pool_idx] == label]
        size = min(n_per_class, len(class_pool))
        chosen.append(rng.choice(class_pool, size=size, replace=False))
    return np.concatenate(chosen)


def run_reliability_curve(
    codes: np.ndarray, labels: np.ndarray, seed: int = 0
) -> Tuple[List[int], List[float], List[float]]:
    """Reliability vs. N-per-class, for the simple (unweighted)
    potentials classifier -- the analogue of Fig. 42.
    """
    rng = np.random.default_rng(seed)
    n = len(codes)
    perm = rng.permutation(n)
    test_idx = perm[:TEST_SIZE]
    pool_idx = perm[TEST_SIZE:]
    test_codes, test_labels = codes[test_idx], labels[test_idx]

    acc_means, acc_stds = [], []
    for n_per_class in N_PER_CLASS_SIZES:
        trial_accs = []
        for trial in range(N_TRIALS):
            trial_rng = np.random.default_rng(seed * 1000 + trial)
            train_idx = _sample_n_per_class(
                labels, pool_idx, n_per_class, trial_rng
            )
            clf = PotentialClassifier(metric="euclidean", alpha=ALPHA).fit(
                codes[train_idx], labels[train_idx]
            )
            preds = clf.predict(test_codes)
            trial_accs.append(float(np.mean(preds == test_labels)))
        acc_means.append(float(np.mean(trial_accs)))
        acc_stds.append(float(np.std(trial_accs)))
        print(
            f"N={n_per_class:>3} reliability="
            f"{acc_means[-1]:.3f}+/-{acc_stds[-1]:.3f}"
        )

    return N_PER_CLASS_SIZES, acc_means, acc_stds


def _per_class_accuracy(
    preds: np.ndarray, labels: np.ndarray, test_labels: np.ndarray
) -> Dict[int, float]:
    result = {}
    for label in np.unique(labels):
        idx = test_labels == label
        result[int(label)] = float(np.mean(preds[idx] == test_labels[idx]))
    return result


def run_improved_comparison(
    codes: np.ndarray, labels: np.ndarray, seed: int = 0
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """Simple vs. improved (reweighted) classifier, at the same N
    values as the book's own Table XVII, with a per-class breakdown.
    """
    rng = np.random.default_rng(seed)
    n = len(codes)
    perm = rng.permutation(n)
    test_idx = perm[:TEST_SIZE]
    pool_idx = perm[TEST_SIZE:]
    test_codes, test_labels = codes[test_idx], labels[test_idx]

    results: Dict[int, Dict[str, Dict[int, float]]] = {}
    for n_per_class in TABLE_XVII_NS:
        train_idx = _sample_n_per_class(labels, pool_idx, n_per_class, rng)
        train_codes, train_labels = codes[train_idx], labels[train_idx]

        simple = PotentialClassifier(metric="euclidean", alpha=ALPHA).fit(
            train_codes, train_labels
        )
        simple_preds = simple.predict(test_codes)
        simple_by_class = _per_class_accuracy(
            simple_preds, labels, test_labels
        )
        simple_overall = float(np.mean(simple_preds == test_labels))

        improved = PotentialClassifier(
            metric="euclidean", alpha=ALPHA
        ).fit_improved(train_codes, train_labels)
        improved_preds = improved.predict(test_codes)
        improved_by_class = _per_class_accuracy(
            improved_preds, labels, test_labels
        )
        improved_overall = float(np.mean(improved_preds == test_labels))

        print(f"\nN={n_per_class} (cycles to converge: {improved.n_cycles_})")
        print(f"{'digit':>5} {'simple':>8} {'improved':>10}")
        for label in sorted(simple_by_class):
            print(
                f"{label:>5} {simple_by_class[label]:>8.3f} "
                f"{improved_by_class[label]:>10.3f}"
            )
        print(f"{'avg':>5} {simple_overall:>8.3f} {improved_overall:>10.3f}")

        results[n_per_class] = {
            "simple": simple_by_class,
            "improved": improved_by_class,
        }
        results[n_per_class]["simple"]["avg"] = simple_overall
        results[n_per_class]["improved"]["avg"] = improved_overall

    return results


def run_receptor_field_comparison(
    images: np.ndarray, labels: np.ndarray, seed: int = 0
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Sec. 2's receptor-field potential encoding vs. its own baseline,
    at the book's own Table XVIII training size (N=12), both
    classified by the simple (unweighted) classifier.

    Run twice, against two different baselines:

    - 'grayscale': the continuous [0,1] codes used everywhere else in
      this reconstruction (ch3's convention) -- already carries
      fine-grained intensity information, unlike the book's own
      black/white-only receptor field.
    - 'binary': a thresholded black/white baseline, matching the
      book's *actual* Table XVIII regime (its "old method of coding"
      was plain 0/1, not continuous grayscale).
    """
    flat_grayscale = images.reshape(len(images), -1)
    binary_images = (images > 0.5).astype(float)  # threshold at 8/16
    flat_binary = binary_images.reshape(len(images), -1)
    potential_from_grayscale = encode_figures_potential(
        images, neighbor_weight=0.25
    )
    potential_from_binary = encode_figures_potential(
        binary_images, neighbor_weight=0.25
    )

    rng = np.random.default_rng(seed)
    n = len(images)
    perm = rng.permutation(n)
    test_idx = perm[:TEST_SIZE]
    pool_idx = perm[TEST_SIZE:]
    train_idx = _sample_n_per_class(labels, pool_idx, TABLE_XVIII_N, rng)
    test_labels = labels[test_idx]

    def _fit_predict(codes: np.ndarray) -> Tuple[Dict[int, float], float]:
        clf = PotentialClassifier(metric="euclidean", alpha=ALPHA).fit(
            codes[train_idx], labels[train_idx]
        )
        preds = clf.predict(codes[test_idx])
        by_class = _per_class_accuracy(preds, labels, test_labels)
        overall = float(np.mean(preds == test_labels))
        return by_class, overall

    results: Dict[str, Dict[str, Dict[int, float]]] = {}
    for baseline_name, (plain_codes, encoded_codes) in {
        "grayscale": (flat_grayscale, potential_from_grayscale),
        "binary": (flat_binary, potential_from_binary),
    }.items():
        plain_by_class, plain_overall = _fit_predict(plain_codes)
        potential_by_class, potential_overall = _fit_predict(encoded_codes)

        print(
            f"\nreceptor-field potential encoding check "
            f"(N={TABLE_XVIII_N}, baseline={baseline_name}):"
        )
        print(f"{'digit':>5} {'plain':>8} {'potential-encoded':>18}")
        for label in sorted(plain_by_class):
            print(
                f"{label:>5} {plain_by_class[label]:>8.3f} "
                f"{potential_by_class[label]:>18.3f}"
            )
        print(f"{'avg':>5} {plain_overall:>8.3f} {potential_overall:>18.3f}")

        results[baseline_name] = {
            "plain": {**plain_by_class, "avg": plain_overall},
            "potential": {**potential_by_class, "avg": potential_overall},
        }

    return results


def plot_reliability_curve(
    sizes: List[int], acc_means: List[float], acc_stds: List[float]
) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(sizes, acc_means, yerr=acc_stds, marker="o", color="#2b6cb0")
    ax.set_xlabel("N (training examples per class)")
    ax.set_ylabel("reliability of recognition")
    ax.set_title("Ch. 4 sec. 1: simple potentials classifier (cf. Fig. 42)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / "reliability_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    images, flat_codes, labels = load_encoded_digits()

    print("=== Reliability vs. N (cf. Fig. 42) ===")
    sizes, acc_means, acc_stds = run_reliability_curve(flat_codes, labels)
    plot_path = plot_reliability_curve(sizes, acc_means, acc_stds)
    print(f"saved {plot_path}")

    print("\n=== Simple vs. improved (reweighted) (cf. Table XVII) ===")
    run_improved_comparison(flat_codes, labels)

    print("\n=== Receptor-field potential encoding (cf. Table XVIII) ===")
    run_receptor_field_comparison(images, labels)
