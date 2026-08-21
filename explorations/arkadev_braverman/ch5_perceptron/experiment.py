"""Chapter 5's Perceptron, tested on real handwriting -- the direct
analogue of the book's own Figs. 53/54/58 (MARK-1 on Roman letters),
but on scikit-learn's `load_digits` corpus (1797 real handwritten
8x8 = 64-pixel images, 10 classes) instead of the book's letters.

Four things come out of this, matching the chapter's own structure:

  1. `run_algorithm_comparison` -- reliability *during training*
     (trailing-window accuracy) for algorithm 1 (unconditional) vs.
     algorithm 2 (error-correcting), on one two-class pair -- the
     direct analogue of Figs. 53/54. The book's claim: algorithm 1
     plateaus around 70% and stalls; algorithm 2 reaches ~100%.
  2. The same comparison repeated on a harder two-class pair, using
     Ch. 2's own compactness finding (digit 8 least compact, 6 most)
     to pick an "easy" and a "hard" case rather than guessing.
  3. `run_ablation_curve` -- held-out reliability vs. number of
     switched-off A-elements on an already-trained Perceptron, the
     analogue of Fig. 58.
  4. `run_multiclass_comparison` -- the one-vs-rest extension
     (perceptron.py's own extension, not book content) on the full
     10-digit problem, reported alongside Ch. 3's and Ch. 4's own
     final numbers for a same-corpus cross-chapter comparison, since
     the book itself claims (sec. 4) that the Perceptron and the
     potentials/dissecting-planes algorithms are close cousins.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from perceptron import (  # noqa: E402
    MulticlassPerceptron,
    PerceptronLayer,
    TwoClassPerceptron,
    build_random_layer,
    evaluate_ablation,
    rolling_reliability,
    train_algorithm_1,
    train_algorithm_2,
    train_multiclass_algorithm_2,
)

MAX_INTENSITY = 16.0
PLOTS_DIR = Path(__file__).parent / "plots"
BINARIZE_THRESHOLD = 0.3  # receptors are excited/unexcited (sec. 1),
# not graded, so grayscale digits are thresholded into 0/1 first.
N_A_ELEMENTS = 256
INPUTS_PER_A_ELEMENT = 16  # book: 20 of 400 (5%); we use 16 of 64 (25%)
# since our receptor field is far smaller -- see notes.md.
THETA = 0.0  # book leaves theta unspecified beyond "common to all
# A-elements"; 0 keeps every A-element's decision boundary passing
# through the origin, an arbitrary but neutral choice.
N_PRESENTATIONS_PER_CLASS = 40
ROLLING_WINDOW = 10
TEST_SIZE = 60  # per class, held out from training presentations
ABLATION_TRAIN_PRESENTATIONS = 60
ABLATION_COUNTS = [0, 32, 64, 96, 128, 160, 192, 224, 240, 250]
N_ABLATION_TRIALS = 5
MULTICLASS_TRAIN_SIZE = 200
MULTICLASS_TEST_SIZE = 300
N_MULTICLASS_TRIALS = 5
MULTICLASS_N_A_ELEMENTS = 512  # book's own MARK-1 count (footnote,
# sec. 1); a single-pass 256-A-element layer proved noticeably weaker
# for 10-way separation -- see notes.md.
MULTICLASS_INPUTS_PER_ELEMENT = 20  # book's own MARK-1 count


def load_encoded_digits():
    """Real digits, thresholded into the book's binary receptor
    convention (Ch. 2 sec. 1: excited=1, unexcited=0) rather than
    the continuous [0,1] space Ch. 3/4 used, since Ch. 5's A-elements
    are explicitly defined over binary x_i.
    """
    data = load_digits()
    flat = data.images.reshape(len(data.images), -1) / MAX_INTENSITY
    codes = (flat > BINARIZE_THRESHOLD).astype(np.int8)
    return codes, data.target


def _class_pool(codes, labels, digit, rng, n):
    idx = np.where(labels == digit)[0]
    chosen = rng.choice(idx, size=n, replace=False)
    return codes[chosen]


def _make_two_class_stream(codes, labels, digit_a, digit_b, n_per_class, rng):
    """Interleave `n_per_class` presentations of each digit in random
    order, matching the book's "objects of each image are presented"
    (sec. 1) rather than block-by-block presentation.
    """
    pool_a = _class_pool(codes, labels, digit_a, rng, n_per_class)
    pool_b = _class_pool(codes, labels, digit_b, rng, n_per_class)
    X = np.concatenate([pool_a, pool_b])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    order = rng.permutation(len(X))
    return X[order], y[order].astype(int)


def _one_algorithm_comparison_trial(
    codes, labels, digit_a, digit_b, seed
) -> Tuple[np.ndarray, np.ndarray]:
    """A single run (one random wiring, one presentation order) of
    algorithm 1 vs. algorithm 2 on one two-class pair, both starting
    from the *same* random wiring so only the update rule differs.
    """
    rng = np.random.default_rng(seed)
    layer = build_random_layer(
        n_receptors=codes.shape[1],
        n_a_elements=N_A_ELEMENTS,
        inputs_per_element=INPUTS_PER_A_ELEMENT,
        theta=THETA,
        rng=rng,
    )
    X, y = _make_two_class_stream(
        codes, labels, digit_a, digit_b, N_PRESENTATIONS_PER_CLASS, rng
    )

    model1 = TwoClassPerceptron.new(
        PerceptronLayer(layer.weights.copy(), layer.theta)
    )
    model2 = TwoClassPerceptron.new(
        PerceptronLayer(layer.weights.copy(), layer.theta)
    )

    correct1 = train_algorithm_1(model1, X, y)
    correct2 = train_algorithm_2(model2, X, y)

    return (
        rolling_reliability(correct1, ROLLING_WINDOW),
        rolling_reliability(correct2, ROLLING_WINDOW),
    )


def run_algorithm_comparison(
    codes: np.ndarray,
    labels: np.ndarray,
    digit_a: int,
    digit_b: int,
    n_trials: int = N_ABLATION_TRIALS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reliability-during-training for algorithm 1 vs. algorithm 2 on
    one two-class pair, averaged over `n_trials` independent random
    wirings and presentation orders -- the direct analogue of Figs.
    53/54. A single trial turns out to be too noisy to compare fairly
    for harder digit pairs (a handful of unlucky early presentations
    can dominate a 40-per-class run), so results here are averaged
    rather than read off one seed -- see notes.md.
    """
    curves1, curves2 = [], []
    for trial in range(n_trials):
        r1, r2 = _one_algorithm_comparison_trial(
            codes, labels, digit_a, digit_b, seed=trial
        )
        curves1.append(r1)
        curves2.append(r2)
    return np.mean(curves1, axis=0), np.mean(curves2, axis=0)


def run_ablation_curve(
    codes: np.ndarray, labels: np.ndarray, digit_a: int, digit_b: int
) -> Tuple[List[int], List[float], List[float]]:
    """Held-out reliability vs. number of switched-off A-elements on
    an already-trained (algorithm 2) Perceptron -- the analogue of
    Fig. 58.
    """
    rng = np.random.default_rng(1)
    n_train = ABLATION_TRAIN_PRESENTATIONS

    idx_a = np.where(labels == digit_a)[0]
    idx_b = np.where(labels == digit_b)[0]
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)

    test_a_end = n_train + TEST_SIZE
    train_a, test_a = idx_a[:n_train], idx_a[n_train:test_a_end]
    train_b, test_b = idx_b[:n_train], idx_b[n_train:test_a_end]

    X_train = np.concatenate([codes[train_a], codes[train_b]])
    y_train = np.concatenate(
        [np.zeros(len(train_a)), np.ones(len(train_b))]
    ).astype(int)
    order = rng.permutation(len(X_train))
    X_train, y_train = X_train[order], y_train[order]

    X_test = np.concatenate([codes[test_a], codes[test_b]])
    y_test = np.concatenate(
        [np.zeros(len(test_a)), np.ones(len(test_b))]
    ).astype(int)

    layer = build_random_layer(
        n_receptors=codes.shape[1],
        n_a_elements=N_A_ELEMENTS,
        inputs_per_element=INPUTS_PER_A_ELEMENT,
        theta=THETA,
        rng=rng,
    )
    model = TwoClassPerceptron.new(layer)
    train_algorithm_2(model, X_train, y_train)

    means, stds = [], []
    for n_off in ABLATION_COUNTS:
        trials = [
            evaluate_ablation(
                model, X_test, y_test, n_off, np.random.default_rng(t)
            )
            for t in range(N_ABLATION_TRIALS)
        ]
        means.append(float(np.mean(trials)))
        stds.append(float(np.std(trials)))
    return ABLATION_COUNTS, means, stds


def run_multiclass_comparison(
    codes: np.ndarray, labels: np.ndarray, n_epochs: int = 1
):
    """Full 10-digit one-vs-rest Perceptron, held-out accuracy over
    several trials -- our own multiclass extension, reported for
    cross-chapter comparison against Ch. 3/Ch. 4's own final numbers
    (see notes.md), since the book itself (sec. 4) claims the
    Perceptron and the potentials/dissecting-planes algorithms are
    close cousins.

    `n_epochs` controls how many passes over the same training set
    are made (each in a freshly-shuffled order). The book's own
    description ("training ... is performed in a sequence of steps"
    at which "an object from one of the images" is shown) never
    explicitly limits training to a single pass through the data, so
    repeated passes are a legitimate reading of the procedure, not a
    departure from it -- and single-pass training turns out to
    undersell the 10-way case badly (see notes.md).
    """
    classes = np.unique(labels)
    accs = []
    for trial in range(N_MULTICLASS_TRIALS):
        rng = np.random.default_rng(100 + trial)
        n = len(codes)
        perm = rng.permutation(n)
        test_idx = perm[:MULTICLASS_TEST_SIZE]
        pool_idx = perm[MULTICLASS_TEST_SIZE:]

        train_idx = []
        for c in classes:
            class_pool = pool_idx[labels[pool_idx] == c]
            size = min(MULTICLASS_TRAIN_SIZE, len(class_pool))
            train_idx.append(rng.choice(class_pool, size=size, replace=False))
        train_idx = np.concatenate(train_idx)

        layer = build_random_layer(
            n_receptors=codes.shape[1],
            n_a_elements=MULTICLASS_N_A_ELEMENTS,
            inputs_per_element=MULTICLASS_INPUTS_PER_ELEMENT,
            theta=THETA,
            rng=rng,
        )
        model = MulticlassPerceptron.new(layer, classes)
        for _ in range(n_epochs):
            order = rng.permutation(len(train_idx))
            train_multiclass_algorithm_2(
                model, codes[train_idx][order], labels[train_idx][order]
            )

        preds = model.predict_batch(codes[test_idx])
        accs.append(float(np.mean(preds == labels[test_idx])))
    return float(np.mean(accs)), float(np.std(accs))


def make_algorithm_comparison_plot(
    reliability1, reliability2, digit_a, digit_b, filename
):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    steps = np.arange(1, len(reliability1) + 1)
    ax.plot(steps, reliability1 * 100, label="Algorithm 1 (unconditional)")
    ax.plot(steps, reliability2 * 100, label="Algorithm 2 (error-correcting)")
    ax.set_xlabel("Presentation number")
    ax.set_ylabel(f"Reliability, trailing {ROLLING_WINDOW} (%)")
    ax.set_title(
        f"Digit {digit_a} vs. {digit_b}: algorithm 1 vs. 2 "
        "(cf. book Figs. 53/54)"
    )
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def make_ablation_plot(counts, means, stds, digit_a, digit_b, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    means_pct = np.array(means) * 100
    stds_pct = np.array(stds) * 100
    ax.errorbar(counts, means_pct, yerr=stds_pct, marker="o", capsize=3)
    ax.set_xlabel(f"No. of switched-off A-elements (of {N_A_ELEMENTS})")
    ax.set_ylabel("Held-out reliability (%)")
    ax.set_title(
        f"Digit {digit_a} vs. {digit_b}: robustness to ablation "
        "(cf. book Fig. 58)"
    )
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def _final_reliability(reliability: np.ndarray, tail: int = 20) -> float:
    """Mean of the last `tail` points of an (already trial-averaged)
    reliability curve, as a less noisy stand-in for "the asymptote"
    than the single last point -- see notes.md.
    """
    return float(reliability[-tail:].mean())


def main():
    codes, labels = load_encoded_digits()

    print(
        "=== Algorithm 1 vs. 2: easy pair (digits 0 vs. 1), "
        f"{N_ABLATION_TRIALS}-trial average ==="
    )
    rel1, rel2 = run_algorithm_comparison(codes, labels, 0, 1)
    final1 = _final_reliability(rel1) * 100
    final2 = _final_reliability(rel2) * 100
    print(f"Algorithm 1 final reliability: {final1:.1f}%")
    print(f"Algorithm 2 final reliability: {final2:.1f}%")
    make_algorithm_comparison_plot(rel1, rel2, 0, 1, "algo_comparison_0_1.png")

    print()
    print(
        "=== Algorithm 1 vs. 2: hard pair (digits 8 vs. 9), "
        f"per Ch.2's compactness ranking, {N_ABLATION_TRIALS}-trial "
        "average ==="
    )
    rel1h, rel2h = run_algorithm_comparison(codes, labels, 8, 9)
    final1h = _final_reliability(rel1h) * 100
    final2h = _final_reliability(rel2h) * 100
    print(f"Algorithm 1 final reliability: {final1h:.1f}%")
    print(f"Algorithm 2 final reliability: {final2h:.1f}%")
    make_algorithm_comparison_plot(
        rel1h, rel2h, 8, 9, "algo_comparison_8_9.png"
    )

    print()
    print("=== Ablation (Fig. 58 analogue): digits 8 vs. 9 ===")
    counts, means, stds = run_ablation_curve(codes, labels, 8, 9)
    for c, m, s in zip(counts, means, stds):
        print(f"  {c:4d} switched off: {m * 100:5.1f}% +/- {s * 100:.1f}%")
    make_ablation_plot(counts, means, stds, 8, 9, "ablation_8_9.png")

    print()
    print(
        "=== Multiclass (one-vs-rest extension), all 10 digits, "
        "single pass ==="
    )
    mean_acc, std_acc = run_multiclass_comparison(codes, labels, n_epochs=1)
    print(f"Held-out accuracy: {mean_acc * 100:.1f}% +/- {std_acc * 100:.1f}%")

    print()
    print(
        "=== Multiclass (one-vs-rest extension), all 10 digits, "
        "5 passes ==="
    )
    mean_acc5, std_acc5 = run_multiclass_comparison(codes, labels, n_epochs=5)
    print(
        f"Held-out accuracy: {mean_acc5 * 100:.1f}% +/- {std_acc5 * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
