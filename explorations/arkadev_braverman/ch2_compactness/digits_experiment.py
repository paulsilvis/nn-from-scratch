"""Chapter 2's compactness hypothesis, tested on real handwriting.

Loads scikit-learn's bundled `load_digits` corpus (1797 real
handwritten 8x8 = 64-pixel digit images -- close in scale to the
book's own 60-cell examples) and asks the actual question Chapter 2
poses: do real digit classes form compact sets in receptor space?

Runs the measurement twice, "in succession" as requested:

  1. Binarized -- true to Ch. 2's exact black/white formalism
     (sec. 1, Fig. 6): each pixel becomes 0 or 1 via a threshold.
  2. Grayscale -- true to the data itself, using the continuous
     receptor space Ch. 2 sec. 3 sketches as an extension.

A synthetic control (`generate_compact.generate_compact_clouds`) is
included for comparison: those clouds are compact by construction,
so they show what "maximally compact" looks like by our own
measurements, as a reference point for judging the real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

from generate_compact import generate_compact_clouds  # noqa: E402
from receptor_space import (  # noqa: E402
    CompactnessReport,
    compactness_report,
)

BINARIZE_THRESHOLD = 8.0  # midpoint of the 0-16 intensity range
MAX_INTENSITY = 16.0
PLOTS_DIR = Path(__file__).parent / "plots"


def load_real_digits():
    """Return (images, labels) for the full sklearn digits corpus."""
    data = load_digits()
    return data.images, data.target


def print_report(title: str, reports: List[CompactnessReport]) -> None:
    print(f"\n{title}")
    print(
        f"{'digit':>5} {'n':>5} {'boundary_frac':>14} "
        f"{'same_nn':>9} {'other_nn':>9} {'margin':>9}"
    )
    for r in sorted(reports, key=lambda x: x.label):
        print(
            f"{r.label:>5} {r.n_points:>5} "
            f"{r.boundary_fraction:>14.3f} "
            f"{r.mean_nearest_same_class:>9.3f} "
            f"{r.mean_nearest_other_class:>9.3f} "
            f"{r.same_class_neighbor_margin:>9.3f}"
        )
    mean_margin = float(
        np.mean([r.same_class_neighbor_margin for r in reports])
    )
    print(f"\nmean same_class_neighbor_margin: {mean_margin:.3f}")


def plot_margins(
    real_binary: List[CompactnessReport],
    real_gray: List[CompactnessReport],
    synthetic: List[CompactnessReport],
) -> Path:
    """Bar chart of same_class_neighbor_margin per digit: real
    (binarized) vs. synthetic control on the left (both Hamming
    distance, directly comparable), real (grayscale) alone on the
    right (Euclidean distance, a different scale).

    Saves to plots/compactness_margins.png and returns the path.
    """
    real_binary = sorted(real_binary, key=lambda r: r.label)
    real_gray = sorted(real_gray, key=lambda r: r.label)
    synthetic = sorted(synthetic, key=lambda r: r.label)
    digits = [r.label for r in real_binary]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    width = 0.35
    x = np.arange(len(digits))
    ax_left.bar(
        x - width / 2,
        [r.same_class_neighbor_margin for r in real_binary],
        width,
        label="real digits (binarized)",
        color="#2b6cb0",
    )
    ax_left.bar(
        x + width / 2,
        [r.same_class_neighbor_margin for r in synthetic],
        width,
        label="synthetic control",
        color="#a0aec0",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(digits)
    ax_left.set_xlabel("digit class")
    ax_left.set_ylabel("same-class neighbor margin (Hamming)")
    ax_left.set_title("Binarized: real vs. synthetic control")
    ax_left.legend()

    ax_right.bar(
        x,
        [r.same_class_neighbor_margin for r in real_gray],
        color="#2f855a",
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(digits)
    ax_right.set_xlabel("digit class")
    ax_right.set_ylabel("same-class neighbor margin (Euclidean)")
    ax_right.set_title("Real digits, grayscale")

    fig.suptitle(
        "Compactness margin per digit class " "(Arkadev & Braverman, Ch. 2)"
    )
    fig.tight_layout()

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / "compactness_margins.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    images, labels = load_real_digits()
    flat = images.reshape(len(images), -1)

    # --- 1. Binarized, true to Ch. 2's exact formalism ---
    binary_codes = (flat > BINARIZE_THRESHOLD).astype(np.int8)
    binary_reports = compactness_report(binary_codes, labels, metric="hamming")
    print_report(
        f"REAL DIGITS -- binarized at threshold={BINARIZE_THRESHOLD} "
        "(Hamming distance)",
        binary_reports,
    )

    # --- 2. Grayscale, true to the data ---
    grayscale_codes = flat / MAX_INTENSITY
    grayscale_reports = compactness_report(
        grayscale_codes, labels, metric="euclidean"
    )
    print_report(
        "REAL DIGITS -- grayscale, normalized to [0, 1] "
        "(Euclidean distance)",
        grayscale_reports,
    )

    # --- 3. Synthetic control: compact by construction ---
    synth_codes, synth_labels = generate_compact_clouds(
        n_classes=10,
        n_bits=64,
        n_per_class=120,
        flip_prob=0.1,
        seed=0,
    )
    synth_reports = compactness_report(
        synth_codes, synth_labels, metric="hamming"
    )
    print_report(
        "SYNTHETIC CONTROL -- Ch. 2 sec. 4 cloud generator "
        "(compact by construction)",
        synth_reports,
    )

    plot_path = plot_margins(binary_reports, grayscale_reports, synth_reports)
    print(f"\nplot saved to: {plot_path}")


if __name__ == "__main__":
    main()
