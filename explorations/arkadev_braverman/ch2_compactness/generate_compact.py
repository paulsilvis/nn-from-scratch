"""Synthetic compact-cloud generator -- a control for the real-data
experiment in `digits_experiment.py`.

Reconstructs the generative procedure in Arkadev & Braverman,
Ch. 2 sec. 4 (Figs. 15-16): pick a random seed bitmap for each
class, then generate variants by flipping each cell independently
with a small probability p. The book uses a 20x20 = 400-cell field
and p = 0.1 (sec. 4, paragraph beginning "Compact sets of figures
may be generated...").

This is a *control*, not the main experiment: clouds built this way
are compact by construction, since compactness is exactly what the
generative process enforces. The real question -- whether actual
handwritten digits are compact -- can only be answered by running
the same measurements (see `receptor_space.compactness_report`) on
real data, which is what `digits_experiment.py` does. Comparing the
two tells us how close real handwriting variation is to this
idealized model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


def random_seed_codes(
    n_classes: int, n_bits: int, rng: np.random.Generator
) -> Array:
    """Draw `n_classes` independent random binary codes of length
    `n_bits`, one per class -- the "arbitrary field" seed figures of
    Ch. 2 sec. 4.
    """
    return rng.integers(0, 2, size=(n_classes, n_bits)).astype(np.int8)


def generate_cloud(
    seed: Array, n_variants: int, flip_prob: float, rng: np.random.Generator
) -> Array:
    """Generate `n_variants` noisy copies of `seed`, each cell
    flipped independently with probability `flip_prob`.

    Matches Ch. 2 sec. 4: "new figures are produced in such a way
    that each cell of the new figure is, with small probability
    (say 0.1), coloured differently from that cell in the original
    figure."
    """
    n_bits = seed.shape[0]
    flips = rng.random((n_variants, n_bits)) < flip_prob
    variants = np.tile(seed, (n_variants, 1))
    variants[flips] = 1 - variants[flips]
    return variants.astype(np.int8)


def generate_compact_clouds(
    n_classes: int,
    n_bits: int,
    n_per_class: int,
    flip_prob: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Array, Array]:
    """Generate `n_classes` synthetic compact clouds, each built from
    one random seed code and `n_per_class` noisy variants
    (flip_prob per cell, default 0.1 as in the book).

    Returns (codes, labels): codes has shape
    (n_classes * n_per_class, n_bits); labels has shape
    (n_classes * n_per_class,), values 0..n_classes-1.
    """
    rng = np.random.default_rng(seed)
    seeds = random_seed_codes(n_classes, n_bits, rng)

    all_codes = []
    all_labels = []
    for class_idx, class_seed in enumerate(seeds):
        cloud = generate_cloud(class_seed, n_per_class, flip_prob, rng)
        all_codes.append(cloud)
        all_labels.append(np.full(n_per_class, class_idx, dtype=np.int64))

    return np.vstack(all_codes), np.concatenate(all_labels)
