"""Potentials-method classifier and receptor-field encoding, Ch. 4
("Algorithms Based on Potentials Methods"), secs. 1-2.

Read directly from the book scan the user supplied (this chapter had
not previously been reconstructed from summary/memory).

Book definitions used here:

Sec. 1 -- Potentials in the Receptor Space:

  - Electrostatic analogy: a point-source generates a potential
    phi(R) = 1 / (1 + alpha * R^2), maximal at R=0 and decaying with
    distance R (Euclidean or Hamming -- the book uses either).
  - "Image potential": for a class a with n_a training points, the
    potential of a query point is the *mean* of the individual
    potentials from every training point of that class (sum divided
    by n_a -- explicit in the book's formula, sec. 1, p. 63-64).
  - Classification rule: compute the image potential for every
    class, assign to whichever is largest (two-class case: sign of
    Phi_a - Phi_b).
  - The book's own real-digit experiment (Fig. 42) found this simple
    rule plateaus at ~85% reliability past N~13 training examples per
    class, and diagnosed the plateau as non-uniform point density: a
    dense cluster of one class near a sparse region of another can
    out-vote a query point's true class, and this can misclassify
    even already-seen training points.
  - "Improved algorithm" (Fig. 44's flow chart): after training,
    re-run recognition on the training set itself; for every point
    misclassified, increase its *weight* by one (this doubles that
    point's contribution to its class's mean potential; the
    denominator n_a is explicitly left unchanged per the book).
    Repeat in cycles until all training points are recognized
    correctly (or a cycle cap is hit). Table XVII: this raised
    average reliability from 85.0% to 89.3% at N=21, with the
    largest gains on the worst-performing classes.

Sec. 2 -- Potentials in the Receptor Field:

  - A different problem: raw binary receptor codes are blind to
    *how far* an excited element shifted -- a "5" whose vertical
    stroke shifts one square (still a "5") and one that shifts five
    squares (now a "3") can differ from the original by the same
    Hamming distance, so plain receptor-space distance treats them
    as equally different.
  - Fix: re-encode each figure *before* computing any distance, by
    giving each excited receptor element a potential of 1 at itself,
    spreading a fraction of that potential to elements adjoining it
    (vertically, horizontally, or diagonally -- the book's own
    worked example, Figs. 45/48/49, uses a fraction of 1/4).
    Contributions from all excited elements of a figure are summed
    per receptor-field position, replacing the raw 0/1 code with a
    real-valued, spatially-blurred code.
  - The book's own worked numeric example (Fig. 48, a single excited
    element on an 18-element field, shifted by one and then two
    squares) is not reproduced exactly here: the OCR of the scan
    garbles both the fractional codes and the quoted root values
    (e.g. "x)2" for a square root, "$440004140004" for a code
    string), so the precise numbers cannot be recovered reliably.
    What *is* unambiguous from the prose, and is what's checked in
    notes.md, is the qualitative claim: under plain binary/Hamming
    coding a shift of any size (short of overlap) gives the same
    Euclidean distance, while under this potential encoding distance
    grows with shift size.
  - Table XVIII (N=12, real experiment): layering this encoding under
    the simplest potentials classifier raised average reliability
    from 85.0% to 94.0%, again with the largest gains on the
    previously-worst classes (digit 8: 76.2% -> 100%; digit 9:
    42.0% -> 64.0%).

Reuses `receptor_space`'s distance functions (ch2_compactness) rather
than duplicating them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "ch2_compactness"))
from receptor_space import euclidean_distance, hamming_distance  # noqa: E402

Array = np.ndarray

_DISTANCE_FNS = {
    "euclidean": euclidean_distance,
    "hamming": hamming_distance,
}


def potential(distance: Array, alpha: float) -> Array:
    """Electrostatic-analogy potential phi(R) = 1 / (1 + alpha*R^2)
    (Ch. 4 sec. 1, p. 61): maximal at R=0, decreasing with distance.
    """
    distance = np.asarray(distance, dtype=float)
    return 1.0 / (1.0 + alpha * distance**2)


def _pairwise_potentials(
    query: Array, sources: Array, metric: str, alpha: float
) -> Array:
    """Potential generated at `query` by each row of `sources`."""
    if metric == "euclidean":
        query = np.asarray(query, dtype=float)
        sources = np.asarray(sources, dtype=float)
        diffs = sources - query
        r = np.sqrt(np.sum(diffs**2, axis=1))
    elif metric == "hamming":
        query = np.asarray(query)
        sources = np.asarray(sources)
        r = np.sum(sources != query, axis=1).astype(float)
    else:
        raise ValueError(f"unknown metric: {metric!r}")
    return potential(r, alpha)


@dataclass
class PotentialClassifier:
    """The book's Ch. 4 sec. 1 potentials-method classifier.

    fit() records training codes/labels/weights (learning is just
    memorization, per the book: "the codes of all given points are
    recorded together with ... membership ... in corresponding
    images", p. 63). predict() computes each class's mean weighted
    potential at a query point and returns the argmax class (p. 64).

    fit_improved() implements sec. 4's iterative reweighting
    (Fig. 44): points the current model misclassifies among its own
    training set have their weight increased by one, repeated in
    cycles until all training points are recognized or a cycle cap
    is reached.
    """

    metric: str = "euclidean"
    alpha: float = 1.0

    codes_: Optional[Array] = field(default=None, repr=False)
    labels_: Optional[Array] = field(default=None, repr=False)
    weights_: Optional[Array] = field(default=None, repr=False)
    classes_: Optional[Array] = field(default=None, repr=False)
    n_cycles_: int = 0

    def fit(self, codes: Array, labels: Array) -> "PotentialClassifier":
        self.codes_ = np.asarray(codes)
        self.labels_ = np.asarray(labels)
        self.weights_ = np.ones(len(codes), dtype=float)
        self.classes_ = np.unique(self.labels_)
        self.n_cycles_ = 0
        return self

    def class_potentials(self, queries: Array) -> Dict[int, Array]:
        """Mean weighted image potential of every class at every
        query point. Returns {class_label: array of shape
        (n_queries,)}.

        The denominator is the (fixed) *count* of points in the
        class, not the sum of weights -- the book is explicit that
        reweighting a misclassified point doesn't change n_a in the
        formula, only that point's own contribution (p. 66-67).
        """
        if self.codes_ is None:
            raise RuntimeError("call fit() before class_potentials()")

        queries = np.asarray(
            queries,
            dtype=float if self.metric == "euclidean" else queries.dtype,
        )
        result: Dict[int, Array] = {}
        for label in self.classes_:
            idx = np.where(self.labels_ == label)[0]
            sources = self.codes_[idx]
            weights = self.weights_[idx]
            n_a = len(idx)

            phis = np.stack(
                [
                    _pairwise_potentials(q, sources, self.metric, self.alpha)
                    for q in queries
                ]
            )  # (n_queries, n_a)
            weighted_sum = phis @ weights
            result[int(label)] = weighted_sum / n_a
        return result

    def predict(self, queries: Array) -> Array:
        phis_by_class = self.class_potentials(queries)
        labels = list(phis_by_class.keys())
        stacked = np.stack([phis_by_class[label] for label in labels], axis=1)
        argmax = np.argmax(stacked, axis=1)
        return np.array([labels[i] for i in argmax])

    def fit_improved(
        self, codes: Array, labels: Array, max_cycles: int = 20
    ) -> "PotentialClassifier":
        """Sec. 4's reweighting improvement (Fig. 44): fit, then
        repeatedly re-recognize the training set and bump the weight
        of every currently-misclassified training point by one,
        until a full cycle makes no errors or `max_cycles` is hit.
        """
        self.fit(codes, labels)
        for cycle in range(max_cycles):
            preds = self.predict(self.codes_)
            wrong = preds != self.labels_
            self.n_cycles_ = cycle + 1
            if not wrong.any():
                break
            self.weights_[wrong] += 1.0
        return self


def receptor_field_potential_encode(
    image: Array, neighbor_weight: float = 0.25
) -> Array:
    """Ch. 4 sec. 2's receptor-field potential encoding.

    Each excited (nonzero) element of a 2-D receptor field
    contributes a potential of 1 to itself and `neighbor_weight` to
    every element adjoining it vertically, horizontally, or
    diagonally (8-connectivity), per the book's crude step-like
    approximation (p. 70-71, Fig. 47c). Contributions from all
    excited elements of a figure are summed. Zero elements of the
    input contribute nothing themselves, but can still receive
    spillover from excited neighbors, becoming nonzero in the output
    -- exactly the "blurring" the book intends (Figs. 49).

    Works on binary or already-continuous (grayscale) inputs -- a
    grayscale pixel's own intensity is treated as the strength of its
    contribution, generalizing the book's black/white-only example.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("expected a 2-D receptor field")

    out = image.copy()
    h, w = image.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(image)
            src_y0, src_y1 = max(0, -dy), h - max(0, dy)
            src_x0, src_x1 = max(0, -dx), w - max(0, dx)
            dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
            dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[
                src_y0:src_y1, src_x0:src_x1
            ]
            out += neighbor_weight * shifted
    return out


def encode_figures_potential(
    images: Array, neighbor_weight: float = 0.25
) -> Array:
    """Apply `receptor_field_potential_encode` to a batch of 2-D
    images and flatten each to a receptor-space code, matching
    `receptor_space.encode_figures`'s output shape.
    """
    encoded = [
        receptor_field_potential_encode(im, neighbor_weight) for im in images
    ]
    return np.stack([e.ravel() for e in encoded])
