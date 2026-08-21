"""Receptor-space encoding and Chapter 2 machinery.

Reconstructs the definitions from Arkadev & Braverman, "Computers
and Pattern Recognition" (1966), Chapter 2 ("Encoding Two-Dimensional
Figures. The Notion of a Compact Set"), sections 1-3, plus the
summary list at the end of section 3.

Book definitions used here:

  - A receptor field of n elements maps each figure to a code of n
    binary digits (black = 1, white = 0): a vertex of the unit
    n-dimensional hypercube (sec. 1-2).
  - A point is an INTERNAL point of a set if no single-digit change
    ("step") transfers it to a point of a different set.
  - A point is a BOUNDARY point of a set if at least one single-digit
    change transfers it to a point of a different set.
  - A set is COMPACT if it has few boundary points relative to
    internal points, its internal points are joined by smooth paths
    staying within the set, and most internal points have large
    same-class neighborhoods (sec. 3, summary list, items 2-5).

For continuous (grayscale) receptor spaces the book only sketches an
extension (sec. 3, final paragraph) without a precise internal/
boundary definition, since "one-bit-flip" has no literal meaning once
co-ordinates are continuous. That case is handled here by a
documented nearest-neighbor generalization rather than a literal bit
flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist

Array = np.ndarray


def hamming_distance(a: Array, b: Array) -> int:
    """Number of differing digits between two binary codes.

    This is the book's own notion of distance in receptor space
    (Ch. 2 sec. 3; reused in Ch. 4 sec. 1): the count of positions
    in which two codes disagree.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("codes must have the same shape")
    return int(np.sum(a != b))


def euclidean_distance(a: Array, b: Array) -> float:
    """Euclidean distance between two receptor-space points.

    Used for the continuous (grayscale) receptor space the book
    describes as an extension of the black/white case (Ch. 2 sec. 3,
    final paragraph).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("codes must have the same shape")
    return float(np.sqrt(np.sum((a - b) ** 2)))


def binarize(image: Array, threshold: float) -> Array:
    """Threshold a grayscale receptor field into a binary code.

    Pixels strictly greater than `threshold` become 1 ("black"),
    all others become 0 ("white"), matching the black/white
    encoding used throughout Ch. 2 (sec. 1, Fig. 6).
    """
    image = np.asarray(image, dtype=float)
    return (image > threshold).astype(np.int8)


def normalize_grayscale(image: Array, max_value: float) -> Array:
    """Scale a grayscale receptor field to the continuous [0, 1]
    receptor space.

    Per Ch. 2 sec. 3 (final paragraph): zero corresponds to white,
    one to black, intermediate values to shades of grey.
    """
    image = np.asarray(image, dtype=float)
    return image / max_value


def encode_figures(
    images: Sequence[Array],
    mode: str,
    threshold: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Array:
    """Encode a batch of raw images into receptor-space codes.

    mode='binary'    requires `threshold`; uses `binarize`.
    mode='grayscale' requires `max_value`; uses `normalize_grayscale`.

    Returns an array of shape (n_images, n_receptors): one flat code
    per figure, matching Ch. 2 sec. 1's formula x_1, x_2, ..., x_n,
    one digit per elementary square.
    """
    flat = [np.asarray(im, dtype=float).ravel() for im in images]
    if mode == "binary":
        if threshold is None:
            raise ValueError("mode='binary' requires a threshold")
        return np.stack([binarize(im, threshold) for im in flat])
    if mode == "grayscale":
        if max_value is None:
            raise ValueError("mode='grayscale' requires max_value")
        return np.stack([normalize_grayscale(im, max_value) for im in flat])
    raise ValueError(f"unknown mode: {mode!r}")


def _code_label_index(codes: Array, labels: Array) -> dict:
    """Map each observed binary code to the set of labels seen with
    it, for O(1) neighbor lookups during the bit-flip test.

    Several figures can legitimately share one code once digits are
    reduced to a coarse receptor field (64 cells here), so a code
    may correctly index more than one label.
    """
    index: dict = {}
    for code, label in zip(codes, labels):
        key = tuple(int(v) for v in code)
        index.setdefault(key, set()).add(int(label))
    return index


def is_internal_point_binary(
    point_index: int,
    codes: Array,
    labels: Array,
    code_index: Optional[dict] = None,
) -> bool:
    """Exact Ch. 2 internal-point test for a binary code.

    A point is internal iff flipping any single digit of its code
    never lands on a code belonging to a different label, among the
    codes actually observed in `codes`. This checks the book's
    definition against our finite sample, exactly as its own worked
    examples do (Ch. 2 Table I discussion; Ch. 3's "opponents").
    """
    if code_index is None:
        code_index = _code_label_index(codes, labels)

    own_label = int(labels[point_index])
    code = codes[point_index]
    n_bits = code.shape[0]

    for bit in range(n_bits):
        flipped = code.copy()
        flipped[bit] = 1 - flipped[bit]
        key = tuple(int(v) for v in flipped)
        neighbor_labels = code_index.get(key)
        if neighbor_labels and (neighbor_labels - {own_label}):
            return False
    return True


def boundary_fraction_binary(codes: Array, labels: Array) -> dict:
    """Fraction of boundary points per class, using the exact Ch. 2
    bit-flip definition. Returns {label: fraction}.
    """
    code_index = _code_label_index(codes, labels)
    result: dict = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        boundary = sum(
            not is_internal_point_binary(i, codes, labels, code_index)
            for i in idx
        )
        result[int(label)] = boundary / len(idx)
    return result


def pairwise_distances(codes: Array, metric: str) -> Array:
    """Full pairwise distance matrix for a set of receptor-space
    codes.

    metric='hamming'   -> integer digit-disagreement count (Ch. 2
                           sec. 3), computed on binary codes.
    metric='euclidean' -> Euclidean distance, for binary or
                           grayscale codes (Ch. 2 sec. 3, final
                           paragraph).
    """
    codes = np.asarray(codes, dtype=float)
    if metric == "hamming":
        # cdist's 'hamming' returns a *fraction* of differing
        # positions; rescale to the book's digit-count convention.
        return cdist(codes, codes, metric="hamming") * codes.shape[1]
    if metric == "euclidean":
        return cdist(codes, codes, metric="euclidean")
    raise ValueError(f"unknown metric: {metric!r}")


@dataclass
class CompactnessReport:
    """Per-class compactness summary, generalizing the book's
    internal/boundary/neighborhood criteria (Ch. 2 sec. 3, summary
    list) into numbers we can compute and compare across classes.
    """

    label: int
    n_points: int
    boundary_fraction: float
    mean_nearest_same_class: float
    mean_nearest_other_class: float

    @property
    def same_class_neighbor_margin(self) -> float:
        """How much closer same-class neighbors are than
        other-class neighbors, on average.

        Large and positive matches the book's compactness property;
        near zero or negative indicates a poorly separated set.
        """
        return self.mean_nearest_other_class - self.mean_nearest_same_class


def compactness_report(
    codes: Array, labels: Array, metric: str = "hamming"
) -> list:
    """Per-class compactness report combining:

    - boundary_fraction: the exact Ch. 2 bit-flip definition
      (only meaningful, and only computed, for metric='hamming').
    - mean nearest same-class / other-class distances: a
      continuous-friendly stand-in for "large same-class
      neighborhoods" (sec. 3, items 4-5 of the summary list),
      usable for both binary and grayscale codes.
    """
    dist_matrix = pairwise_distances(codes, metric)
    labels = np.asarray(labels)

    boundary_by_label = (
        boundary_fraction_binary(codes, labels) if metric == "hamming" else {}
    )

    reports = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        other_idx = np.where(labels != label)[0]

        same_nearest = []
        other_nearest = []
        for i in idx:
            same_others = idx[idx != i]
            if len(same_others) > 0:
                same_nearest.append(dist_matrix[i, same_others].min())
            other_nearest.append(dist_matrix[i, other_idx].min())

        reports.append(
            CompactnessReport(
                label=int(label),
                n_points=len(idx),
                boundary_fraction=boundary_by_label.get(
                    int(label), float("nan")
                ),
                mean_nearest_same_class=float(np.mean(same_nearest)),
                mean_nearest_other_class=float(np.mean(other_nearest)),
            )
        )
    return reports
