"""Dissecting-planes learning, Ch. 3 ("Dissecting Planes Algorithm"),
secs. 1-2 (geometry + Tables IV-IX) and sec. 4 (Tables XV-XVIII).

Checked against the book's own text (the scan the user supplied)
after an earlier session had to guess at some details. Two things
that earlier guess got wrong, corrected here:

1. The book's *base* algorithm (sec. 2, p. 54) draws each new plane
   with genuinely random coefficients A_1..A_n, each drawn from the
   set {-1, 0, +1}, plus a random threshold chosen between the two
   opponents' resulting sums. It is NOT the perpendicular bisector
   of the opponent pair -- that would make every run of the
   algorithm identical, which the book explicitly says would defeat
   the "parallel variants" reliability trick (sec. 4 footnote, p.
   56). `random_separating_hyperplane` implements this correctly.

2. The perpendicular-bisector construction this module originally
   used turns out to be a real thing in the book after all -- just a
   *different*, later variant: sec. 4's "improved algorithm" draws
   planes within some small angle k of the perpendicular bisector,
   and the book notes that k=0 (the literal bisector, no randomness
   left at all) is the deterministic limiting case. Kept here as
   `bisecting_hyperplane`, now labeled correctly. The general k>0
   angle-constrained case is not implemented (open question, see
   notes.md).

Also implements the book's own reconstruction workflow faithfully:
points are presented one at a time in random order (Fig. 32's flow
chart), each new point's sign vector is compared only against
previously-seen points, and a new plane is drawn only when an
"opponent" (a previously-seen point of a different class sharing the
new point's current sign vector) is found -- looping until the new
point has no opponent left, exactly as Fig. 32 describes and as
Table VII/VIII's point-6 example (two simultaneous opponents, two
planes drawn in response) requires.

Also implements sec. 4's "method of parallel variants": training the
same data several independent times and combining predictions by
majority vote, which the book reports as a further reliability boost
(pp. 55-58) independent of which single-plane construction is used.

Reuses `receptor_space`'s encoding and distance functions rather than
duplicating them; this module adds only what Ch. 3 needs on top of
Ch. 2's receptor space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

Array = np.ndarray


@dataclass
class Hyperplane:
    """A single dissecting hyperplane w . x + b = 0 in receptor
    space (Ch. 3 sec. 2)."""

    normal: Array
    bias: float

    def sign(self, points: Array) -> Array:
        """+1/-1 sign of each point w.r.t. this hyperplane.

        Points exactly on the plane (w.x + b == 0) are broken toward
        +1 rather than left undefined -- the book's polyhedra are
        closed half-spaces, not open ones, so every point must land
        in exactly one cell.
        """
        values = points @ self.normal + self.bias
        return np.where(values >= 0, 1, -1).astype(np.int8)


def random_separating_hyperplane(
    p: Array,
    q: Array,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> Hyperplane:
    """The book's actual sec. 2 construction (p. 41-42, p. 54): draw
    random coefficients A_1..A_n, one per receptor dimension, from
    the set {-1, 0, +1} (p. 54: "coefficients ... were chosen from
    the set -1, 0, +1"); compute sigma_p = A.p and sigma_q = A.q;
    then choose a threshold uniformly between them. The plane
    A.x - threshold = 0 separates p and q by construction, whatever
    A turns out to be, as long as sigma_p != sigma_q.

    A random A can land exactly perpendicular to (p - q) (sigma_p ==
    sigma_q), which the book doesn't address explicitly; we just
    redraw, up to `max_attempts` times, since this is a
    measure-zero-in-spirit event with a ternary coefficient set that
    still occurs occasionally in low dimensions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n = p.shape[0]

    for _ in range(max_attempts):
        coeffs = rng.choice([-1.0, 0.0, 1.0], size=n)
        sigma_p = float(coeffs @ p)
        sigma_q = float(coeffs @ q)
        if sigma_p != sigma_q:
            break
    else:
        raise ValueError(
            "could not find a separating random direction for this "
            "pair after max_attempts tries (they may be identical "
            "or degenerate in every coordinate that matters)"
        )

    lo, hi = sorted((sigma_p, sigma_q))
    threshold = rng.uniform(lo, hi)
    plane = Hyperplane(normal=coeffs, bias=-threshold)
    if plane.sign(p[None, :])[0] < 0:
        plane = Hyperplane(normal=-coeffs, bias=threshold)
    return plane


def bisecting_hyperplane(p: Array, q: Array) -> Hyperplane:
    """The perpendicular-bisector hyperplane between two points.

    normal = p - q, offset so the midpoint of (p, q) lies exactly on
    the plane; oriented so p reads +1 and q reads -1.

    This is the deterministic (k=0) limit of Ch. 3 sec. 4's
    "improved algorithm": the book proposes drawing planes close to
    the perpendicular bisector of an object and its opponent to make
    the separating surface hug the true class border more closely,
    but notes that taking literally *the* bisector (no random angle
    left at all) makes learning fully deterministic and removes the
    benefit of running parallel variants (sec. 4, footnote, p. 56).
    Kept here as the extreme case of that family, not the book's
    default algorithm.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    normal = p - q
    if not np.any(normal):
        raise ValueError(
            "cannot bisect two identical points (opponents with "
            "identical codes but different labels; no hyperplane "
            "can separate them)"
        )
    midpoint = (p + q) / 2.0
    bias = -float(normal @ midpoint)
    plane = Hyperplane(normal=normal, bias=bias)
    if plane.sign(p[None, :])[0] < 0:
        plane = Hyperplane(normal=-normal, bias=-bias)
    return plane


def _majority_labels(sign_vecs: Array, labels: Array) -> Dict[tuple, int]:
    """Majority-vote class for every distinct sign vector (polyhedron
    cell) that appears among the training points."""
    groups: Dict[tuple, List[int]] = {}
    for sv, lbl in zip(map(tuple, sign_vecs.tolist()), labels):
        groups.setdefault(sv, []).append(int(lbl))
    result = {}
    for sv, lbls in groups.items():
        values, counts = np.unique(lbls, return_counts=True)
        result[sv] = int(values[np.argmax(counts)])
    return result


@dataclass
class SignTable:
    """The trained arrangement: every plane drawn, in the order
    drawn, and the final sign vector each training point ended up
    with (Ch. 3 sec. 2's "sign table", Tables IV-IX)."""

    planes: List[Hyperplane] = field(default_factory=list)
    signs: Optional[Array] = None  # (n_points, n_planes), filled after fit

    def sign_vectors_for(self, points: Array) -> Array:
        """Sign vector of arbitrary (e.g. held-out) points against
        every plane drawn so far."""
        if not self.planes:
            return np.zeros((points.shape[0], 0), dtype=np.int8)
        cols = [plane.sign(points) for plane in self.planes]
        return np.stack(cols, axis=1)


@dataclass
class DissectingPlanesModel:
    """A fitted arrangement of dissecting planes plus the per-cell
    class vote, ready to classify new points (Ch. 3 sec. 2)."""

    table: SignTable
    cell_labels: Dict[tuple, int]
    train_codes: Array
    train_labels: Array

    @property
    def n_planes(self) -> int:
        return len(self.table.planes)

    def predict(self, points: Array) -> Array:
        """Classify new points by which polyhedron they fall into
        (Ch. 3 sec. 2, "Recognition of New Objects").

        The book's algorithm has no rule for a polyhedron that no
        training point ever landed in (with enough planes and
        enough training data this becomes rare, but with a finite
        corpus it does happen). We fall back to 1-nearest-neighbor
        among the training codes for those points, using Euclidean
        distance -- documented here as an addition of ours, not
        something the 1966 text specifies.
        """
        points = np.asarray(points, dtype=float)
        sign_vecs = self.table.sign_vectors_for(points)
        predictions = np.empty(points.shape[0], dtype=int)
        unseen_mask = np.zeros(points.shape[0], dtype=bool)

        for i, sv in enumerate(map(tuple, sign_vecs.tolist())):
            label = self.cell_labels.get(sv)
            if label is None:
                unseen_mask[i] = True
            else:
                predictions[i] = label

        if np.any(unseen_mask):
            diffs = points[unseen_mask, None, :] - self.train_codes[None, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            nearest = np.argmin(dists, axis=1)
            predictions[unseen_mask] = self.train_labels[nearest]

        return predictions


def fit_dissecting_planes(
    codes: Array,
    labels: Array,
    rng: Optional[np.random.Generator] = None,
    max_planes: int = 2000,
    construction: str = "original",
) -> DissectingPlanesModel:
    """Ch. 3 sec. 2's learning algorithm, reconstructed to match
    Fig. 32's flow chart: points are presented one at a time (in
    random order here, since the book doesn't specify an order and
    "the material used for training was always the same" across its
    own six variants -- p. 54 -- implying order/randomness of the
    planes is what varied between variants, not the data). Each new
    point's sign vector is compared only against previously-seen
    points; when it matches an existing point of a different class
    (an "opponent"), a new plane separating the two is drawn and the
    search repeats for the same new point, since one new point can
    have several simultaneous opponents (Table VII/VIII's point 6,
    resolved by two planes in a row).

    `construction` selects how each forced plane is built:

      "original"  -- `random_separating_hyperplane` (sec. 2, p. 54):
                     random +/-1/0 coefficients, random threshold.
                     This is the book's actual base algorithm and
                     matches Table XV's setup.
      "bisecting" -- `bisecting_hyperplane`: the deterministic k=0
                     limit of sec. 4's "improved algorithm" (see that
                     function's docstring). Included for comparison,
                     not as the default reconstruction.

    Returns once every training point is free of opponents, or
    `max_planes` is reached (possible if two points share an
    identical code but different labels, which no hyperplane can
    resolve -- their shared cell is then settled by majority vote).
    """
    if construction not in ("original", "bisecting"):
        raise ValueError(f"unknown construction: {construction!r}")

    rng = rng if rng is not None else np.random.default_rng()
    codes = np.asarray(codes, dtype=float)
    labels = np.asarray(labels)
    n = codes.shape[0]

    planes: List[Hyperplane] = []
    order = rng.permutation(n)
    added_order: List[int] = []
    rows: Dict[int, List[int]] = {}

    def make_plane(p: Array, q: Array) -> Hyperplane:
        if construction == "bisecting":
            return bisecting_hyperplane(p, q)
        return random_separating_hyperplane(p, q, rng)

    def sign_of(plane: Hyperplane, idx: int) -> int:
        point = codes[idx].reshape(1, -1)
        return int(plane.sign(point)[0])

    for idx in order:
        idx = int(idx)
        row = [sign_of(plane, idx) for plane in planes]

        while len(planes) < max_planes:
            opponent_idx = None
            for prev_idx in added_order:
                if rows[prev_idx] == row and labels[prev_idx] != labels[idx]:
                    opponent_idx = prev_idx
                    break
            if opponent_idx is None:
                break
            try:
                plane = make_plane(codes[idx], codes[opponent_idx])
            except ValueError:
                # identical codes, different labels -- no plane can
                # ever separate this pair; leave it to majority vote
                # and stop chasing this contradiction.
                break
            planes.append(plane)
            for prev_idx in added_order:
                rows[prev_idx].append(sign_of(plane, prev_idx))
            row.append(sign_of(plane, idx))

        rows[idx] = row
        added_order.append(idx)

    table = SignTable(planes=planes)
    signs = np.array([rows[i] for i in range(n)], dtype=np.int8).reshape(
        n, len(planes)
    )
    table.signs = signs
    cell_labels = _majority_labels(signs, labels)
    return DissectingPlanesModel(
        table=table,
        cell_labels=cell_labels,
        train_codes=codes,
        train_labels=labels,
    )


def fit_parallel_variants(
    codes: Array,
    labels: Array,
    n_variants: int,
    rng: Optional[np.random.Generator] = None,
    construction: str = "original",
    max_planes: int = 2000,
) -> List[DissectingPlanesModel]:
    """Sec. 4's "method of parallel variants" (pp. 55-58): train the
    same data `n_variants` times independently (different random
    plane choices and, here, different presentation orders each
    time), to be combined by majority vote at prediction time via
    `predict_parallel_variants`. The book reports this as a
    reliability boost on top of either plane-construction method,
    since independent variants tend to err on different points.
    """
    rng = rng if rng is not None else np.random.default_rng()
    models = []
    for _ in range(n_variants):
        variant_rng = np.random.default_rng(rng.integers(2**32))
        models.append(
            fit_dissecting_planes(
                codes,
                labels,
                rng=variant_rng,
                max_planes=max_planes,
                construction=construction,
            )
        )
    return models


def predict_parallel_variants(
    models: Sequence[DissectingPlanesModel], points: Array
) -> Array:
    """Combine several variants' predictions by majority vote per
    point (sec. 4, "the object is attributed to that image to which
    it has been attributed by a majority of the machines")."""
    votes = np.stack([m.predict(points) for m in models], axis=1)
    predictions = np.empty(votes.shape[0], dtype=int)
    for i in range(votes.shape[0]):
        values, counts = np.unique(votes[i], return_counts=True)
        predictions[i] = values[np.argmax(counts)]
    return predictions
