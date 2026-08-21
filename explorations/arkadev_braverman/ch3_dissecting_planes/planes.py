"""Dissecting-planes learning, Ch. 3 ("Some Methods of Machine
Learning"), sec. 2 and Tables IV-XIX.

The book's idea: rather than solving for one separating hyperplane
directly, throw down a sequence of hyperplanes in receptor space one
at a time. Each hyperplane assigns every point a sign (+1/-1); after
k planes, every point carries a length-k sign vector, i.e. an address
of the *polyhedron* (cell of the arrangement) it falls into. Learning
stops being needed once every polyhedron that contains training
points contains points of only one class -- classification of a new
point then just means finding which polyhedron it lands in and
reading off that cell's class.

The only real design freedom the book leaves open is *which*
hyperplane to draw next. We use the reading implied by "opponents"
(Ch. 3's term for two points of different classes the current
arrangement still can't tell apart): don't draw a plane speculatively
at random and hope it helps -- only draw one when the data actually
forces it, i.e. when some polyhedron still contains an opponent pair,
and then draw the specific plane that resolves *that* pair (its
perpendicular bisector). This keeps the "random" element (which
opponent pair gets picked, when several exist) while guaranteeing
each new plane makes measurable progress, which matters once real
digit corpora are ~1800 points instead of the book's dozen
hand-picked representatives per class.

Reuses `receptor_space`'s encoding and distance functions rather than
duplicating them; this module adds only what Ch. 3 needs on top of
Ch. 2's receptor space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


def bisecting_hyperplane(p: Array, q: Array) -> Hyperplane:
    """The perpendicular-bisector hyperplane between two points.

    normal = p - q, offset so the midpoint of (p, q) lies exactly on
    the plane; oriented so p reads +1 and q reads -1. This is the
    "opponent-forced" plane: given two points the current
    arrangement can't yet distinguish, it's the natural minimal
    choice that separates exactly that pair.
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


@dataclass
class SignTable:
    """Bookkeeping for the arrangement built so far: the list of
    planes, and each training point's accumulated sign vector (its
    polyhedron address).

    The signs matrix grows one column at a time as planes are added,
    which is why training maintains it incrementally rather than
    recomputing sign vectors from scratch after every new plane --
    that would cost O(n_points * n_planes) work per plane instead of
    O(n_points).
    """

    planes: List[Hyperplane] = field(default_factory=list)
    signs: Optional[Array] = None  # (n_points, n_planes), filled during fit

    def sign_vectors_for(self, points: Array) -> Array:
        """Sign vector of arbitrary (e.g. held-out) points against
        every plane drawn so far."""
        if not self.planes:
            return np.zeros((points.shape[0], 0), dtype=np.int8)
        cols = [plane.sign(points) for plane in self.planes]
        return np.stack(cols, axis=1)


def _find_opponent_pair(
    sign_vecs: Array,
    labels: Array,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    """Return one (i, j) index pair of different-class points that
    currently share a sign vector (an "opponent" pair), or None if
    the arrangement already separates every class.

    Grouping by sign vector rather than comparing all O(n^2) pairs
    keeps this cheap even as n_points grows into the thousands: only
    points that already collide on every plane so far are candidates.
    Point order is shuffled by `rng` before grouping, so which
    opponent pair gets returned (when several contradicting cells
    exist) is genuinely randomized rather than always the
    lowest-index one -- this is the "random" in "random hyperplane
    construction": which contradiction gets resolved next is random,
    even though the plane that resolves it is not.
    """
    order = rng.permutation(sign_vecs.shape[0])
    groups: Dict[tuple, List[int]] = {}
    for i in order:
        sv = tuple(int(v) for v in sign_vecs[i])
        groups.setdefault(sv, []).append(int(i))

    for idx_list in groups.values():
        if len(idx_list) < 2:
            continue
        first_label = int(labels[idx_list[0]])
        for k in idx_list[1:]:
            if int(labels[k]) != first_label:
                return idx_list[0], k
    return None


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
        """Classify new points by which polyhedron they fall into.

        The book's algorithm has no rule for a polyhedron that no
        training point ever landed in (with enough planes and enough
        training data this becomes rare, but with a finite corpus it
        does happen). We fall back to 1-nearest-neighbor among the
        training codes for those points, using Euclidean distance --
        documented here as an addition of ours, not something the
        1966 text specifies.
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
) -> DissectingPlanesModel:
    """Ch. 3 sec. 2's learning algorithm: repeatedly draw a new plane
    only when some polyhedron still contains an "opponent" pair
    (two training points of different classes the arrangement can't
    yet tell apart), until no such pair remains or `max_planes` is
    reached.

    Each new plane is the perpendicular bisector of one randomly
    chosen opponent pair still outstanding -- see module docstring
    for why this reading of "random hyperplane, only when forced" was
    chosen. If `max_planes` is hit first (possible if some opponent
    pair is a genuine duplicate: identical code, different label),
    remaining contradictions are left to majority vote at the cell
    level.
    """
    rng = rng if rng is not None else np.random.default_rng()
    codes = np.asarray(codes, dtype=float)
    labels = np.asarray(labels)

    table = SignTable()
    signs = np.zeros((codes.shape[0], 0), dtype=np.int8)

    while len(table.planes) < max_planes:
        opponents = _find_opponent_pair(signs, labels, rng)
        if opponents is None:
            break
        i, j = opponents
        try:
            plane = bisecting_hyperplane(codes[i], codes[j])
        except ValueError:
            # identical codes, different labels: no plane can ever
            # separate this pair, so stop looking for one and let
            # majority vote settle their shared cell.
            break
        table.planes.append(plane)
        new_col = plane.sign(codes).reshape(-1, 1)
        signs = np.hstack([signs, new_col])

    table.signs = signs
    cell_labels = _majority_labels(signs, labels)
    return DissectingPlanesModel(
        table=table,
        cell_labels=cell_labels,
        train_codes=codes,
        train_labels=labels,
    )
