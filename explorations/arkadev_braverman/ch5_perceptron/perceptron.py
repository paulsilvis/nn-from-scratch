"""Perceptron classifier and training algorithms, Ch. 5 ("Algorithm
of the Perceptron"), secs. 1-4.

Read directly from the book scan the user supplied.

Book structure used here:

Sec. 1 -- Structure and Algorithm of the Perceptron:

  - Receptors: n elements, each x_i in {0, 1} (excited/unexcited).
  - A-elements: m elements. Before any experiment, each A-element is
    wired to a *fixed, random* subset of receptors, each connection
    independently signed +1 or -1 (the book's r_ij in {+1, -1, 0},
    0 meaning "not connected"). Footnote: MARK-1 itself used n=400
    receptors, m=512 A-elements, 20 connections per A-element -- a
    small fraction of all receptors, not all of them. This is the
    one respect in which Ch. 5's planes differ from Ch. 3's (there,
    every dissecting plane used all n coordinates).
  - A-element output: y_j = 1 if sum_i r_ij*x_i - theta >= 0, else 0.
    theta is a single scalar, shared by every A-element ("common to
    all A-elements").
  - R-element: N = sum_j A_j*y_j; output (the Perceptron's overall
    response) R = 1 if N >= 0, else 0. The A_j are the *only*
    trainable parameters; the receptor<->A-element wiring is fixed
    for the whole experiment. NOTE ON NOTATION: the book's actual
    symbol for this per-A-element amplifier coefficient is lambda_j,
    not A_j -- the extracted OCR text renders lambda as "A"
    throughout this passage, and earlier work on this module
    (including this variable's name, `a_weights`) followed that
    garbled rendering before the page image was checked directly.
    Left as `a_weights` here to avoid a disruptive rename of working
    code, but doc comments below use the book's real symbol.
  - Two training algorithms, both restricted to a two-image problem
    (image a vs. image b), presented one example at a time:
      Algorithm 1 (unconditional): on every step, A_j (lambda_j) of
        every *excited* A-element is nudged in the same fixed
        direction determined solely by which image was shown,
        whether or not the current response was already correct.
      Algorithm 2 (error-correcting, the classic perceptron rule):
        A_j (lambda_j) changes only on steps where the current
        response was *wrong*, nudging excited A-elements' weights in
        the direction that would have produced the correct output.
    For the plain two-image machine of Fig. 51 that this module
    implements, the direction is symmetric (+ for one image, - for
    the other), matching the sec. 1 prose and the worked example of
    sec. 3. The Fig. 52 multi-image machine's own algorithms are
    one-sided instead (increase-only) -- see below.
  - Book's own results (Figs. 53/54, MARK-1 on 8 Roman letters):
    algorithm 1 plateaus near 70% reliability after 20-25 samples of
    each letter and does not improve further; algorithm 2 reaches
    ~100% after 35-40 samples. This qualitative gap -- 2 clearly
    better than 1, and 1 having a hard ceiling -- is the main thing
    checked against real digits in notes.md.
  - Sec. 1 also describes, and its footnote (p. 82) fully specifies
    training for, a multi-image extension (Fig. 52): several image
    classes share one A-element layer, but each A-element gets one
    amplifier *per class* (lambda_ja, lambda_jb, ... not a single
    A_j), each class's amplifiers feed a per-class adder, and a
    comparison device picks the class whose adder is largest. This
    architecture is what `MulticlassPerceptron` below implements --
    book-literal, not an extension (an earlier pass through this
    module mislabeled it as our own invention; see that class's
    docstring for the full correction). The footnote's own training
    rule -- coefficients "can only be increased"; algorithm 1
    increases the *shown* class's amplifiers on every step, algorithm
    2 does the same but only on error -- is implemented literally as
    `train_multiclass_book_algorithm_1`/`_2`. `train_multiclass_ovr_
    symmetric` is the one genuine extension in this module: a
    standard symmetric multiclass-perceptron update (increase true
    class, decrease wrongly-predicted class), not attested in the
    book, kept for comparison.

Sec. 2 -- Functions Performed by the A-elements:

  - Purely geometric commentary: fixing an A-element's r_ij defines
    a hyperplane sum_i r_ij*x_i - theta = 0 in receptor space, and
    y_j records which side of it a figure's code falls on. All m
    A-elements together cut receptor space into a large number of
    polyhedra (random, since the r_ij are random); a state of the
    A-element layer (the binary vector y) identifies a polyhedron.
    No new machinery beyond `PerceptronLayer.activate` is needed to
    reconstruct this -- it explains why sec. 3's worked example can
    talk about "polyhedra" at all.

Sec. 3 -- An Example (Table XIX, Figs. 55-57):

  - A fully worked 8-A-element, 21-polyhedron training run under
    algorithm 2. Table XIX itself did not survive OCR in usable form
    (see notes.md) so it is not reproduced as data, but the training
    *procedure* it walks through step-by-step is exactly
    `train_algorithm_2` below -- checked line by line against the
    prose (start all A_j at 1, want output 1 <-> image b, decrease
    excited A_j when a point of image a is wrongly called b, etc.)
    rather than against the garbled numbers.

Sec. 4 -- The Perceptron's Algorithm from the Standpoint of the
Potential Method:

  - Reinterprets algorithm 2's training as summing, over all
    "error" steps, a per-polyhedron function Delta-sigma that is
    maximal in the polyhedron where the error occurred and falls
    off (by exactly 1 per plane crossed on the correct side) moving
    away from it -- explicitly likened to Ch. 4's potential
    functions phi, with the difference that Delta-sigma depends on
    receptor-space geometry (which planes separate two polyhedra)
    rather than on Euclidean/Hamming distance alone. This is a
    conceptual claim about *why* the two chapters' algorithms behave
    alike, not a separate computational recipe; nothing here adds a
    new classifier, so no separate function reconstructs it. What
    *is* checked empirically (notes.md) is the chapter's own
    downstream prediction from this analogy (Fig. 58): reliability
    should degrade *gracefully*, not catastrophically, as A-elements
    are switched off after training -- `evaluate_ablation` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

Array = np.ndarray


@dataclass
class PerceptronLayer:
    """The random, fixed receptor -> A-element wiring (sec. 1).

    `weights` has shape (n_a_elements, n_receptors) with entries in
    {-1, 0, +1}: row j is A-element j's r_ij vector. `theta` is the
    single scalar threshold shared by every A-element.
    """

    weights: Array
    theta: float

    @property
    def n_a_elements(self) -> int:
        return self.weights.shape[0]

    @property
    def n_receptors(self) -> int:
        return self.weights.shape[1]

    def activate(self, x: Array) -> Array:
        """A-element output vector y (sec. 1): y_j = 1 if
        sum_i r_ij*x_i >= theta, else 0.
        """
        x = np.asarray(x, dtype=float)
        s = self.weights @ x
        return (s >= self.theta).astype(np.int8)

    def activate_batch(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        s = X @ self.weights.T
        return (s >= self.theta).astype(np.int8)


def build_random_layer(
    n_receptors: int,
    n_a_elements: int,
    inputs_per_element: int,
    theta: float,
    rng: np.random.Generator,
) -> PerceptronLayer:
    """Random wiring matching MARK-1's construction (sec. 1 and its
    footnote): each A-element connects to a fixed, small number of
    randomly-chosen receptors (not all of them, unlike Ch. 3's
    planes), each connection independently signed +1 or -1, drawn
    once and then held fixed ("during the experiment the connections
    between receptors and A-elements remain unchanged").
    """
    if inputs_per_element > n_receptors:
        raise ValueError("inputs_per_element cannot exceed n_receptors")
    weights = np.zeros((n_a_elements, n_receptors), dtype=np.int8)
    for j in range(n_a_elements):
        idx = rng.choice(n_receptors, size=inputs_per_element, replace=False)
        signs = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=inputs_per_element
        )
        weights[j, idx] = signs
    return PerceptronLayer(weights=weights, theta=theta)


@dataclass
class TwoClassPerceptron:
    """A two-image Perceptron (Fig. 51): the shared A-element layer
    plus a single R-element weight vector A_j.

    Convention, fixed throughout this module to match the sec. 3
    worked example: label 0 <-> "image a", label 1 <-> "image b",
    and training aims for R-element output 1 <-> image b.
    """

    layer: PerceptronLayer
    a_weights: Array  # shape (n_a_elements,)

    @classmethod
    def new(cls, layer: PerceptronLayer) -> "TwoClassPerceptron":
        """Fresh Perceptron with all A_j initialized to 1, matching
        the sec. 3 worked example ("We set the initial values of A_j
        for all amplifiers to one").
        """
        return cls(layer=layer, a_weights=np.ones(layer.n_a_elements))

    def score(self, x: Array) -> float:
        y = self.layer.activate(x)
        return float(self.a_weights @ y)

    def predict(self, x: Array) -> int:
        return 1 if self.score(x) >= 0 else 0

    def predict_batch(self, X: Array) -> Array:
        Y = self.layer.activate_batch(X)
        scores = Y @ self.a_weights
        return (scores >= 0).astype(np.int8)


def train_algorithm_1(
    model: TwoClassPerceptron,
    X: Array,
    labels: Array,
    step: float = 1.0,
) -> List[int]:
    """Algorithm of the first kind (sec. 1, first sub-algorithm):
    update on *every* presentation, irrespective of whether the
    current response is already correct. Excited A-elements' A_j
    are increased if the shown image is label 1 ("b"), decreased if
    label 0 ("a") -- unconditionally.

    Returns the per-step correctness (1/0) list, computed *before*
    each step's update, for reliability tracking.
    """
    correctness = []
    for x, label in zip(X, labels):
        pred = model.predict(x)
        correctness.append(int(pred == label))
        y = model.layer.activate(x)
        delta = step if label == 1 else -step
        model.a_weights = model.a_weights + delta * y
    return correctness


def train_algorithm_2(
    model: TwoClassPerceptron,
    X: Array,
    labels: Array,
    step: float = 1.0,
) -> List[int]:
    """Algorithm of the second kind (sec. 1, second sub-algorithm;
    walked through step-by-step in sec. 3's worked example): the
    classic error-correction perceptron rule. A_j of excited
    A-elements changes *only* on steps where the current response
    was wrong, in the direction that would have produced the
    correct output.
    """
    correctness = []
    for x, label in zip(X, labels):
        pred = model.predict(x)
        correct = int(pred == label)
        correctness.append(correct)
        if not correct:
            y = model.layer.activate(x)
            delta = step if label == 1 else -step
            model.a_weights = model.a_weights + delta * y
    return correctness


def rolling_reliability(correctness: Sequence[int], window: int) -> Array:
    """Trailing-window reliability (fraction correct over the last
    `window` steps), matching what Figs. 53/54 plot: reliability as
    a function of "no. of samples ... shown" *during* training, not
    a separate held-out test.
    """
    c = np.asarray(correctness, dtype=float)
    out = np.full(len(c), np.nan)
    for i in range(len(c)):
        start = max(0, i - window + 1)
        end = i + 1
        out[i] = c[start:end].mean()
    return out


def evaluate_ablation(
    model: TwoClassPerceptron,
    X_test: Array,
    y_test: Array,
    n_switched_off: int,
    rng: np.random.Generator,
) -> float:
    """Reliability of an already-trained Perceptron after randomly
    switching off `n_switched_off` A-elements (sec. 4 / Fig. 58):
    zero out their rows in the wiring matrix and their A_j, then
    measure held-out accuracy. The book's own claim (Fig. 58, MARK-1
    on letters E/X) is graceful degradation -- accuracy staying
    above ~80% even with 7/8 of all A-elements switched off.
    """
    m = model.layer.n_a_elements
    if n_switched_off > m:
        raise ValueError("cannot switch off more A-elements than exist")
    off_idx = rng.choice(m, size=n_switched_off, replace=False)

    weights = model.layer.weights.copy()
    a_weights = model.a_weights.copy()
    weights[off_idx, :] = 0
    a_weights[off_idx] = 0.0

    ablated_layer = PerceptronLayer(weights=weights, theta=model.layer.theta)
    ablated_model = TwoClassPerceptron(
        layer=ablated_layer, a_weights=a_weights
    )

    preds = ablated_model.predict_batch(X_test)
    return float(np.mean(preds == y_test))


@dataclass
class MulticlassPerceptron:
    """The multi-image Perceptron of Fig. 52 (secs. 1's footnote,
    p. 82; the figure itself, p. 79), read directly off the book
    scan (the OCR text layer garbles this passage badly -- lambda
    renders as "A" and sigma as "2" -- so this was read from the
    page image, not the extracted text).

    Fig. 52's architecture, for C images (a, b, c, ...):

      - A single shared A-element layer (this module's
        `PerceptronLayer`), exactly as in the two-image machine.
      - A "lambda layer": each A-element j's output y_j branches to
        *one amplifier per image class* -- coefficients
        lambda_ja, lambda_jb, lambda_jc, ... -- not one shared A_j as
        in the two-image machine. Across all m A-elements and C
        classes that's m*C independent scalars, i.e. exactly
        `class_weights`, shape (C, m): `class_weights[c, j]` is
        lambda_jc.
      - A "sigma layer": one adder per class, each summing that
        class's amplifier outputs across *every* A-element:
        sigma_b = sum_j lambda_jb * y_j. This is `scores`, i.e.
        `class_weights @ y`.
      - A comparison device, not a threshold: "the object is related
        to that image whose adder produces the biggest output
        signal" -- argmax over the adders, i.e. `predict_from_y`.

    This architecture is book-literal, not an extension -- an earlier
    pass through this module mislabeled it as "our own one-vs-rest
    extension"; that was wrong about the *architecture* (which Fig.
    52 specifies exactly) though right that the book's own *training*
    rule differs from what was first implemented here -- see
    `train_multiclass_book_algorithm_1`/`_2` below, which are the
    literal rule, versus `train_multiclass_ovr_symmetric`, which is
    the genuine extension (a standard, but not book-attested,
    symmetric multiclass-perceptron update).
    """

    layer: PerceptronLayer
    class_weights: Array  # shape (n_classes, n_a_elements)
    classes: Array

    @classmethod
    def new(
        cls, layer: PerceptronLayer, classes: Sequence[int]
    ) -> "MulticlassPerceptron":
        classes = np.asarray(classes)
        return cls(
            layer=layer,
            class_weights=np.zeros((len(classes), layer.n_a_elements)),
            classes=classes,
        )

    def scores(self, y: Array) -> Array:
        return self.class_weights @ y

    def predict_from_y(self, y: Array) -> int:
        return int(self.classes[np.argmax(self.scores(y))])

    def predict(self, x: Array) -> int:
        return self.predict_from_y(self.layer.activate(x))

    def predict_batch(self, X: Array) -> Array:
        Y = self.layer.activate_batch(X)
        scores = Y @ self.class_weights.T
        idx = np.argmax(scores, axis=1)
        return self.classes[idx]


def _class_index(model: MulticlassPerceptron) -> dict:
    return {int(c): i for i, c in enumerate(model.classes)}


def train_multiclass_book_algorithm_1(
    model: MulticlassPerceptron,
    X: Array,
    labels: Array,
    step: float = 1.0,
) -> List[int]:
    """Book-literal algorithm 1 for Fig. 52 (sec. 1 footnote,
    p. 82): "the coefficients lambda_j ... can only be increased. In
    algorithms of the first type, the lambda_j corresponding to a
    given figure are increased at each step" -- unconditionally, on
    every presentation, whether or not the current response is
    already correct. Only the *shown* class's amplifiers move, and
    only upward; no other class's amplifiers are touched, and
    nothing is ever decreased.
    """
    class_index = _class_index(model)
    correctness = []
    for x, label in zip(X, labels):
        y = model.layer.activate(x)
        pred = model.predict_from_y(y)
        correctness.append(int(pred == label))
        idx = class_index[int(label)]
        model.class_weights[idx] += step * y
    return correctness


def train_multiclass_book_algorithm_2(
    model: MulticlassPerceptron,
    X: Array,
    labels: Array,
    step: float = 1.0,
) -> List[int]:
    """Book-literal algorithm 2 for Fig. 52 (same footnote): "in
    algorithms of the second type the coefficients are increased in
    exactly the same fashion but only when the Perceptron's response
    is incorrect." Same increase-only, shown-class-only update as
    algorithm 1, gated on an incorrect response.
    """
    class_index = _class_index(model)
    correctness = []
    for x, label in zip(X, labels):
        y = model.layer.activate(x)
        pred = model.predict_from_y(y)
        correct = int(pred == label)
        correctness.append(correct)
        if not correct:
            idx = class_index[int(label)]
            model.class_weights[idx] += step * y
    return correctness


def train_multiclass_ovr_symmetric(
    model: MulticlassPerceptron,
    X: Array,
    labels: Array,
    step: float = 1.0,
) -> List[int]:
    """Our own extension, NOT book content: the standard symmetric
    multiclass-perceptron update. On an error, increase the true
    class's weights on excited A-elements *and* decrease the
    (wrongly) predicted class's weights on excited A-elements --
    unlike the book's Fig. 52 rule, which only ever increases the
    shown class's own amplifiers and never touches a competitor's.
    """
    class_index = _class_index(model)
    correctness = []
    for x, label in zip(X, labels):
        y = model.layer.activate(x)
        pred = model.predict_from_y(y)
        correct = int(pred == label)
        correctness.append(correct)
        if not correct:
            true_i = class_index[int(label)]
            pred_i = class_index[int(pred)]
            model.class_weights[true_i] += step * y
            model.class_weights[pred_i] -= step * y
    return correctness
