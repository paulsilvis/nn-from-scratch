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
    for the whole experiment.
  - Two training algorithms, both restricted to a two-image problem
    (image a vs. image b), presented one example at a time:
      Algorithm 1 (unconditional): on every step, A_j of every
        *excited* A-element is nudged in the same fixed direction
        determined solely by which image was shown, whether or not
        the current response was already correct.
      Algorithm 2 (error-correcting, the classic perceptron rule):
        A_j changes only on steps where the current response was
        *wrong*, nudging excited A-elements' weights in the
        direction that would have produced the correct output.
    Both are one-sided in the book's literal statement (footnote,
    p. 82: for the multi-image machine of Fig. 52, A_j "can only be
    increased"); for the plain two-image machine of Fig. 51 that
    this module implements, the direction is symmetric (+ for one
    image, - for the other), matching the sec. 1 prose and the
    worked example of sec. 3.
  - Book's own results (Figs. 53/54, MARK-1 on 8 Roman letters):
    algorithm 1 plateaus near 70% reliability after 20-25 samples of
    each letter and does not improve further; algorithm 2 reaches
    ~100% after 35-40 samples. This qualitative gap -- 2 clearly
    better than 1, and 1 having a hard ceiling -- is the main thing
    checked against real digits in notes.md.
  - Sec. 1 also notes the machine can be extended to more than two
    images (Fig. 52: several groups of A-elements, each with its own
    adder/R-element, either "biggest adder wins" or a binary-code
    combination of R-element outputs). The book does not spell out a
    training rule for this multi-image case beyond "analogous to"
    the two-image one; `train_multiclass_algorithm_2` below is our
    own one-vs-rest extension of algorithm 2 (flagged as such), used
    only for the full 10-digit experiment.

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
    """One-vs-rest extension of the two-image machine to C classes,
    sharing a single A-element layer (Fig. 52's "several groups"
    idea, but with one shared random projection rather than
    per-class A-element groups -- see the docstring note in sec. 1
    above: the book does not fully specify a multiclass training
    rule, so this is our own extension, used only for the full
    10-digit comparison, not attributed to the book as literal
    content).
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


def train_multiclass_algorithm_2(
    model: MulticlassPerceptron,
    X: Array,
    labels: Array,
    step: float = 1.0,
) -> List[int]:
    """Our one-vs-rest extension of algorithm 2: on an error, increase
    the true class's weights on excited A-elements and decrease the
    (wrongly) predicted class's weights on excited A-elements -- the
    standard multiclass-perceptron generalization of the book's
    two-image error-correction rule.
    """
    class_index = {int(c): i for i, c in enumerate(model.classes)}
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
