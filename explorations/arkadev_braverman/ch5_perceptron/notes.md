# Chapter 5 — Algorithm of the Perceptron, tested on real handwriting

Read directly from the book scan the user supplied (secs. 1-4, pp. 76-93),
before writing any code. Secs. 5 (PAPA, Gamba's analog realization) and 6
(the Perceptron as a model of the brain) were also read but are
hardware/historical discussion with no computational recipe to
reconstruct, so nothing here implements them.

## What the chapter actually says

**Sec. 1 (Structure and Algorithm of the Perceptron).** MARK-1's
architecture: n receptors (photoresistances, x_i in {0,1}) feed m
A-elements, each wired at setup time to a fixed, random subset of
receptors with random +1/-1 signs (footnote: MARK-1 itself used
n=400, m=512, 20 connections per A-element — a small fraction of all
receptors, not all of them, unlike Ch. 3's dissecting planes which
used every coordinate). Each A-element fires (y_j=1) if its weighted
receptor sum clears a shared threshold theta, otherwise 0. A single
R-element sums A_j*y_j over all A-elements and outputs 1 if that sum
is >= 0. Two training algorithms, both restricted to a two-image
problem, examples shown one at a time:

  - **Algorithm 1** (unconditional): every presentation nudges the
    A_j of *excited* A-elements in a fixed direction determined only
    by which image was shown — correct or not.
  - **Algorithm 2** (error-correcting — the classic perceptron rule):
    A_j changes *only* on incorrect steps, in the direction that
    would have produced the right answer.

Book's own result (Figs. 53/54, MARK-1 on 8 Roman letters): algorithm
1 plateaus near 70% after 20-25 samples/letter and never improves
further; algorithm 2 reaches ~100% after 35-40 samples/letter.

**Sec. 2 (Functions Performed by the A-elements).** Purely
geometric: fixing an A-element's connections defines a hyperplane in
receptor space; y_j records which side of it a figure falls on. All
m A-elements together cut receptor space into a large number of
polyhedra, and a state of the whole A-element layer identifies one
polyhedron. This is what lets sec. 3 talk about "polyhedra" at all —
no separate machinery needed beyond the activation function itself.

**Sec. 3 (An Example, Table XIX, Figs. 55-57).** A fully worked
8-A-element, 21-polyhedron run under algorithm 2: start all A_j at
1, define output 1 <-> image b, and walk through 8 training steps,
tracking which polyhedron each point falls in and how A_j changes.
**Table XIX did not survive OCR usably** (row/column structure
completely scrambled by the scan — see "What we built" below); it
isn't reproduced as data. The *procedure* the prose describes step by
step, though, is unambiguous and matches `train_algorithm_2` exactly.

**Sec. 4 (The Perceptron's Algorithm from the Standpoint of the
Potential Method).** Reframes algorithm 2's training as summing a
per-polyhedron function Delta-sigma (maximal in the polyhedron where
an error just occurred, falling by exactly 1 per correctly-facing
plane crossed moving away from it) over every error step — explicitly
likened to Ch. 4's potential functions phi, with the caveat that
Delta-sigma depends on receptor-space *geometry* (which planes
separate two polyhedra) rather than distance alone. From this
analogy the book predicts (and shows, Fig. 58, MARK-1 on letters E/X)
that reliability degrades *gracefully* — not catastrophically — as
A-elements are switched off after training, even losing 7/8 of them
without falling below ~80%. This conceptual claim, not a new
classifier, is what's actually checked against real digits below.

## What we built

- `perceptron.py`:
  - `PerceptronLayer` / `build_random_layer` — sec. 1's random,
    fixed receptor-to-A-element wiring, with a *bounded* number of
    connections per A-element (book's 20-of-400 ratio), unlike Ch.
    3's planes which always used every receptor.
  - `TwoClassPerceptron` plus `train_algorithm_1` / `train_algorithm_2`
    — the book's two training rules, sharing one convention (label
    0 <-> image a, label 1 <-> image b) throughout, matching sec. 3's
    worked example.
  - `rolling_reliability` — trailing-window accuracy, matching what
    Figs. 53/54 actually plot (reliability *during* training, not a
    separate held-out test).
  - `evaluate_ablation` — sec. 4/Fig. 58's robustness check: zero out
    a random subset of A-elements (both their receptor wiring and
    their R-element weight) and re-measure held-out accuracy.
  - `MulticlassPerceptron` / `train_multiclass_algorithm_2` — **our
    own one-vs-rest extension** to more than two classes, sharing one
    random A-element layer across per-class weight vectors. The book
    only gestures at multi-image Perceptrons (Fig. 52, "several
    groups of A-elements... analogous to" the two-image case) without
    fully specifying a training rule for them, so this is flagged as
    an extension, not literal book content, and used only for the
    10-digit cross-chapter comparison below.
- `experiment.py` — four checks against real digits, mirroring the
  chapter's own structure (Figs. 53/54, Fig. 58, plus our own
  multiclass cross-check).

## Results (real digits, `load_digits`, binary receptor codes
thresholded at intensity 0.3; algorithm-comparison and ablation
experiments use 256 A-elements/16 inputs each, the multiclass
experiment uses 512/20 — noted per section below)

### Algorithm 1 vs. 2 (cf. Figs. 53/54)

A single training run (40 presentations/class) turned out too noisy
to compare fairly, especially on a hard pair — a handful of unlucky
early presentations can dominate. Results below average 5 independent
random wirings/presentation orders, and report the mean of the last
20 rolling-reliability points (a less noisy stand-in for "the
asymptote" than the single final point):

| pair | algorithm 1 | algorithm 2 |
|---|---|---|
| 0 vs. 1 (easy) | 79.7% | 96.7% |
| 8 vs. 9 (hard, per Ch. 2's compactness ranking) | 51.4% | 69.5% |

**This matches the book's qualitative claim well**: algorithm 2
clearly and consistently beats algorithm 1, on both an easy and a
hard pair. The book's own specific numbers (70% -> ~100%) are a
bigger gap than ours (partly because 8-vs-9 is a genuinely hard
real-digit pair that even algorithm 2 doesn't fully solve at 69.5%,
unlike MARK-1's letters). Worth flagging as a methodological note
rather than a substantive divergence: reading reliability off a
*single* seed's final data point (rather than averaging trials and
using a trailing window) can make algorithm 1 look competitive with,
or even better than, algorithm 2 purely from noise — an early run at
default settings did show exactly that inversion before this fix.

### Ablation (cf. Fig. 58)

Trained (algorithm 2) on digits 8 vs. 9, then measured held-out
accuracy after randomly switching off increasing numbers of the 256
A-elements (5 trials per count):

| A-elements switched off | reliability |
|---:|---|
| 0 | 90.8% +/- 0.0% |
| 32 | 89.5% +/- 1.9% |
| 64 | 86.7% +/- 2.7% |
| 96 | 88.5% +/- 2.5% |
| 128 | 86.7% +/- 2.8% |
| 160 | 84.3% +/- 4.2% |
| 192 | 81.7% +/- 5.8% |
| 224 | 75.7% +/- 5.2% |
| 240 | 65.7% +/- 8.9% |
| 250 (~half) | 60.2% +/- 9.9% |

**Matches the book's qualitative shape** — degradation is gradual,
not a cliff, all the way through roughly the first half of the
A-elements. **Diverges somewhat in the specific floor**: the book
reports MARK-1 staying above 80% even with 7/8 of all A-elements gone
(near-total ablation); our curve drops below 80% well before that
point (already at 224/256 = 87.5% switched off, matching the book's
own 7/8 fraction exactly, but landing at 75.7% instead of >80%).
Plausibly our smaller, lower-input-count A-element layer (256
elements, 16 receptors each, on a 64-receptor field) has less
built-in redundancy per A-element than MARK-1's — with fewer total
planes and less overlap between them to begin with, losing any given
fraction removes proportionally more of the *distinguishing*
structure.

### Multiclass one-vs-rest, all 10 digits (our extension, not book
content — for cross-chapter comparison only)

| training | held-out accuracy |
|---|---|
| single pass over 200 examples/class | 81.0% +/- 6.2% |
| 5 passes over the same 200 examples/class | 88.0% +/- 2.4% |

**Divergence worth flagging**: read completely literally ("training
... is performed in a sequence of steps" showing one example at a
time), a single pass through the data underperforms Ch. 3's
dissecting-planes (~92% at a comparable training size) and Ch. 4's
potentials method (~93.6%) by a real margin. The book's text never
actually says training is limited to one pass, though, and repeating
the same training set for several epochs — each in a freshly-shuffled
order — closes most of that gap (81.0% -> 88.0%). This reads as a
genuine property of the error-correcting perceptron rule (it needs
to *see* an error to correct it, and one-vs-rest 10-way separation on
512 shared, only-20-connections-wide A-elements needs more corrective
passes to shake out than the direct plane-fitting or potential-mean
approaches of Ch. 3/4), not an artifact of a bad hyperparameter — a
quick sweep (not shown in `experiment.py`, run interactively) found
more A-elements alone (1024 vs. 512) barely moved single-pass
accuracy, while more inputs per A-element (32 vs. 20) helped
noticeably more than more A-elements did.

## Verification against sec. 3's worked example

Table XIX's actual numbers weren't recoverable from the OCR (the
scan mangled row/column alignment badly enough that even individual
cell values are ambiguous), so no attempt was made to reproduce them.
What *was* checked line-by-line against the prose (not the table) is
the training procedure itself: `TwoClassPerceptron.new` starts every
A_j at 1 (matching "we set the initial values of A_j for all
amplifiers to one"), and `train_algorithm_2`'s logic — change A_j of
excited A-elements only on an incorrect step, in the direction that
would have corrected it — matches the prose's own account of steps
1-8 (decrease on the first, incorrect step; leave unchanged on
correct steps 2, 4, 6; change again on further incorrect steps 3, 5,
7; correct from step 8 onward) exactly in *kind*, even without the
table to check exact magnitudes against.

## Open questions / things to revisit

- Sec. 4's Delta-sigma / potential-method reinterpretation of
  algorithm 2 is a conceptual claim, not a separate computation; a
  literal implementation of Delta-sigma per polyhedron (rather than
  just checking its downstream prediction via ablation) isn't
  attempted here and could be a genuine follow-up.
- The ablation floor (### Ablation above) is real-data/architecture
  sensitive; a version with MARK-1's actual receptor-to-A-element
  ratio (20-of-400, i.e. ~5% connectivity, vs. our 16-of-64 = 25%
  used in the ablation experiment, since our receptor field is much
  smaller) wasn't tried, and might change the degradation curve's
  shape.
- The one-vs-rest multiclass extension is explicitly *ours*, not the
  book's; the book's own multi-image sketch (Fig. 52, shared
  A-elements feeding several adders, or binary-coded R-element
  outputs) was not implemented as literally described, since it
  doesn't specify a training rule precisely enough to reconstruct
  without guessing.
- theta (the shared A-element threshold) was fixed at 0 without any
  sweep, mirroring alpha's treatment in Ch. 4; a proper sweep is a
  natural next check, as is a sweep of inputs-per-A-element (the
  quick interactive check above suggests this matters more than raw
  A-element count for multiclass separation).
