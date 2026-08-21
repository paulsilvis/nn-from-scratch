# Chapter 4 — Algorithms Based on Potentials Methods, tested on real handwriting

Read directly from the book scan the user supplied (secs. 1 and 2, pp. 59-74),
before writing any code.

## What the chapter actually says

**Sec. 1 (Potentials in the Receptor Space).** An electrostatic
analogy: a point-source generates potential phi(R) = 1/(1+alpha*R^2),
maximal at R=0, decaying with distance R (Euclidean or Hamming — the
book uses either). A class's "image potential" at a query point is
the *mean* potential from every training point of that class (sum
over sources, divided by n_a — the book is explicit about the
normalization). Classify by whichever class's mean potential is
largest. The book's own experiment (Fig. 42, real digits 0-9) found
this plateaus at ~85% reliability past N~13 training examples per
class, and diagnoses it as non-uniform point density letting one
class's dense cluster out-vote another class's sparse region, even
misclassifying already-*seen* training points. The fix (sec. 4's
Fig. 44 flow chart): after training, re-recognize the training set
itself; bump the weight of every point still misclassified by one;
repeat in cycles until the training set is fully recognized. Table
XVII: this raised average reliability 85.0% -> 89.3% at N=21, with
the biggest gains on the worst classes (digit 9: 49.3% -> 65.3%).

**Sec. 2 (Potentials in the Receptor Field).** A different problem:
plain 0/1 receptor codes are blind to *how far* a stroke shifted — a
"5" nudged one square (still a "5") and one nudged five squares (now
plausibly a "3") can be equidistant, in Hamming/Euclidean terms, from
the original. Fix: re-encode each figure before computing any
distance — every excited receptor element contributes potential 1 to
itself and a fraction (the book uses 1/4) to elements adjoining it
(vertically, horizontally, diagonally), summed across all excited
elements of the figure. This turns the binary code into a real-valued,
spatially-blurred one. Table XVIII (N=12): average reliability rose
85.0% -> 94.0%, biggest gains again on the previously-worst classes
(digit 8: 76.2% -> 100%; digit 9: 42.0% -> 64.0%).

**A caveat on the book's own worked numeric example (Fig. 48):** the
OCR of the scan garbles both the fractional codes ("$440004140004")
and the quoted root values ("x)2" for a square root, "ay6" for
another). These couldn't be reliably parsed into exact numbers, so
this reconstruction doesn't try to reproduce that specific example —
see "Verification" below for what was checked instead.

## What we built

- `potentials.py`:
  - `PotentialClassifier` — sec. 1's classifier: `fit()` just records
    codes/labels/weights (the book's "learning" is memorization);
    `class_potentials()` / `predict()` compute mean weighted potential
    per class and argmax. `fit_improved()` — sec. 4's iterative
    reweighting (Fig. 44): predict on the training set itself, bump
    weights of misclassified points by one, repeat until converged or
    a cycle cap.
  - `receptor_field_potential_encode` / `encode_figures_potential` —
    sec. 2's spatial blur: each excited (or grayscale-weighted)
    receptor element spreads `neighbor_weight` of its value to its
    8-connected neighbors, on top of its own value, summed across
    all sources.
  - Reuses `receptor_space.euclidean_distance` / `hamming_distance`
    from `ch2_compactness` rather than duplicating them.
- `experiment.py` — three checks against real digits, mirroring the
  chapter's own structure (Fig. 42, Table XVII, Table XVIII).

## Verification

- **Qualitative shift check** (stand-in for the garbled Fig. 48
  example): built a single excited element on an 18-cell strip and
  shifted it by 0-6 squares.
  - Plain binary coding: Euclidean distance between the original and
    *any* non-overlapping shift is constant at sqrt(2) (Hamming
    distance 2, always) — exactly the book's claim that its "old"
    method treats all sufficiently-large shifts as equally different.
  - Potential-encoded (neighbor_weight=0.25): distance grows with
    shift size — 1.118 (shift 1) -> 1.458 (shift 2) -> 1.5 (shift >=3,
    where it saturates, since our kernel only reaches one cell out).
    This confirms the book's qualitative point directly, even though
    the exact numbers in the book's own example couldn't be recovered
    from the OCR. The saturation at shift>=3 is a real limitation of
    a radius-1 kernel worth flagging: the book's own potential
    function decays continuously and never fully saturates, so a
    literal continuous phi(R) applied per-source (rather than our
    crude 3x3 step kernel) would keep separating farther shifts —
    not implemented here, since the book itself calls the step
    function with a single fraction a "crude approximation" adequate
    for its coarse receptor field.

## Results (real digits, `load_digits`, alpha=1.0, Euclidean metric)

alpha isn't specified numerically by the book; 1.0 was chosen (not
tuned) to leave meaningful potential mass beyond R=0 for our
[0,1]-normalized grayscale codes.

### Reliability vs. N (cf. Fig. 42)

| N per class | 2 | 4 | 6 | 8 | 10 | 13 | 16 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| reliability | 0.733 | 0.837 | 0.883 | 0.886 | 0.901 | 0.911 | 0.913 | 0.925 | 0.929 | 0.936 |

**Divergence from the book:** our curve does *not* plateau the way
Fig. 42's does. The book's reliability flatlines around 85% past
N~13; ours keeps climbing smoothly past N=30, reaching ~93.6%.
Plausibly the same explanation as Ch. 2/3's own divergences: real
scanned handwriting (scikit-learn's corpus) is more internally
consistent than the book's ~200 hand-prepared 1960s numerals, so the
non-uniform-density failure mode the book diagnoses is milder here —
there's less of a plateau to hit in the first place.

### Simple vs. improved (reweighted) (cf. Table XVII)

| N | simple avg | improved avg | cycles to converge |
|---|---|---|---|
| 12 | 0.937 | 0.937 | 1 |
| 21 | 0.947 | 0.947 | 1 |

**Divergence from the book:** *zero* difference between simple and
improved, at both training sizes — `n_cycles_=1` means the very first
recognition pass over the training set already had no errors, so the
reweighting loop never fires. This makes sense given how the
classifier works: a training point's own potential contribution to
its own class includes an R=0 term (its own maximal potential of
exactly 1), so a training point trivially tends to "recognize itself"
as long as no other class's cluster potential exceeds that — which,
per Ch. 2's own compactness finding, is rare for these well-separated
real digit classes. The book's own numeral set apparently *did* have
enough self-recognition errors on its training set for the reweighting
mechanism to matter (Table XVII shows real gains, especially for
digit 9). Worth flagging as a case where the book's own machinery is
correctly reconstructed but genuinely has nothing to do here, because
the failure mode it was designed to fix barely occurs in our data.

### Receptor-field potential encoding (cf. Table XVIII)

Two baselines were tried, since our other chapters default to
continuous grayscale codes, but the book's own baseline was plain
black/white:

| baseline | plain avg | potential-encoded avg (neighbor_weight=0.25) |
|---|---|---|
| grayscale ([0,1]-normalized) | 0.937 | 0.877 |
| binary (thresholded at 8/16) | 0.850 | 0.837 |

**The binary-baseline number (0.850) lands almost exactly on the
book's own reported 85.0%** for the simplest algorithm at N=12 — a
nice cross-check that our classifier's calibration matches the
book's, even though the corpora differ.

**Divergence from the book:** encoding *hurt* overall reliability in
both baselines, opposite the book's own +9-point gain to 94.0%.
Looking at the binary-baseline per-class breakdown, the pattern is
mixed rather than uniformly bad — some previously-hard classes
improved a lot (digit 8: 0.686 -> 0.829; digit 4: 0.960 -> 1.000;
digit 7: 0.941 -> 1.000), matching the book's own qualitative claim
that hard classes benefit most — but two classes got substantially
worse (digit 1: 0.824 -> 0.559; digit 2: 0.926 -> 0.704), enough to
flip the average.

A follow-up sweep of `neighbor_weight` on the binary baseline (0.1,
0.15, 0.25, 0.4, 0.5) found reliability peaks around **0.86-0.87** at
`neighbor_weight` in [0.1, 0.15] — a modest real gain over the 0.850
plain-binary baseline — and *degrades* monotonically above 0.25 (down
to 0.79 at 0.5). Plausibly the book's own choice of "say 1/4" was
tuned (even if only informally) to its own coarser "large-grain"
receptor field; our finer 8x8 real-digit grid has thin one- and
two-pixel-wide strokes (digits 1 and 2 especially), where a 1/4
spread to all 8 neighbors blurs distinguishing detail away rather
than adding useful tolerance to small shifts. `experiment.py` still
uses the book's literal 1/4 as its headline number for comparability
with Table XVIII, but this is worth revisiting if the encoding is
reused elsewhere in later stages.

## Open questions / things to revisit

- The angle/weight of receptor-field spreading (`neighbor_weight`)
  is real-data-sensitive, per the sweep above; a per-class or
  resolution-adaptive weight isn't implemented.
- The reweighting improvement (sec. 4/Fig. 44) has nothing to
  correct on our well-separated data; it might show a real effect on
  a harder synthetic dataset (e.g. digits with injected label noise,
  or Ch. 2's synthetic non-compact control), which could be a useful
  follow-up check to actually exercise this part of the algorithm.
- Sec. 1's alpha (decay rate) wasn't tuned or swept, only chosen to
  be non-degenerate; a proper sweep (as loosely done for
  `neighbor_weight`) is a natural next check.
- The book's own literal Fig. 48 numeric example remains unverified
  against exact numbers, due to OCR corruption of the scan at that
  specific passage; only the qualitative claim was confirmed.
