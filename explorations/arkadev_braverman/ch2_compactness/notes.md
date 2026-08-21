# Chapter 2 — The Compactness Hypothesis, tested on real handwriting

## What we built

- `receptor_space.py` — the book's Ch. 2 machinery: binary/grayscale
  encoding, Hamming/Euclidean distance, the exact single-bit-flip
  internal/boundary-point definition (sec. 3), and a generalized
  `CompactnessReport` (boundary fraction + same-class vs. other-class
  nearest-neighbor margin) that also works on continuous codes, which
  the book's own definition can't handle.
- `generate_compact.py` — the book's own synthetic cloud generator
  (sec. 4, Figs. 15-16): random seed bitmap + independent per-cell
  flip noise (default p=0.1, as in the book). Used only as a
  *control* — these clouds are compact by construction, so they show
  what "maximally compact" looks like under our own measurements.
- `digits_experiment.py` — runs the compactness measurement on
  scikit-learn's real `load_digits` corpus (1797 real handwritten
  8x8 = 64-pixel images), both binarized (threshold=8, the book's
  exact black/white formalism) and grayscale (normalized [0,1], the
  continuous extension the book only sketches), plus the synthetic
  control for comparison. Produces `plots/compactness_margins.png`.

## Verification against the book itself

Before trusting any of this on real data, we checked
`is_internal_point_binary` / `boundary_fraction_binary` against the
book's own two worked hand examples (Ch. 2 sec. 3, pp. 17-20):

- Sets (00, 11) vs. (01, 10) — literally XOR. The book states "all
  points of these sets are boundary points." Our code agrees exactly
  (boundary_fraction = 1.0 for both).
- Sets {11} vs. {00, 01}. The book says 00 is internal, 01 is a
  boundary point, and 11 (whose only neighbors 01/10 include one
  point from the other set) is also a boundary point. Our code
  reproduces all three verdicts.

## Results

### Binarized real digits (threshold=8, Hamming distance)

| digit | n   | boundary_frac | same-class NN | other-class NN | margin |
|-------|-----|---------------|----------------|------------------|--------|
| 0     | 178 | 0.000         | 2.185          | 6.449            | 4.264  |
| 1     | 182 | 0.000         | 2.005          | 5.093            | 3.088  |
| 2     | 177 | 0.000         | 3.305          | 7.305            | 4.000  |
| 3     | 183 | 0.000         | 3.508          | 6.191            | 2.683  |
| 4     | 181 | 0.000         | 2.972          | 7.414            | 4.442  |
| 5     | 182 | 0.000         | 3.319          | 6.582            | 3.264  |
| 6     | 181 | 0.000         | 2.326          | 7.365            | 5.039  |
| 7     | 179 | 0.000         | 3.140          | 7.318            | 4.179  |
| 8     | 174 | 0.000         | 3.908          | 5.638            | 1.730  |
| 9     | 180 | 0.000         | 3.711          | 5.872            | 2.161  |

mean margin: **3.485**

### Grayscale real digits (normalized [0,1], Euclidean distance)

Same qualitative ranking (8 lowest, 0/4/6 among the highest), mean
margin **0.814** on this scale. `boundary_fraction` is `nan` here —
expected, not a bug: the single-bit-flip definition has no meaning
once co-ordinates are continuous (documented in
`receptor_space.py`'s module docstring).

### Synthetic control (compact by construction)

Same measurement, boundary_frac = 0.000 across the board (as
expected), mean margin **15.394** — roughly 4-5x every real digit
class's margin.

## Interpretation

**The headline finding: `boundary_fraction = 0.000` for every real
digit class**, exactly matching the synthetic control. Independently
verified by brute-force pairwise check outside the module (see
conversation record): across all ~1.6 million pairs in the 1797-point
corpus, the minimum cross-class Hamming distance is >= 2. Not one
single-bit change anywhere in the dataset crosses from one digit
class to another, after this coarse 64-cell binarization.

That's a genuine, unforced confirmation of the compactness
hypothesis on real handwriting, not synthetic data engineered to
satisfy it. It's also a nice validation of *why* the book's dissecting-
planes and Perceptron algorithms (Ch. 3, Ch. 5) can work at all: if
opposing classes never came within one bit of each other, a
separating hyperplane is guaranteed to exist in principle.

**But the boundary-fraction test alone is too coarse to be the whole
story.** It's a binary pass/fail criterion that both real and
synthetic data satisfy trivially at only 64 dimensions -- it doesn't
distinguish "barely compact" from "extremely compact." The margin
statistic does: real handwriting's margins (1.7-5.0) are an order of
magnitude smaller than the synthetic control's (14-17), even though
both pass the same boundary test. The margin is doing the real
discriminating work; boundary_fraction alone would have made real
handwriting and pure noise-cloud synthetic data look identical.

**Per-class asymmetry**: digit 8 is consistently the least compact
real class (margin 1.73 binarized, 0.487 grayscale) -- matching
visual intuition, since 8 shares structural features with 0, 3, and
9 at this resolution. Digit 6 (binarized) and digits 0/4/6
(grayscale) are the most compact. This foreshadows the
threshold-sensitivity question raised earlier in conversation: if
some digits are already less compact than others at a "neutral"
threshold, they're presumably also more fragile to bad binarization
choices -- a natural first thing to check when we get to the
threshold-sweep experiment.

## Open questions / things to revisit

- Is `same_class_neighbor_margin` the right single-number summary of
  compactness, or should we weight it by within-class variance too?
- The threshold-sweep experiment (does accuracy/compactness degrade
  linearly or in a U-shape as the binarization threshold moves off
  center?) is still pending -- a natural next step once we move to
  Chapter 3's classifier, since compactness alone doesn't yet measure
  classification accuracy, just separability.
