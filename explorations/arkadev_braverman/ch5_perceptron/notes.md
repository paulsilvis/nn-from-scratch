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
R-element sums lambda_j*y_j over all A-elements and outputs 1 if that
sum is >= 0 (book's actual symbol is lambda_j, not A_j — the
extracted OCR text renders lambda as "A" throughout this passage;
this was only caught by checking the page image directly, see the
correction under "Multiclass" in Results below). Two training
algorithms, both restricted to a two-image problem, examples shown
one at a time:

  - **Algorithm 1** (unconditional): every presentation nudges the
    lambda_j of *excited* A-elements in a fixed direction determined
    only by which image was shown — correct or not.
  - **Algorithm 2** (error-correcting — the classic perceptron rule):
    lambda_j changes *only* on incorrect steps, in the direction that
    would have produced the right answer.

Book's own result (Figs. 53/54, MARK-1 on 8 Roman letters): algorithm
1 plateaus near 70% after 20-25 samples/letter and never improves
further; algorithm 2 reaches ~100% after 35-40 samples/letter.

Sec. 1 goes on to describe a multi-image extension for more than two
classes (Fig. 52, p. 79): A-elements stay shared across classes, but
each A-element's output branches to one amplifier *per class*
(lambda_ja, lambda_jb, lambda_jc, ...), each class's amplifiers feed
one adder (sigma_a, sigma_b, sigma_c, ...) summing over every
A-element, and a comparison device picks the class whose adder is
largest. A footnote (p. 82) specifies training for this case too:
coefficients "can only be increased" — algorithm 1 increases the
*shown* class's amplifiers on every step; algorithm 2 does the same
only on error. See "Multiclass" under Results for how this was
initially missed (both the figure's detail and the footnote) and
what implementing it literally revealed.

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
  - `MulticlassPerceptron` — Fig. 52's multi-image architecture
    (book-literal, not an extension: shared A-element layer, one
    amplifier lambda_jc per (A-element, class) pair, one adder
    sigma_c per class, argmax comparison — see "Correction" under
    Results below for how an earlier pass mischaracterized this).
    Trained by `train_multiclass_book_algorithm_1` / `_2` (the
    book's own increase-only rules, from a footnote easy to miss on
    a first pass through the OCR text) or `train_multiclass_ovr_
    symmetric` (**our actual extension**: the standard symmetric
    multiclass-perceptron update, decreasing a wrongly-predicted
    class's amplifiers too — something neither book rule does).
- `experiment.py` — four checks against real digits, mirroring the
  chapter's own structure (Figs. 53/54, Fig. 58, plus a three-way
  multiclass training-rule comparison).

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

**Follow-up: why is 8-vs-9 hard, and does it ever actually converge?**
Two things worth separating out, checked after the initial pass above:

- *Why 8-vs-9 is hard*: Ch. 2's own `compactness_report` on these
  binarized codes shows digit 8 (same-class-neighbor margin 1.56) and
  digit 9 (margin 1.98) have the two *smallest* margins of any digit
  in the corpus — versus digit 0's margin of 5.32, the *largest* of
  any digit, and digit 1's 2.63 (still comfortably above either 8 or
  9). Visually, 8 and 9 share a closed loop at the top and a
  down-right trailing stroke; at 8x8 resolution a meaningful fraction
  of examples are genuinely ambiguous between them. This is exactly
  the mechanism sec. 4's potential-method analogy would predict: a
  low receptor-space margin means few of the random A-element planes
  cleanly separate the two classes, so both algorithms need more
  corrective evidence to settle on a working combination.
- *Does more training resolve it?* At just 40 presentations/class,
  neither algorithm had actually converged — per-trial tails bounce
  unpredictably enough that algorithm 1 sometimes edges out algorithm
  2 at the very last data point (a coin flip from residual
  oscillation, not a reversal of which algorithm is better). Rerunning
  at 150 presentations/class (5-trial average) resolves this cleanly:
  algorithm 2 settles into a stable ~15-20-point lead over algorithm 1
  for nearly the entire run (86-96% vs. 60-74% through most of
  training), rather than the ambiguous ordering seen at 40. Neither
  algorithm fully flatlines at 100% the way the easy pair eventually
  does — both keep oscillating somewhat even at 150/class — but the
  *ranking* becomes unambiguous well before that point. So 8-vs-9
  isn't unsolvable at this A-element budget; the 40-presentation
  comparison above was simply cut short relative to how much evidence
  a low-margin pair needs. (Not run to full convergence or added as a
  permanent `experiment.py` function, to keep the headline comparison
  at a consistent, book-comparable presentation count — this was a
  one-off diagnostic check, saved as `plots/algo_comparison_8_9_long.png`.)
- *A related correction on algorithm 1's easy-pair number*: checking
  the un-averaged per-trial tails for 0-vs-1 shows algorithm 1 does
  eventually reach ~94-100% in 4 of 5 trials by presentation 80 — it's
  simply slower to get there than algorithm 2, climbing steadily
  rather than jumping early. The 79.7% reported above (mean of the
  last 20 rolling-reliability points) partly captures that slow climb
  rather than the true endpoint, understating how close the two
  algorithms end up on an easy, well-separated pair. This is a real,
  if small, divergence from the book's own letters result (Fig. 53),
  where algorithm 1 hits a *hard* ceiling that more training does not
  move past at all — on this easy digit pair, algorithm 1 isn't
  capped, just slower.

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

### Multiclass, all 10 digits (Fig. 52's actual architecture — see
correction below)

**Correction to an earlier pass through this file**: the multiclass
architecture used here (shared A-element layer, one amplifier per
(A-element, class) pair, one adder per class, argmax comparison) was
previously described as "our own one-vs-rest extension, not book
content." That was wrong. Rendering page 79 of the scan directly
(the OCR text layer garbles this passage badly — lambda renders as
"A", sigma as "2" — so this wasn't visible in the extracted text)
shows Fig. 52 specifies exactly this architecture, down to the
per-class amplifiers (book's own notation: lambda_ja, lambda_jb,
lambda_jc for A-element j's three class-amplifiers) and the adders
(sigma_a, sigma_b, sigma_c). What genuinely *is* new here is only the
*training rule* — and even there, a footnote on p. 82 (also missed
on the first pass, since it sits attached to a different paragraph)
turns out to specify one: coefficients "can only be increased";
algorithm 1 increases the *shown* class's amplifiers on every step,
algorithm 2 does the same only on error. Neither book rule ever
decreases a competing class's amplifiers.

`perceptron.py` now implements all three rules —
`train_multiclass_book_algorithm_1`, `_book_algorithm_2` (both
book-literal), and `train_multiclass_ovr_symmetric` (the actual
extension: the standard symmetric multiclass-perceptron update,
increasing the true class and decreasing the wrongly-predicted one)
— run head to head:

| training rule | 1 pass | 5 passes |
|---|---|---|
| book algorithm 1 (unconditional, increase-only) | 47.8% +/- 13.6% | 47.8% +/- 13.6% |
| book algorithm 2 (error-correcting, increase-only) | 72.2% +/- 6.0% | 80.6% +/- 6.1% |
| our extension (symmetric multiclass perceptron) | 81.0% +/- 6.2% | 88.0% +/- 2.4% |

**Book algorithm 1's identical accuracy at 1 and 5 passes is not a
coincidence — it's provable, and was checked directly.** Because the
rule only ever *increases* the shown class's amplifiers, by a fixed
amount, on *every* presentation regardless of correctness, repeating
the same fixed training set E times (each in a freshly-shuffled
order) scales every class's final weight vector by exactly E — checked
directly: the 5-pass weight matrix came out to be *exactly* 5.0x the
1-pass matrix, entry for entry, and the two models' predictions on
held-out data were bit-for-bit identical. Since argmax is invariant
to a uniform positive rescaling, more epochs over the *same* data
literally cannot change book algorithm 1's decisions. Its only lever
for improvement is more/new examples, never repetition — a real
limitation of an unconditional, increase-only rule that the
two-image algorithm 1 (sec. 1, symmetric +/-) doesn't share in the
same way, since a symmetric update's *sign* pattern (not just
magnitude) is what an R-element's threshold responds to, though it's
similarly insensitive to the *order* of a fixed, unweighted stream.

**Book algorithm 2 breaks the invariance and does improve with
epochs** (72.2% -> 80.6%), because gating on the current prediction
means later passes see a partially-corrected model and only touch
the examples it's still getting wrong — the same mechanism that
makes error-correction work in the two-image case.

**Our symmetric extension outperforms both book rules at every
epoch count**, plausibly because it's the only one of the three that
ever gives a class explicit negative evidence (decreasing a
wrongly-predicted class's amplifiers) rather than relying solely on
other classes' amplifiers failing to grow as fast. Ch. 3's
dissecting-planes (~92% at a comparable training size) and Ch. 4's
potentials method (~93.6%) still edge out even our best multiclass
number (88.0%) — worth another look (theta/inputs-per-element sweep,
flagged already below) rather than assumed to be a hard ceiling.


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
- The one-vs-rest *architecture* (Fig. 52) is now implemented
  literally, and so are its own two training rules — see the
  correction under "Multiclass" in Results. `train_multiclass_ovr_
  symmetric` remains the one genuine addition in this module.
- theta (the shared A-element threshold) was fixed at 0 without any
  sweep, mirroring alpha's treatment in Ch. 4; a proper sweep is a
  natural next check, as is a sweep of inputs-per-A-element (an
  earlier interactive check found this matters more than raw
  A-element count for multiclass separation).
- The 150-presentations/class long-run check (algorithm comparison
  section above) resolved the 8-vs-9 ordering but wasn't extended to
  the easy pair or wired into `experiment.py` as a proper function —
  worth doing if a later stage wants a clean "reliability vs. training
  length, held fixed A-element budget" curve per digit pair, closer
  in spirit to Ch. 3/4's own reliability-vs-N curves than the
  fixed-length runs used here.
- The epoch-invariance proof for book algorithm 1 (Multiclass,
  Results) generalizes beyond this one experiment: any unconditional,
  increase-only update over a *fixed, finite* training set gains
  nothing from repeated passes, only from new examples. Worth keeping
  in mind if a later stage's own "more training" checks use repeated
  epochs over a small fixed set rather than genuinely new data — the
  two can look superficially similar but behave very differently
  depending on whether the update rule is error-gated.
