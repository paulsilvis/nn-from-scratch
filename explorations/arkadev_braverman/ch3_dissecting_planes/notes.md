# Chapter 3 — Dissecting Planes, tested on real handwriting

*Revised after the user supplied the actual book scan. The first
pass at this chapter (see git history) had to guess at the plane
construction; this revision checks that guess against the book's own
text and corrects it.*

## What we built

- `planes.py` — the book's Ch. 3 machinery, now checked against the
  text directly:
  - `random_separating_hyperplane` — the book's **actual** base
    construction (sec. 2, p. 54): coefficients drawn from
    `{-1, 0, +1}`, a random threshold between the two opponents'
    weighted sums.
  - `bisecting_hyperplane` — kept, but re-labeled: this is the
    deterministic (k=0) limit of sec. 4's *"improved algorithm"*,
    not the base algorithm (see "The correction" below).
  - `fit_dissecting_planes` — rebuilt as a genuinely **online**
    reconstruction of Fig. 32's flow chart: points are presented one
    at a time in random order; each new point's sign vector is
    compared only against previously-seen points; a new plane is
    drawn only when an "opponent" (a prior point of a different
    class, same sign vector) is found, looping until the new point
    has none left — matching Table VII/VIII's point 6, which needed
    two planes in a row for two simultaneous opponents.
  - `fit_parallel_variants` / `predict_parallel_variants` — sec. 4's
    "method of parallel variants": train several independent models,
    combine by majority vote.
- `experiment.py` — reliability-vs-training-set-size curves (the
  shape of Tables XV-XVIII) for both plane constructions, plus a
  parallel-variants check, on real digits. Produces
  `plots/reliability_curve.png`.

## The correction

The first pass at this chapter used the perpendicular bisector of
each opponent pair as *the* plane construction, reasoning that
"random hyperplane, only when forced" meant randomizing *which*
contradiction gets resolved while making each resolving plane as
targeted as possible. The book's own text (secs. 2 and 4) says
otherwise on both counts:

- **The base algorithm's planes are genuinely random, not targeted.**
  Sec. 2 (p. 41-42, p. 54): the machine picks random coefficients
  A_1...A_n from the set {-1, 0, +1}, forms the weighted sums of the
  two opponents' co-ordinates, and picks a random threshold between
  them. The resulting plane separates the pair, but its *orientation*
  has nothing to do with minimizing distance to either point — it's
  whatever the random coefficients happened to produce.
- **The bisector construction is real, but it's a different, later
  idea.** Sec. 4 ("Ways of Increasing the Reliability of
  Recognition", p. 56) proposes drawing planes *close to* the
  perpendicular bisector of an object and its opponent, on the
  reasoning that this hugs the true class border more closely. A
  footnote on the same page notes that taking the literal bisector
  (no randomness left at all) makes the outcome of learning "fully
  determined" for a fixed set of examples, and specifically kills the
  benefit of the parallel-variants trick, since every variant would
  then be identical. That is exactly the construction the first pass
  used — it's real, just not the book's default, and not something
  meant to be run without some remaining randomness.

Both constructions are now in `planes.py`, correctly labeled, with
`"original"` (the actual sec. 2 algorithm) as the default.

## Verification

- **XOR-style toy** (as in Ch. 2's checks): both constructions
  separate the four points perfectly and self-consistently.
  `"bisecting"` needs exactly 2 planes (the theoretical minimum);
  `"original"` needed 3 in one run — a real difference, since
  untargeted random planes aren't guaranteed to resolve a
  contradiction as efficiently as the exact bisector.
- **Table VII/VIII's "two simultaneous opponents" case**: built a
  small synthetic analogue (two points already sharing a cell, then
  a third point of a different class landing in that same cell) and
  confirmed the algorithm draws exactly two planes in response, in
  the same order the book describes for its point 6.

## Results: reliability vs. training-set size (real digits)

| train size | original acc (mean+/-std) | original planes | bisecting acc | bisecting planes |
|-----------:|---------------------------|------------------|-----------------|--------------------|
|         10 | 0.345 +/- 0.039           | 5.8 +/- 1.0     | 0.405 +/- 0.052 | 5.6 +/- 1.4       |
|         20 | 0.514 +/- 0.096           | 8.6 +/- 2.1     | 0.551 +/- 0.094 | 7.0 +/- 1.5       |
|         40 | 0.735 +/- 0.077           | 11.8 +/- 0.7    | 0.752 +/- 0.066 | 10.4 +/- 1.2      |
|         80 | 0.866 +/- 0.031           | 15.4 +/- 1.0    | 0.871 +/- 0.034 | 13.2 +/- 1.2      |
|        160 | 0.917 +/- 0.022           | 18.6 +/- 2.2    | 0.911 +/- 0.017 | 15.8 +/- 0.4      |
|        320 | 0.956 +/- 0.007           | 22.0 +/- 3.7    | 0.948 +/- 0.011 | 19.8 +/- 1.5      |
|        640 | 0.977 +/- 0.007           | 26.0 +/- 3.0    | 0.975 +/- 0.007 | 22.6 +/- 1.6      |
|       1000 | 0.986 +/- 0.005           | 30.2 +/- 3.9    | 0.984 +/- 0.002 | 26.2 +/- 1.2      |
|       1497 | 0.985 +/- 0.004           | 30.8 +/- 1.8    | 0.987 +/- 0.007 | 26.4 +/- 1.4      |

Parallel-variants check (train size 160, 7 variants):

| construction | single model | parallel variants |
|--------------|--------------|--------------------|
| original     | 0.897        | 0.917              |
| bisecting    | 0.890        | 0.917              |

## Interpretation

**The two constructions perform almost identically here** — reliability
curves overlap within noise at every training size, and `"original"`
needs only modestly more planes (~15-20% more at larger sizes) than
the "ideal" bisector. That's a genuinely interesting empirical
result, and arguably makes sense given Ch. 2's finding: in a
64-dimensional receptor space where real digit classes are already
compact with wide margins, most random {-1,0,+1} planes have a
decent chance of landing somewhere reasonable, so there's less to
gain from choosing the "best" plane deliberately. This is a real
contrast with the book's own numbers (below), where the gap between
"original" and "improved" was substantial (~76% vs ~85-89%) — worth
flagging as a place our reconstruction and the book's own experiment
genuinely diverge, not just noise.

**The book's own headline numbers, for comparison** (Ch. 3 sec. 3-4,
Tables XV/XVI, 5 numeral classes, 60-cell receptor field, ~200
training examples): original algorithm averaged **76%** reliability
across 6 variants (best variant ~80%); the "improved" (angle-limited
bisector) algorithm averaged **~85.5%** at its best angle setting
(best single variant ~90%); parallel variants pushed the original
algorithm to **88.5%** and the improved algorithm to **98.5%**. Our
real-digit numbers land noticeably higher across the board (~92% at
a comparable ~160 training points, climbing to ~98.5% with more
data) — plausibly because our receptor field is finer (64 vs 60
cells isn't the difference; more likely scikit-learn's digit corpus,
being machine-scanned handwriting rather than the book's own
hand-prepared representative sets, is simply more internally
consistent than what six variants of hand-drawn 1960s numerals would
have produced).

**Parallel variants help, and help about equally for both
constructions here** (+0.020 for `"original"`, +0.027 for
`"bisecting"`) — a smaller boost than the book's own dramatic
improvement (12.5 points for the original algorithm, 13 points for
the improved one), plausibly because our single-model reliability is
already high enough (~90%) that there's less room for independent
errors to cancel out via voting.

**Arrangement size still grows roughly logarithmically with
training-set size for both constructions** (~5-6x more planes for a
150x increase in data), reconfirming Ch. 2's compactness finding
regardless of which plane-drawing rule is used.

## Open questions / things to revisit

- The angle-parameterized general case of the "improved algorithm"
  (any k between 0 and "fully random", not just the k=0 limit) isn't
  implemented — would need a way to sample a random direction within
  angle k of a fixed vector in 64 dimensions, which is more involved
  than either construction currently here.
- "Deletion of redundant planes" and "deletion of redundant
  fragments of planes" (Ch. 3 sec. 2, Tables X-XIV) — the pruning
  stage that turns the raw sign table into a minimal separating
  surface — isn't implemented. Doesn't affect classification accuracy
  as reconstructed here (majority vote over the full, unpruned sign
  table already classifies correctly), but would matter for the
  book's "memory volume after learning" comparisons (Table XV) if we
  wanted to reproduce those too.
- Per-class reliability (does digit 8 need disproportionately more
  planes, matching its Ch. 2 compactness deficit?) is still an open
  question from the previous revision of these notes.
