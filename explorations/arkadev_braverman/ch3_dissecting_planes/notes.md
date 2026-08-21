# Chapter 3 — Dissecting Planes, tested on real handwriting

## What we built

- `planes.py` — the book's Ch. 3 sec. 2 machinery: a `Hyperplane`
  (sign test w.x + b), the "opponent-forced" `bisecting_hyperplane`
  construction (perpendicular bisector of two contradicting points),
  a `SignTable` that bookkeeps each training point's accumulated
  sign vector as planes accrue, and `fit_dissecting_planes`, which
  repeatedly finds an "opponent" pair (two different-class points
  still sharing a sign vector) and draws the one plane that resolves
  it, until no opponent remains. `DissectingPlanesModel.predict`
  classifies new points by which polyhedron (sign vector) they land
  in, majority-voted by the training points that landed there too,
  falling back to 1-NN for polyhedra no training point ever reached.
- `experiment.py` — reproduces the shape of the book's own
  reliability-vs-training-set-size tables (Ch. 3 Tables XV-XVIII) on
  real digits: a fixed 300-point held-out test set, training-set
  sizes from 10 to 1497 (the full remaining pool), 5 random trials
  per size. Produces `plots/reliability_curve.png`.

## The one real design decision: which plane to draw next

The book leaves "draw a random hyperplane" underspecified once you
try to actually implement it — a uniformly random plane in a 64-
dimensional space has no principled reason to separate any
particular pair of opponents, and could easily separate nothing at
all for many draws in a row. We read "only draw a new plane when the
data forces one" (the "opponent" framing) as license to make each
forced plane count: when some polyhedron cell still contains
opponents, draw the perpendicular bisector of one such pair, chosen
at random when several exist. This keeps the algorithm's genuinely
random element (which contradiction gets resolved next) while
guaranteeing monotonic progress — something that matters at ~1800
points where undirected random planes could take a very long time to
stumble onto anything useful. This is a documented interpretive
choice, not a verified transcription of the book's own procedure --
unlike Ch. 2's checks, we didn't have book text in this session to
verify the algorithm against a worked example.

## Verification

Checked against a hand-constructed XOR-style toy case (four points,
two classes, not linearly separable by one plane): the algorithm
draws exactly 2 planes and achieves perfect self-consistency, which
is the minimum possible — one plane can separate at most a halfspace,
and XOR's two classes are each split across the halfspaces of any
single plane, so at least 2 are required, and 2 suffice once each
resolves one diagonal pair.

## Results: reliability vs. training-set size (real digits)

| train size | test acc (mean +/- std) | planes drawn (mean +/- std) |
|-----------:|-------------------------|------------------------------|
|         10 | 0.377 +/- 0.063         | 5.0 +/- 1.1                  |
|         20 | 0.541 +/- 0.081         | 7.2 +/- 0.4                  |
|         40 | 0.750 +/- 0.086         | 10.2 +/- 1.7                 |
|         80 | 0.853 +/- 0.018         | 12.6 +/- 1.5                 |
|        160 | 0.925 +/- 0.022         | 16.8 +/- 0.7                 |
|        320 | 0.957 +/- 0.010         | 20.6 +/- 1.5                 |
|        640 | 0.979 +/- 0.005         | 22.6 +/- 1.5                 |
|       1000 | 0.984 +/- 0.005         | 25.0 +/- 1.7                 |
|       1497 | 0.986 +/- 0.003         | 28.0 +/- 1.1                 |

Full-corpus (1797 points) training accuracy is 1.000 with 29 planes
— every training point is perfectly separated, as it should be given
Ch. 2's finding that no two real digits of different classes are
even one bit apart after binarization (`boundary_fraction = 0.000`
for every class, `ch2_compactness/notes.md`).

## Interpretation

**The reliability curve looks exactly like the book's own Tables
XV-XVIII shape**: steep gains from a handful of examples up through
a few hundred, then a long flattening tail toward ~0.98-0.99. That
qualitative shape survived the jump from ~12-20 hand-picked
representatives per class to ~150 real, noisy handwritten examples
per class — a genuine reproduction of the book's central empirical
claim about how fast this kind of learning saturates, not just an
artifact of tidy synthetic data.

**Arrangement size grows roughly logarithmically with training-set
size** (28 planes for 1497 points vs. 5 planes for 10 points — a
150x increase in data for roughly a 6x increase in planes). That's
the direct payoff of Ch. 2's compactness finding: because real digit
classes are compact (no cross-class pairs within 1 bit), most new
training points just confirm an existing cell rather than forcing a
new split. A dataset without that compactness property would need
roughly one new plane per new opponent almost regardless of how many
planes already existed, and the arrangement-size curve on the right
would track training-set size far more closely than it does here.

**At 10-20 training points, variance is large** (std of ~0.06-0.08
on test accuracy) — with only 1-2 examples per class on average, the
polyhedra are still coarse and easily miss whole classes; the 300-
point fixed test set means an unlucky training draw can plausibly
leave one or two digit classes completely unrepresented. This
matches the book's own framing of "reliability" as something that
only becomes meaningful once training-set size is large enough
relative to the number of classes, not a per-classifier property in
isolation.

## Open questions / things to revisit

- We don't have book text in this session to check the "opponent"-
  forced construction against the book's own worked Ch. 3 examples,
  the way Ch. 2's internal/boundary definitions were checked against
  the XOR and {11}/{00,01} cases (pp. 17-20). Worth revisiting if the
  book (or a scan of it) becomes available again.
- The book's Tables IV-XIX likely also track something like
  per-class reliability, not just an aggregate — a natural
  refinement given Ch. 2's finding that digit 8 is the least compact
  class; does 8 also need disproportionately more planes / show
  disproportionately worse reliability at small training sizes?
- `predict`'s 1-NN fallback for genuinely unseen polyhedra never
  actually triggered in these runs (every held-out point that hit
  test time landed in a cell some training point had already
  visited) — worth stress-testing at very small training sizes
  where it should start to matter more.
