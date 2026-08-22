# Stage 1: single perceptron, AND / OR / XOR

## The model

A single linear threshold unit:

    f(x) = sign(w . x + b)

where `w` is the weight vector, `b` the bias, and `sign(0)` is
defined as +1 by convention (arbitrary, but must be pinned down —
see conversation for why it can matter with binary 0/1 inputs).

## The learning rule

Perceptron learning rule (not gradient descent, though it can be
read as gradient descent on a hinge-style loss): for each
misclassified point `(x, y)` with `y in {-1, +1}`,

    w <- w + eta * y * x
    b <- b + eta * y

## Geometric argument for why XOR must fail

AND and OR each have three points on one side of the unit square
and one on the other (or vice versa) — a single straight line
easily separates them. XOR's positive points, (0,1) and (1,0), sit
on one diagonal of the unit square; its negative points, (0,0) and
(1,1), sit on the other diagonal. No straight line separates one
diagonal's pair from the other's — proved algebraically by deriving
a contradiction from the four required inequalities (b<0,
w1+w2+b<0, w2+b>0, w1+b>0: adding the last two gives
w1+w2+2b>0, which combined with b<0 forces w1+w2+b>0,
contradicting the second inequality).

## Results

| Gate | Converged? | Epochs | Final weights | Final bias |
|------|-----------|--------|----------------|------------|
| AND  | Yes | 7  | [2.14, 1.77] | -3.00 |
| OR   | Yes | 5  | [1.14, 1.77] | -1.00 |
| XOR  | No  | 50 (max, never converges) | oscillates | oscillates |

See `plots/and_boundary.png`, `plots/or_boundary.png`,
`plots/xor_boundary.png` for the evolution of the decision line
across training (faint = early, solid black = final), and the
corresponding `*_errors.png` for misclassification count per epoch.

Key observation: AND and OR settle into a single stable boundary
and stay there (errors -> 0 and remain 0). XOR never stabilizes —
after a brief transient it locks into a fixed cycle misclassifying
all 4 points every single epoch, forever. This isn't "hasn't
converged yet" — it's provably incapable of converging, because no
separating hyperplane exists for this pattern. This is the
Minsky-Papert (1969) observation that stalled single-layer
perceptron research; Stage 2 (backprop, multi-layer networks) exists
specifically because a second layer can combine two lines into a
boundary that wraps around XOR's diagonal pattern.

## Design notes

- `viz/snapshot.py` and `viz/replay.py` are now real (previously
  empty placeholders from project kickoff). `Recorder` logs a
  `Snapshot` after every weight update; `replay.py` turns a
  recorder's snapshots into boundary-evolution and error-curve
  plots. This is meant to be the shared convention for all later
  stages, not perceptron-specific.
- `sign(0) = +1` convention lives in `Perceptron.predict` as
  `>= 0`.
