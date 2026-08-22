# Stage 2: backprop, derived by hand, on a two-layer network

## Architecture

2 inputs -> 2 hidden units (sigmoid) -> 1 output (sigmoid). See the
diagram from the conversation for the full picture: every edge is
one of 6 weights, every computing unit (h1, h2, yhat) has its own
bias, sigma applied at both layers.

    z1[j] = W1[j] . x + b1[j]        a1[j] = sigmoid(z1[j])
    z2    = W2 . a1 + b2             yhat  = sigmoid(z2)
    L     = 0.5 * (y - yhat)^2

## The derivation, in summary

Sigmoid derivative, derived via the quotient rule and a rewrite of
e^-z as (1+e^-z) - 1:

    sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

Output layer error signal (chain rule through 3 links: w -> z2 ->
yhat -> L):

    delta2 = dL/dz2 = -(y - yhat) * yhat * (1 - yhat)
    dL/dW2[j] = delta2 * a1[j]        dL/db2 = delta2

Hidden layer error signal (chain rule through the *already computed*
delta2, propagated backward through W2):

    delta1[j] = delta2 * W2[j] * a1[j] * (1 - a1[j])
    dL/dW1[j][i] = delta1[j] * x[i]   dL/db1[j] = delta1[j]

This recursive delta pattern - each layer's error signal built from
the next layer's error signal times the connecting weight times the
local activation derivative - is the entire backward pass. Update
rule uses a MINUS sign (descend the gradient), the opposite
convention from Stage 1's perceptron rule (which added eta*y*x to
move toward correct classification).

## Results

Full-batch gradient descent, 10000 epochs, learning_rate=5.0, seed=0,
sigmoid activations throughout.

| Gate | Final loss | All correct? |
|------|-----------|---------------|
| AND  | 0.000016  | Yes |
| OR   | 0.000014  | Yes |
| XOR  | 0.000054  | Yes |

**XOR succeeds where Stage 1's single perceptron provably could
not.** See `plots/xor_region.png`: the decision region is a
diagonal *band* bounded by two roughly-parallel lines, not a single
line - exactly the mechanism promised by adding a hidden layer. Each
hidden unit contributes one line; the output layer combines them
("true only between these two lines"). Compare to `plots/and_region.png`,
which still uses a single line - the network didn't invent
unneeded complexity for a linearly-separable problem.

`plots/xor_loss.png` shows an extended flat plateau near loss=0.125
for several hundred epochs before a sharp collapse to near zero -
the network sitting near a saddle point before finding a useful
descent direction. Flagged as a preview of Stage 3 (optimization
landscape), not investigated further here.

## Design notes

- `forward`/`backward` in `network.py` operate on one example at a
  time, matching the hand derivation's notation exactly (no hidden
  vectorization tricks in the core math). `predict_batch` is a
  separate, clearly-commented vectorized restatement used only for
  plotting decision regions over a grid - not a different algorithm.
- Full-batch gradient descent (average gradient over all 4 examples,
  one update per epoch) rather than per-example (online) updates,
  since the whole dataset is 4 points and fits trivially in memory.
  Stage 3 will look at this choice properly (batch vs. online vs.
  minibatch).
- `viz/replay.py` gained `plot_loss_curve` (continuous loss, vs.
  Stage 1's discrete `plot_error_curve`) and `plot_decision_region_2d`
  (filled contour + 0.5-threshold contour line, since the boundary is
  no longer guaranteed to be straight).
