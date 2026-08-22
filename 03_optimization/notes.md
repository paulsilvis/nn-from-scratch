# Stage 3: the loss surface and gradient descent dynamics

## The slice

The full loss L is a function of all 9 network parameters -
impossible to plot directly. Here, W1, b1, and b2 are frozen at a
converged stage-2 XOR solution, and only the two output-layer
weights (w2_1, w2_2) are left free. This gives a genuine 2D slice
through the true 9D surface - a real mathematical object, but only
one slice of many possible ones (freezing a different 7 parameters
would give a different picture).

Converged solution used: W2 = [-11.24, 10.54], b2 = -4.92 (W1, b1
from the same run, held fixed throughout this stage).

See `plots/loss_surface_3d.png` for the raw shape: a shallow trough
running diagonally, with a steep wall rising on one side. Not a
simple bowl - there's real curvature variation across the slice.

## Gradient descent: a learning rate that works

Starting from w2 = (-6, 6), learning_rate = 2.0, 150 steps:
final w2 = [-6.10, 8.01], loss = 0.0023.

See `plots/descent_converging.png` - the path takes a short, direct
route into the dark (low-loss) trough and stops there.

## Gradient descent: a learning rate that doesn't - but not how
## I expected

Original plan was to demonstrate classic oscillating/runaway
divergence with a large learning rate. That's not what happens here,
and the reason is worth stating rather than glossing over: because
the output activation is a sigmoid, both `yhat` and every hidden
activation are bounded in (0, 1), so the gradient itself
(proportional to yhat*(1-yhat) and a1*(1-a1)) shrinks toward zero as
weights grow large and the units saturate. A "too-large" learning
rate doesn't cause unbounded blowup - the surface itself won't let
it. What happens instead: a single oversized step (learning_rate =
300.0, from the same start (-6, 6)) launches w2_2 from 6 to about
17, straight out of the region where the surface has meaningful
curvature. Once there, the gradient is nearly zero (deep in
saturation), so the parameters essentially freeze in place - stuck
at loss ~0.125, permanently, 60 steps in. See
`plots/descent_diverging.png`: the trajectory visibly exits the top
of the computed grid.

Notable: 0.125 is suspiciously close to the flat plateau loss value
seen in stage 2's XOR training curve (`02_backprop/plots/xor_loss.png`,
which sat near 0.125 for several hundred epochs before escaping).
That's not a coincidence worth treating lightly - it's evidence that
the plateau observed there is this exact same kind of saturation
trap, one the network eventually escaped with a well-behaved
learning rate but could plausibly have gotten stuck in permanently
with a worse one. This connects the two stages' observations rather
than treating them as separate curiosities.

## Design notes

- `loss_surface.py` keeps `gradient()` structurally identical to
  stage 2's `delta2` formula, just averaged over the 4 fixed-A1
  training examples instead of computed per-example - same math,
  restated for a frozen-hidden-layer slice.
- `viz/replay.py` gained `plot_surface_3d` (static 3D view) and
  `plot_contour_with_paths` (top-down contour with one or more
  gradient descent trajectories overlaid, start marked with a
  circle, end with an X).
- Chose NOT to force a fabricated oscillating-divergence example
  once the real behavior turned out to be more informative -
  reporting the actual saturation-trap mechanism rather than a
  generic textbook picture that doesn't apply to this architecture.
