# Stage 4: MNIST, and the first real multi-class network

## Data: this is genuine MNIST, not a substitute

This sandbox's network access is restricted to an allowlist (package
registries, GitHub domains, a few others) - no general internet, no
openml.org, no yann.lecun.com. Real MNIST turned out to still be
reachable: Michael Nielsen's `neural-networks-and-deep-learning`
repo (one of the resources flagged as a good fit for this project at
kickoff) hosts a pickled copy at `data/mnist.pkl.gz`, servable via
`raw.githubusercontent.com`, which is on the allowed list. Verified
structure before use: 50000/10000/10000 train/val/test split, 784
flattened pixel values, already normalized to [0,1], labels 0-9.
This is the standard MNIST split, not the smaller `load_digits`
corpus used throughout the Arkadev/Braverman detour.

## What generalizes from stage 2

Two real changes to `TwoLayerNetwork`, both required by the jump
from 2-input-XOR to 784-input-10-class:

1. **Arbitrary hidden size** - stage 2 fixed it at 2 (just enough for
   XOR); MNIST needs real capacity.
2. **Softmax + cross-entropy output**, replacing the single sigmoid.
   Derived by hand (see conversation): writing
   `L = -log(p_y) = -z_y + log(sum_j exp(z_j))` and differentiating
   w.r.t. an arbitrary logit z_k gives, in both cases (k=y and
   k != y), the single clean result

       delta2_k = p_k - t_k

   which is the K-class generalization of stage 2's
   `delta2 = -(y-yhat)*yhat*(1-yhat)` (that formula is this one's
   K=2 special case). Hidden-layer delta keeps the same recursive
   structure as stage 2, just summed over all K outgoing paths
   instead of one: `delta1_j = (sum_k delta2_k * W2[k,j]) * a1_j *
   (1-a1_j)`.

## What's new: minibatch SGD

Stage 2 used full-batch gradient descent (averaging over all 4 XOR
examples every update) - fine for 4 points, impossible for 50000.
`MLP.fit` shuffles the training set each epoch and updates on
minibatches of 128, which stage 2's notes.md flagged as a choice
"stage 3 would look at properly" - stage 3 ended up being about the
loss surface/learning rate instead, so this is where batch size
actually gets addressed.

## Results: comparing 3 hidden-layer sizes

15 epochs, batch_size=128, learning_rate=0.5, same seed, all other
hyperparameters held fixed.

| Hidden size | Parameters | Train time | Train acc | Val acc | Test acc |
|-------------|-----------|------------|-----------|---------|----------|
| 32          | 25,450    | 9.6s       | 0.9695    | 0.9613  | 0.9603   |
| 128         | 101,770   | 20.7s      | 0.9761    | 0.9701  | 0.9674   |
| 512         | 407,050   | 64.9s      | 0.9741    | 0.9689  | 0.9667   |

See `plots/hidden_size_comparison.png`. Notable finding, reported
plainly rather than smoothed over: hidden=128 and hidden=512 track
each other almost exactly throughout training, and 128 finishes
*marginally ahead* of 512 on both validation and test accuracy,
despite having 4x fewer parameters and roughly a third of the
training time. hidden=32 lags a bit behind both.

This isn't "bigger networks don't help" as a general claim - it's
that at this specific learning rate and epoch budget, 512's extra
capacity isn't being exploited. Plausible reasons, not confirmed
here: (a) a fixed learning_rate=0.5 may be poorly scaled for a much
wider layer, since gradient magnitudes and useful step sizes
typically depend on layer width; (b) 15 epochs may simply not be
enough for the larger model to make use of its added capacity,
whereas the smaller model saturates faster. Worth a real follow-up
(e.g. more epochs or a width-scaled learning rate for the 512 case)
before drawing a firm conclusion - flagged here rather than resolved.

## Design notes

- `MLP.__init__` scales initial weights by `1/sqrt(fan_in)` per
  layer (rather than stage 2's fixed unit-variance init), since with
  784 inputs, unit-variance weights would produce pre-activations
  with far too much variance, saturating the sigmoid before training
  even starts.
- `softmax()` subtracts the row max before exponentiating - standard
  numerical stability trick, doesn't change the output (constant
  shift cancels in the ratio) but prevents overflow.
- Avoided the black/flake8 E203 conflict differently than the
  established `.reshape()` workaround this time, since batch
  indexing genuinely needs a slice: computed `end = start +
  batch_size` as its own variable so the slice itself
  (`order[start:end]`) has no embedded expression, which black
  and flake8 both accept.
