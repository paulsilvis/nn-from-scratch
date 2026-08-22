# Stage 5: convolutional networks, from scratch

## What convolution buys you

A fully-connected layer treats every pixel as an independent,
unrelated input. Convolution instead slides one small kernel (a
handful of learnable weights) across every position of the image,
reusing the exact same weights everywhere - "weight sharing." A
kernel that learns to detect a diagonal edge detects it wherever it
appears (translation equivariance), something a fully-connected
layer doesn't get for free.

## Forward pass

    Z[b,i,j,f] = sum_{u,v} K[f,u,v] * X[b,i+u,j+v] + b[f]

This is cross-correlation (no kernel flip), which is what every
practical "conv layer" actually computes, despite the name.

## Backward pass, derived by hand

Because a single kernel weight K[u,v] appears in the formula for
*every* output position, its gradient sums contributions from all
of them - the direct consequence of weight sharing:

    dL/dK[u,v] = sum_{i,j} G[i,j] * X[i+u,j+v]   (a correlation,
                                                    same op as forward)
    dL/db      = sum_{i,j} G[i,j]
    dL/dX[p,q] = sum_{u,v} G[p-u,q-v] * K[u,v]   (a genuine
                                                    convolution -
                                                    kernel flipped)

where G = dL/dZ is the upstream gradient. The dX formula's flipped
kernel is the one place true convolution (as opposed to
cross-correlation) shows up in this whole stage - only when
differentiating the forward pass to reach the input.

**Verified against finite differences before use**, not just
trusted (`gradient_check.py`): dK, db, and dX all matched numeric
gradients to ~10 significant figures on a random tiny example. This
caught a real bug during development - `sliding_window_view`
appends its window dimensions at the very end of the array, after
any untouched axes, not interleaved where they might naively be
expected; the dX computation's einsum indices were wrong until the
gradient check caught the mismatch.

## Max-pooling

No learnable weights - downsamples by taking the max over small
(2x2) windows. Backward: the entire upstream gradient at a pooled
position routes to whichever input position achieved that max (ties
split evenly, preserving the gradient sum exactly).

## Architecture and results

conv (8 filters, 3x3) -> ReLU -> max-pool (2x2) -> flatten (1352) ->
dense (128, sigmoid) -> dense (10, softmax). Same hidden width (128)
and same softmax/cross-entropy output as stage 4's MLP, so the
comparison isolates what the conv+pool feature extractor adds.

8 epochs, batch_size=128, learning_rate=0.5, real MNIST.

| Model                  | Params  | Epochs | Val acc | Test acc |
|-------------------------|---------|--------|---------|----------|
| Stage 4 MLP (hidden=128) | 101,770 | 15     | 0.9701  | 0.9674   |
| Stage 5 CNN              | 174,554 | 8      | 0.9774  | 0.9771   |

The CNN wins outright - fewer epochs, better accuracy - but the
comparison isn't perfectly apples-to-apples and it's worth saying
so: the CNN has *more* total parameters (174,554 vs 101,770),
because the flattened pooled feature count (1352) exceeds the raw
pixel count (784) that feeds stage 4's first dense layer. This
particular small-filter-count architecture isn't demonstrating
"fewer parameters for the same accuracy" - it's demonstrating that
spatial structure (the inductive bias of weight sharing +
locality), not raw parameter count, is what's driving the
improvement. A fairer parameter-matched comparison is a natural
follow-up, not done here.

## A real, concrete confirmation of the memory-scaling finding

While running the final full-dataset evaluation, the process was
silently OOM-killed on this sandbox. Root cause: `accuracy()`'s
first implementation called `predict()` on the *entire* input at
once - for the full 50,000-image training set, that materializes a
`(50000, 26, 26, 3, 3)` convolution-patches array in one shot
(~2.4 GB), which exceeded this machine's available memory.

This is exactly the failure mode predicted (and merely
extrapolated, not witnessed) in stage 4's `memory_scaling.py`
addendum - except worse here, because convolution patches expand
data by a factor of k^2 per filter, not just by hidden-layer width.
Fixed by batching `accuracy()` internally (1000 images at a time)
rather than working around it - the same fix flagged as the
"real" solution in that earlier addendum, now actually applied
rather than just recommended.

## Design notes

- `conv.py` uses `numpy.lib.stride_tricks.sliding_window_view` to
  extract all patches in one vectorized call rather than a
  quadruple-nested Python loop - the underlying math is unchanged
  from the derivation, just reshaped for numpy's BLAS backend to
  execute. Worth remembering its exact output shape convention
  (window dims appended at the very end, after any untouched axes)
  - getting this wrong caused the dX bug the gradient check caught.
- `cnn.py`'s dense-layer backward pass (softmax + cross-entropy,
  `delta2 = p - t`) is copied unchanged from stage 4 - no new math
  needed there, only the conv/pool layers feeding into it are new.
- Training itself (8 epochs, full 50,000-image dataset, batched
  minibatch updates) took ~306s on this sandbox's single core - the
  per-epoch training cost, not the (now-fixed) accuracy check, is
  the actual time bottleneck, consistent with stage 4's finding that
  compute time matters more than memory for architectures at this
  scale.
