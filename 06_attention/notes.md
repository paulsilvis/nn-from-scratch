# Stage 6: self-attention, derived and verified from scratch

## The mechanism

Every position produces a query, key, and value vector via three
shared weight matrices (weight sharing again, same idea as
convolution - just applied across sequence positions instead of
image positions):

    Q = X @ W_Q   K = X @ W_K   V = X @ W_V
    S = Q @ K.T / sqrt(d_k)        (scaled dot-product scores)
    A = softmax_rows(S)
    O = A @ V

The 1/sqrt(d_k) scaling exists for a reason directly connected to
an earlier finding in this project: as d_k grows, raw dot products
grow too, pushing S into the region where softmax saturates - the
same "confident and uncorrectable" trap discussed for the sigmoid in
stage 3's optimization notes. Scaling keeps the scores in a range
where softmax still has a live gradient.

## Backward pass, derived and verified

    dL/dV = A.T @ G
    dL/dA = G @ V.T
    dL/dS[i] = a_i * (g_i - (a_i . g_i))   -- general softmax
        backward rule; NOT the "p - t" shortcut from stage 4, since
        there's no cross-entropy paired directly with this softmax
        to collapse it to that simpler form
    dL/dQ = dL/dS @ K / sqrt(d_k)
    dL/dK = (dL/dS).T @ Q / sqrt(d_k)
    dL/dW_Q = X.T @ dL/dQ   (same for W_K, W_V)
    dL/dX   = dL/dQ @ W_Q.T + dL/dK @ W_K.T + dL/dV @ W_V.T
        (X feeds all three projections, so its gradient sums all
        three paths)

Verified against finite differences on a tiny random example
(`gradient_check.py`) before use: dX, dW_Q, dW_K, dW_V all matched
to ~9-10 significant figures.

## The task: associative recall

Designed specifically to isolate what attention adds. Each example
is n_pairs=4 (key, value) tokens in freshly randomized order, plus
one query token naming a key. Label = the value paired with that
key. Because the matching position changes every single example,
there is no fixed position a plain fully-connected layer could learn
to always look at - the task requires genuine content-based
addressing, not positional memorization. Chance accuracy = 1/10
(10 possible values).

## Results

20,000 training examples, 30 epochs, batch_size=64,
learning_rate=0.5, both models trained from the same random seed
and identical data.

| Model             | Params | Val acc | Test acc |
|-------------------|--------|---------|----------|
| Attention (d_k=16) | 986    | 1.0000  | 1.0000   |
| Plain MLP (hidden=64) | 6,154 | 0.3840 | 0.3845 |

The attention model solves the task essentially perfectly, with
6x fewer parameters than the MLP baseline. This mirrors stage 1's
XOR story directly: a fixed-position architecture has a genuine,
provable-in-spirit limitation here (no mechanism for "look up by
content, regardless of where"), and the new mechanism (attention)
solves exactly what the old one structurally cannot.

Worth being honest about the MLP's 38.4%, not just calling it
"chance failure": it's meaningfully above the 10% chance floor,
consistently across epochs (see `plots/attention_vs_mlp.png`) rather
than noisy scatter around 10%. Plausible explanation, not confirmed
here: with only 6 possible keys and 10 values, the MLP can partially
exploit marginal statistics in the fixed 20,000-example training set
(e.g. some value being slightly more frequent, or weak correlations
between which keys tend to co-occur) without ever performing the
actual lookup. It plateaus rather than continuing to improve,
consistent with having exhausted whatever shortcut is available
rather than being on a slow path to the real solution.

## Design notes

- `models.py`'s output head (softmax + cross-entropy,
  `delta = p - t`) is unchanged from stage 4 - the only new
  derivation needed this stage was attention itself.
- The attention model reads its prediction from the output at a
  fixed, known query position (the last position); building a
  gradient mask that is zero everywhere except that position is a
  clean way to route the loss's gradient only into the position
  that mattered, without needing a separate "select one position"
  operation with its own backward pass.
- Chose a purpose-built synthetic task over reusing MNIST, since
  MNIST has no meaningful sequential structure for attention to
  exploit - flagged explicitly rather than forcing a fit.
