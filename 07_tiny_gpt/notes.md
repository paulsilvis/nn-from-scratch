# Stage 7: the tiny GPT capstone

Every prior stage's work assembled into one model: stage 4's
softmax/cross-entropy output head, stage 6's self-attention
(extended here with causal masking), and two genuinely new pieces -
layer normalization and residual connections - needed to make a
real, trainable transformer.

## What's new this stage

**Causal masking**: added as an additive mask (0 or -inf) to the
raw attention scores before softmax, so a position can never attend
to a future position (softmax of -inf is exactly 0 weight there).
No change to the backward derivation at all - attention_backward's
formulas only ever look at A (the post-softmax weights), which
already reflects the masking correctly regardless of how S was
constructed.

**Residual connections**: `x -> x + sublayer(x)` around both the
attention and feedforward sublayers. The backward consequence is
what makes deep stacks trainable: `d(output)/dx = I + d(sublayer)/dx`,
so gradient always has a direct, unobstructed path back through
every layer (the identity term), no matter how deep the stack -
this is the specific mechanism that keeps stacked transformer
layers from suffering the kind of saturation/vanishing-gradient
problems explored in stage 3.

**Layer normalization**: per-position rescaling to zero mean, unit
variance, then a learned scale (gamma) and shift (beta). Backward
derived and verified (see layers.py's docstring for the full
formula) - not re-derived step-by-step in conversation this stage,
verified against finite differences instead, same discipline as
every other stage.

**Position-wise feedforward**: the same 2-layer dense network (ReLU
hidden activation, same as stage 5's conv layer) applied
identically at every sequence position - weight sharing again, the
same idea that's driven every architecture since convolution.

## Verification, before any training

Every new component gradient-checked individually first
(`gradient_check.py`: layer norm's dX/dgamma/dbeta, the
feedforward's 5 gradients, and masked attention's dX - 9 checks,
all passing to ~10 significant figures), then the FULLY ASSEMBLED
model checked end-to-end (`full_model_check.py`) - because
individually-correct components can still be wired together wrong.

**They were.** The full-model check caught two real bugs that the
isolated checks couldn't see:

1. `W_O`'s gradient (the attention output projection) was computed
   with `tensordot` arguments in the wrong order, producing a
   transposed matrix - shape-compatible with dumb luck in some
   contexts, numerically wrong everywhere.
2. A double-counted batch average: `attention_backward` (unchanged
   from stage 6) divides its weight gradients by batch size
   internally, which was the right convention in stage 6's isolated
   use, but here the upstream gradient was already pre-averaged by
   `batch * seq_len` before reaching it - applying both averages
   silently shrank every W_Q/W_K/W_V gradient by an extra factor of
   the batch size.

Both were only visible once every piece was wired together and
checked as a whole - exactly the point of doing this check at all,
and consistent with stage 5's einsum bug: isolated correctness does
not imply composed correctness.

## Training

Character-level, on the first 100,000 characters of Andrej
Karpathy's "tiny Shakespeare" dataset (public domain text; corpus
itself fetched from `raw.githubusercontent.com/karpathy/char-rnn`,
one of this sandbox's allowed network domains). Architecture:
d_model=64, 2 transformer blocks, single-head attention (multi-head
- running several attention heads in parallel on slices of d_model
and concatenating - is a real, mechanical extension not implemented
here; flagged rather than silently assumed), context length 48
characters, 110,525 parameters total. Plain minibatch gradient
descent throughout (no Adam, no momentum), matching every prior
stage's optimizer - a real, honest limitation: transformers are
known to be harder to train with vanilla SGD than with adaptive
optimizers, and it shows in the loss curve's bumpiness around steps
1800-3000 even as the overall trend keeps improving.

4000 steps, batch_size=32, learning_rate=0.3: loss fell from 4.79
(worse than the uniform-random baseline of ln(61)=4.11 at
initialization) to 1.45 nats/char. See `plots/training_loss.png`.

## Generated sample (temperature=0.8, seeded "ROMEO:")

    ROMEO:
    Not you insul, may this be besome aborn, they do can the care speak
    There finess sable bear, but at; as the weep asseners
    The bards he which will me to pardonable
    Then were.

    For with read we their good Marcius,
    Which'd reves peaces, upon your voices:
    I spake us well a bediments
    For make unsemty th

Not remotely grammatical - this is an honest limitation of a tiny,
briefly-trained, single-head, plain-SGD model, not a claim of
success beyond what happened. What IS real: correct speaker-header
formatting ("ROMEO:" followed by a colon and newline, exactly the
corpus's convention), plausible English letter/word statistics,
verse-like line lengths, and even reproducing "Marcius" - an actual
name from the training text - in a novel sentence. The model
learned real structure about the character distribution of English
and this corpus's formatting conventions; it did not learn syntax
or meaning at this scale and training budget, and it would be
dishonest to imply otherwise.

## Where this project started vs. where it ends

Stage 1: a single perceptron, providably incapable of solving XOR.
Stage 7: a from-scratch transformer, gradient-checked component by
component and as a whole, generating structured (if not coherent)
English text - built entirely from the chain rule, one honestly-
verified derivation at a time, with no autodiff and no framework.
