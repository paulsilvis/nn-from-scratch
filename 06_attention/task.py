"""Synthetic associative-recall task: isolates why content-based
addressing (attention) beats fixed-position weights (a plain MLP).

Each example is a sequence of n_pairs (key, value) tokens in random
order, followed by one query token naming a key that appeared
somewhere in the sequence. The label is the value that was paired
with that key. Because the pairs are freshly shuffled every example,
the position of the matching key changes every time - there is no
fixed position a plain MLP could learn to always look at.

Token representation: each position's vector is
(key_one_hot, value_one_hot) concatenated, dimension
n_keys + n_values. The query token uses its key's one-hot but an
all-zero value part (it must recall the value, not already carry
it) - a distinct "is this a query" bit is appended so key and query
positions carrying the same key one-hot remain distinguishable.
"""

import numpy as np


def make_batch(n_examples, n_pairs, n_keys, n_values, rng):
    """Returns (X, y): X shape
    (n_examples, n_pairs + 1, n_keys + n_values + 1), y shape
    (n_examples,) with the correct value index for the query.
    """
    d = n_keys + n_values + 1
    seq_len = n_pairs + 1
    X = np.zeros((n_examples, seq_len, d))
    y = np.zeros(n_examples, dtype=int)

    for i in range(n_examples):
        keys = rng.choice(n_keys, size=n_pairs, replace=False)
        values = rng.integers(0, n_values, size=n_pairs)

        for t in range(n_pairs):
            X[i, t, keys[t]] = 1.0
            X[i, t, n_keys + values[t]] = 1.0

        query_idx = rng.integers(0, n_pairs)
        query_key = keys[query_idx]
        X[i, n_pairs, query_key] = 1.0
        X[i, n_pairs, -1] = 1.0  # "this is the query" flag
        y[i] = values[query_idx]

    return X, y
