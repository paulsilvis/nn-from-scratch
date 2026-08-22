#!/usr/bin/env python3
"""Stage 4 addendum: where does this network's memory scheme break?

Measures actual peak process memory (not a theoretical estimate) as
a function of hidden layer size, by running a single realistic
training step + accuracy check in an isolated subprocess for each
size (isolated so each reading is a clean peak, not contaminated by
the previous size's allocations). Fits a line to the results and
extrapolates to find the hidden size at which a given RAM ceiling
would be exceeded.

Run directly (not via the usual experiment.py pattern) since each
data point needs its own fresh process:

    for h in 128 512 1024 2048 3000; do
        python3 memory_scaling.py --hidden $h
    done

then feed the printed numbers to `fit_and_extrapolate()` below, or
just rerun with --fit after collecting a few readings by hand.
"""

import argparse
import gzip
import pickle
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mnist_net import MLP  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "mnist.pkl.gz"


def measure_one(hidden):
    """Peak RSS (GB) for one realistic step: a minibatch training
    update, then the same full-dataset accuracy() calls fit() makes
    every epoch - the actual memory-dominant operation, since it
    forms a (n_examples, hidden) activation array in one shot.
    """
    with gzip.open(DATA_PATH, "rb") as f:
        train, val, _test = pickle.load(f, encoding="latin1")
    X_train, y_train = train
    X_val, y_val = val

    net = MLP(n_inputs=784, n_hidden=hidden, n_classes=10, seed=0)
    onehot = np.eye(10)[y_train[:128]]
    net.step(X_train[:128], onehot)
    net.accuracy(X_train, y_train)
    net.accuracy(X_val, y_val)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_kb / 1e6


def fit_and_extrapolate(hidden_sizes, peak_gb, ceilings=(24, 28, 32)):
    """Least-squares line through measured (hidden, peak_gb) points,
    then solve for the hidden size at which each RAM ceiling is hit.
    """
    h = np.array(hidden_sizes, dtype=float)
    y = np.array(peak_gb, dtype=float)
    a_matrix = np.vstack([h, np.ones_like(h)]).T
    slope, intercept = np.linalg.lstsq(a_matrix, y, rcond=None)[0]

    print(f"peak_GB = {intercept:.4f} + {slope:.6f} * hidden_size")
    for ceiling in ceilings:
        h_limit = (ceiling - intercept) / slope
        print(f"  crosses {ceiling} GB at hidden_size = {h_limit:,.0f}")
    return slope, intercept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, help="hidden size to test")
    parser.add_argument(
        "--fit", action="store_true", help="fit stored measurements"
    )
    args = parser.parse_args()

    if args.hidden is not None:
        peak = measure_one(args.hidden)
        print(f"H={args.hidden}: peak_rss={peak:.3f} GB")
    elif args.fit:
        # Measurements from the conversation, on a ~4 GB sandbox:
        measured_h = [128, 512, 1024, 2048, 3000]
        measured_gb = [0.636, 0.912, 1.512, 2.725, 3.860]
        fit_and_extrapolate(measured_h, measured_gb)
    else:
        parser.print_help()
