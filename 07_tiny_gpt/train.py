#!/usr/bin/env python3
"""Stage 7: train the tiny GPT on a character-level Shakespeare
subset, and generate text from it.
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinyGPT  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "shakespeare_subset.txt"
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
SUBSET_CHARS = 100_000
SEQ_LEN = 48
D_MODEL = 64
D_FF = 256
N_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.3
STEPS = 4000
EVAL_EVERY = 200


def load_data():
    if not DATA_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        full_path = DATA_DIR / "tinyshakespeare_full.txt"
        urllib.request.urlretrieve(SHAKESPEARE_URL, full_path)
        full_text = full_path.read_text()
        DATA_PATH.write_text(full_text[:SUBSET_CHARS])

    text = DATA_PATH.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=int)
    return data, stoi, itos


def get_batch(data, seq_len, batch_size, rng):
    n = len(data)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    token_rows = []
    target_rows = []
    for s in starts:
        e = s + seq_len
        t_start = s + 1
        t_end = e + 1
        token_rows.append(data[s:e])
        target_rows.append(data[t_start:t_end])
    return np.stack(token_rows), np.stack(target_rows)


def plot_loss(history, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["step"] for r in history], [r["loss"] for r in history])
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (nats/char)")
    ax.set_title("TinyGPT training loss (character-level Shakespeare)")
    ax.axhline(
        np.log(61), color="gray", linestyle="--", label="uniform-random"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    data, stoi, itos = load_data()
    vocab_size = len(stoi)
    print(f"data: {len(data):,} chars, vocab_size={vocab_size}")

    net = TinyGPT(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        learning_rate=LEARNING_RATE,
        seed=0,
    )
    print(f"params: {net.n_params():,}")

    rng = np.random.default_rng(0)
    history = []
    t0 = time.time()
    for step in range(STEPS):
        tokens, targets = get_batch(data, SEQ_LEN, BATCH_SIZE, rng)
        loss = net.step(tokens, targets)
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            elapsed = time.time() - t0
            print(f"step {step}: loss={loss:.4f}  ({elapsed:.1f}s)")
            history.append({"step": step, "loss": loss})

    plot_loss(history, "plots/training_loss.png")

    print("\n--- sample generation (temperature=0.8) ---")
    prompt = "ROMEO:"
    prompt_tokens = [stoi[c] for c in prompt if c in stoi]
    gen_rng = np.random.default_rng(42)
    generated = net.generate(
        prompt_tokens, n_new=300, rng=gen_rng, temperature=0.8
    )
    text = "".join(itos[t] for t in generated)
    print(text)

    with open("plots/sample_output.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
