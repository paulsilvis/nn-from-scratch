"""Stage 7: the complete tiny GPT, assembled from every prior stage.

    token embedding + positional embedding
    -> [transformer block] x n_layers
    -> final layer norm
    -> linear + softmax over the vocabulary

One transformer block:

    x  = x + causal_self_attention(layernorm(x))
    x  = x + feedforward(layernorm(x))

(pre-norm, residual around each sublayer - see notes.md for why the
residual connections matter for trainability at any real depth.)

Single-head attention throughout (stage 6's mechanism, unchanged,
with a causal mask added). Multi-head attention - running several
of these in parallel on smaller slices of d_model and concatenating
- is a real, mechanical extension not implemented here, flagged
explicitly rather than silently assumed.

Output head reuses stage 4's softmax + cross-entropy delta
(p - t) unchanged - trained here to predict, at every position
simultaneously, the character that comes next.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_attention"))
from attention import attention_backward, attention_forward  # noqa: E402
from layers import (  # noqa: E402
    causal_mask,
    feedforward_backward,
    feedforward_forward,
    layernorm_backward,
    layernorm_forward,
)


def softmax(z):
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


class TinyGPT:
    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=64,
        d_k=64,
        d_ff=256,
        n_layers=2,
        learning_rate=0.3,
        seed=None,
    ):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.mask = causal_mask(seq_len)

        emb_scale = np.sqrt(1.0 / vocab_size)
        self.token_emb = rng.normal(0, emb_scale, size=(vocab_size, d_model))
        self.pos_emb = rng.normal(0, emb_scale, size=(seq_len, d_model))

        self.layers = []
        for _ in range(n_layers):
            layer = {
                "ln1_gamma": np.ones(d_model),
                "ln1_beta": np.zeros(d_model),
                "W_Q": rng.normal(
                    0, np.sqrt(1.0 / d_model), size=(d_model, d_k)
                ),
                "W_K": rng.normal(
                    0, np.sqrt(1.0 / d_model), size=(d_model, d_k)
                ),
                "W_V": rng.normal(
                    0, np.sqrt(1.0 / d_model), size=(d_model, d_k)
                ),
                "W_O": rng.normal(0, np.sqrt(1.0 / d_k), size=(d_k, d_model)),
                "ln2_gamma": np.ones(d_model),
                "ln2_beta": np.zeros(d_model),
                "ff_W1": rng.normal(
                    0, np.sqrt(1.0 / d_model), size=(d_ff, d_model)
                ),
                "ff_b1": np.zeros(d_ff),
                "ff_W2": rng.normal(
                    0, np.sqrt(1.0 / d_ff), size=(d_model, d_ff)
                ),
                "ff_b2": np.zeros(d_model),
            }
            self.layers.append(layer)

        self.ln_f_gamma = np.ones(d_model)
        self.ln_f_beta = np.zeros(d_model)
        self.W_head = rng.normal(
            0, np.sqrt(1.0 / d_model), size=(vocab_size, d_model)
        )
        self.b_head = np.zeros(vocab_size)

    def forward(self, tokens):
        """tokens: (batch, seq_len) int array. Returns (logits,
        cache) where logits is (batch, seq_len, vocab_size).
        """
        batch, seq_len = tokens.shape
        x = self.token_emb[tokens] + self.pos_emb[np.newaxis, :seq_len]
        caches = {"x0": x, "layers": []}

        for layer in self.layers:
            ln1_out, ln1_cache = layernorm_forward(
                x, layer["ln1_gamma"], layer["ln1_beta"]
            )
            attn_out, attn_cache = attention_forward(
                ln1_out,
                layer["W_Q"],
                layer["W_K"],
                layer["W_V"],
                mask=self.mask[:seq_len, :seq_len],
            )
            proj_out = attn_out @ layer["W_O"]
            x_after_attn = x + proj_out

            ln2_out, ln2_cache = layernorm_forward(
                x_after_attn, layer["ln2_gamma"], layer["ln2_beta"]
            )
            ff_out, ff_cache = feedforward_forward(
                ln2_out,
                layer["ff_W1"],
                layer["ff_b1"],
                layer["ff_W2"],
                layer["ff_b2"],
            )
            x = x_after_attn + ff_out

            caches["layers"].append(
                {
                    "ln1_cache": ln1_cache,
                    "attn_cache": attn_cache,
                    "attn_out": attn_out,
                    "x_after_attn": x_after_attn,
                    "ln2_cache": ln2_cache,
                    "ff_cache": ff_cache,
                }
            )

        ln_f_out, ln_f_cache = layernorm_forward(
            x, self.ln_f_gamma, self.ln_f_beta
        )
        caches["ln_f_cache"] = ln_f_cache
        caches["ln_f_out"] = ln_f_out

        logits = ln_f_out @ self.W_head.T + self.b_head
        return logits, caches

    def backward(self, tokens, targets, logits, caches):
        batch, seq_len = tokens.shape
        p = softmax(logits)
        t_onehot = np.eye(self.vocab_size)[targets]
        delta = (p - t_onehot) / (batch * seq_len)

        grad_w_head = np.tensordot(
            delta, caches["ln_f_out"], axes=([0, 1], [0, 1])
        )
        grad_b_head = delta.sum(axis=(0, 1))
        grad_ln_f_out = delta @ self.W_head

        grad_x, grad_ln_f_gamma, grad_ln_f_beta = layernorm_backward(
            grad_ln_f_out, caches["ln_f_cache"]
        )

        layer_grads = []
        for layer, layer_cache in zip(
            reversed(self.layers), reversed(caches["layers"])
        ):
            grad_ff_out = grad_x
            (
                grad_ln2_out,
                grad_ff_w1,
                grad_ff_b1,
                grad_ff_w2,
                grad_ff_b2,
            ) = feedforward_backward(
                grad_ff_out,
                layer_cache["ff_cache"],
                layer["ff_W1"],
                layer["ff_W2"],
            )
            grad_x_after_attn_2, grad_ln2_gamma, grad_ln2_beta = (
                layernorm_backward(grad_ln2_out, layer_cache["ln2_cache"])
            )
            grad_x_after_attn = grad_x + grad_x_after_attn_2

            grad_proj_out = grad_x_after_attn
            grad_attn_out = grad_proj_out @ layer["W_O"].T
            grad_w_o = np.tensordot(
                layer_cache["attn_out"],
                grad_proj_out,
                axes=([0, 1], [0, 1]),
            )

            grad_ln1_out, grad_w_q, grad_w_k, grad_w_v = attention_backward(
                grad_attn_out,
                layer_cache["attn_cache"],
                layer["W_Q"],
                layer["W_K"],
                layer["W_V"],
            )
            # attention_backward divides dW_Q/dW_K/dW_V by batch
            # internally (correct when its upstream gradient isn't
            # already batch-averaged, as in stage 6's standalone
            # use). Here `delta` above was already divided by
            # (batch * seq_len), so that internal /batch
            # double-counts the averaging - undo it.
            batch = tokens.shape[0]
            grad_w_q = grad_w_q * batch
            grad_w_k = grad_w_k * batch
            grad_w_v = grad_w_v * batch
            grad_x_from_ln1, grad_ln1_gamma, grad_ln1_beta = (
                layernorm_backward(grad_ln1_out, layer_cache["ln1_cache"])
            )
            grad_x = grad_x_after_attn + grad_x_from_ln1

            layer_grads.append(
                {
                    "ln1_gamma": grad_ln1_gamma,
                    "ln1_beta": grad_ln1_beta,
                    "W_Q": grad_w_q,
                    "W_K": grad_w_k,
                    "W_V": grad_w_v,
                    "W_O": grad_w_o,
                    "ln2_gamma": grad_ln2_gamma,
                    "ln2_beta": grad_ln2_beta,
                    "ff_W1": grad_ff_w1,
                    "ff_b1": grad_ff_b1,
                    "ff_W2": grad_ff_w2,
                    "ff_b2": grad_ff_b2,
                }
            )
        layer_grads.reverse()

        grad_token_emb = np.zeros_like(self.token_emb)
        np.add.at(grad_token_emb, tokens, grad_x)
        grad_pos_emb = grad_x.sum(axis=0)

        return {
            "token_emb": grad_token_emb,
            "pos_emb": grad_pos_emb,
            "layers": layer_grads,
            "ln_f_gamma": grad_ln_f_gamma,
            "ln_f_beta": grad_ln_f_beta,
            "W_head": grad_w_head,
            "b_head": grad_b_head,
        }

    def apply_grads(self, grads):
        lr = self.learning_rate
        self.token_emb -= lr * grads["token_emb"]
        self.pos_emb -= lr * grads["pos_emb"]
        self.ln_f_gamma -= lr * grads["ln_f_gamma"]
        self.ln_f_beta -= lr * grads["ln_f_beta"]
        self.W_head -= lr * grads["W_head"]
        self.b_head -= lr * grads["b_head"]
        for layer, grad in zip(self.layers, grads["layers"]):
            for key in grad:
                layer[key] -= lr * grad[key]

    def step(self, tokens, targets):
        logits, caches = self.forward(tokens)
        grads = self.backward(tokens, targets, logits, caches)
        self.apply_grads(grads)

        p = softmax(logits)
        eps = 1e-12
        t_onehot = np.eye(self.vocab_size)[targets]
        loss = -np.mean(np.sum(t_onehot * np.log(p + eps), axis=-1))
        return loss

    def n_params(self):
        total = self.token_emb.size + self.pos_emb.size
        total += self.W_head.size + self.b_head.size
        total += self.ln_f_gamma.size + self.ln_f_beta.size
        for layer in self.layers:
            for v in layer.values():
                total += v.size
        return total

    def generate(self, prompt_tokens, n_new, rng, temperature=1.0):
        tokens = list(prompt_tokens)
        for _ in range(n_new):
            start = max(0, len(tokens) - self.seq_len)
            context = np.array([tokens[start:]], dtype=int)
            logits, _ = self.forward(context)
            last_logits = logits[0, -1, :] / temperature
            p = softmax(last_logits[np.newaxis, :])[0]
            next_token = rng.choice(self.vocab_size, p=p)
            tokens.append(int(next_token))
        return tokens
