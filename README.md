# train-llm-from-scratch-mlx

A small, readable, **from-scratch GPT-style language model** built with [MLX](https://github.com/ml-explore/mlx) and trained on TinyShakespeare — entirely on an Apple Silicon Mac.

Every component (attention, MLP, transformer block, training loop, sampler) is written by hand with no high-level model libraries, so you can read the whole thing top to bottom and understand how a transformer LM actually works.

```
HENRY BOLINGBROKE:
What our clouds no sights at kind us,
Tell the cause the good fairs more by drum and he
in a were I for a wheat out of him.

ANGELO:
But, keep with Warwick word from and langed.
Come; 'tis he was make himself truth To Bolingbroke.
```

*(real output after 5000 iterations — not perfect English, but it has learned speaker labels, line structure, and mostly-real words from characters alone.)*

## What's inside

A decoder-only transformer, the same family as GPT-2:

- **Token + positional embeddings**, with **weight-tied** LM head (GPT-2 style)
- **Causal multi-head self-attention**, using MLX's fused `scaled_dot_product_attention`
- **Pre-norm transformer blocks** with residual connections
- **Position-wise MLP** (4× expansion, **GELU**)
- **Dropout** on the residual paths and embeddings
- Cross-entropy training with **AdamW**, **cosine LR + warmup**, and gradient clipping
- Choice of tokenizer: **GPT-2 BPE** (50257-vocab, the real thing) or a char-level toy

It ships as a few **presets** in [`src/train.py`](src/train.py): a tiny ~4.8M char model
for instant smoke tests, and the actual **GPT-2-small (124M)** config — which trains
comfortably on a 24GB Apple Silicon Mac.

### Can a MacBook Air really train GPT-2?

Yes. With bf16, weight tying, and fused attention, here's where each size lands on a
**24GB M-series** machine (peak RAM measured / estimated):

| Preset | Params | Peak RAM | Verdict |
|--------|--------|----------|---------|
| `gpt2-small` | 124M | **~11GB** (batch 8) · 14.9GB (batch 12) | Comfortable |
| `gpt2-medium` | 355M | ~12–16GB (grad-accum) | Fits |
| GPT-2 large | 774M | ~16–22GB (batch 1–2 + accum) | Borderline |
| GPT-2 XL | 1.5B | >24GB | Not for training |

Memory is *not* the limit at 124M — wall-clock time is. Training GPT-2 to its original
*quality* needs a large corpus and many GPU-hours (a multi-day run on a single Air, which
is also fanless and will thermal-throttle on long sessions). The realistic proof here is
that the 124M architecture **trains end-to-end on the machine** — overfit TinyShakespeare in
minutes, then point it at a bigger corpus for a longer, genuine run.

## Project layout

```
src/
├── model/              # the architecture, from scratch
│   ├── attention.py    # causal MHA (fused scaled_dot_product_attention)
│   ├── mlp.py          # feed-forward block (GELU)
│   └── transformer.py  # blocks + full Transformer (tied LM head)
├── data.py             # download, BPE/char tokenizers, vectorized batching
├── train.py            # presets, training loop, eval, checkpointing
└── generate.py         # autoregressive sampling (top-k)
```

## Requirements

- Apple Silicon Mac (M1–M5); the GPT-2 presets assume **≥24GB** unified memory
- Python ≥ 3.13
- [MLX](https://github.com/ml-explore/mlx) (installed automatically below)

## Setup

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip (editable install):

```bash
pip install -e .
```

## Usage

### Train

```bash
python -m src.train      # or: train-llm  (after installing)
```

This downloads TinyShakespeare on first run, tokenizes it (cached), trains, prints train/val loss and **peak memory** periodically, saves `checkpoint.safetensors`, and prints a sample at the end. The active preset is `CONFIG` at the top of [`src/train.py`](src/train.py) — the default is `gpt2-small`:

```
config=gpt2-small  dtype=bfloat16  effective_batch=8
vocab_size=50257  params=124.44M
iter     0 | train 11.48 | val 11.47 | 2.6s | peak  2.11GB
iter     6 | train  6.88 | val  7.11 | ...  | peak ~11GB
...
```

A freshly initialized model starts near `ln(vocab_size)` (≈10.8 for BPE, ≈4.17 for char) — a quick sanity check that the loss is wired up correctly. For an instant smoke test, set `CONFIG = "char-demo"` (the original ~4.8M char model, ~15 min for 5000 iters on an Air).

### Generate

Sample from a saved checkpoint:

```bash
python -m src.generate   # or: generate-llm  (after installing)
```

### Configuration

Models are defined as **presets** at the top of [`src/train.py`](src/train.py); pick one with `CONFIG`:

```python
CONFIG = "gpt2-small"   # or "gpt2-medium", "char-demo"
```

Each preset bundles the architecture, tokenizer, precision, batch size, and grad-accumulation:

```python
"gpt2-small": dict(           # 124M params
    tokenizer="bpe", context_length=1024, embed_dim=768, num_heads=12,
    num_layers=12, dropout=0.0, dtype="bfloat16",
    batch=8, grad_accum=1, learning_rate=6e-4, max_iters=5000,
),
```

**Knobs that matter on 24GB:** raise `batch` toward 12–16 if you have free RAM (more
throughput), or raise `grad_accum` for a larger *effective* batch at a smaller memory peak
(uses a memory-lean accumulation path — slightly slower, but how the bigger presets fit).

## How it works

The model is trained on next-token prediction. Each batch is a chunk of tokens `x` and the same chunk shifted one token to the right, `y`; the model learns to predict `y[t]` from `x[:t+1]`. Causal masking in the attention ensures position `t` can only attend to positions `≤ t`, so prediction never peeks at the future.

For the architectural details (pre-norm vs. post-norm, why dropout sits where it does, how the heads are batched into a single fused attention call), the model code is written to be read.


## License

MIT
