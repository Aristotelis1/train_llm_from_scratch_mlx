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

- **Token + positional embeddings**
- **Causal multi-head self-attention** (batched across heads in a single matmul)
- **Pre-norm transformer blocks** with residual connections
- **Position-wise MLP** (4× expansion)
- **Dropout** on attention weights, residual paths, and embeddings
- **Final LayerNorm + linear LM head** over the vocabulary
- Character-level tokenizer, cross-entropy training, AdamW, and autoregressive sampling

The default config is ~4.8M parameters and trains comfortably on a 16GB M3 Air.

## Project layout

```
src/
├── model/              # the architecture, from scratch
│   ├── attention.py    # causal multi-head self-attention
│   ├── mlp.py          # feed-forward block
│   └── transformer.py  # blocks + full Transformer (embeddings, LM head)
├── data.py             # TinyShakespeare download, char tokenizer, batching
├── train.py            # training loop, eval, checkpointing
└── generate.py         # autoregressive sampling
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
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

This downloads TinyShakespeare on first run, trains, prints train/val loss periodically, saves `checkpoint.safetensors`, and prints a sample at the end:

```
vocab_size=65  params=4.81M
iter     0 | train 4.2943 | val 4.2975 | 3.4s
iter  1000 | train 1.9939 | val 2.0698 | 164.0s
iter  2500 | train 1.5526 | val 1.7381 | 445.7s
iter  5000 | train 1.3713 | val 1.5870 | 924.1s
saved weights to checkpoint.safetensors
```

A freshly initialized model starts near `ln(vocab_size) ≈ 4.17` — a quick sanity check that the loss is wired up correctly. The full run above takes about **15 minutes** on an M3 Air.

### Generate

Sample from a saved checkpoint:

```bash
python -m src.generate   # or: generate-llm  (after installing)
```

### Configuration

Hyperparameters live as constants at the top of [`src/train.py`](src/train.py) — edit them directly:

```python
context_length = 128
batch_size     = 32
embed_dim      = 256
num_heads      = 8
num_layers     = 6
dropout        = 0.2
learning_rate  = 3e-4
max_iters      = 5000
```

## How it works

The model is trained on next-character prediction. Each batch is a chunk of text `x` and the same chunk shifted one character to the right, `y`; the model learns to predict `y[t]` from `x[:t+1]`. Causal masking in the attention ensures position `t` can only attend to positions `≤ t`, so prediction never peeks at the future.

For the architectural details (pre-norm vs. post-norm, why dropout sits where it does, the batched-attention trick), the model code is written to be read.


## License

MIT
