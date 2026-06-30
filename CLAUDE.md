# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conventions

- **Always use `uv`** as the package manager (`uv sync`, `uv run ...`, `uv add ...`). Never invoke `pip` directly.
- **Avoid unnecessary comments.** The code should explain itself through clear naming and structure. The comments that do exist explain *why* a non-obvious choice was made (e.g. the fp32 upcast for cross-entropy, re-tying weights after `set_dtype`) — match that bar, don't narrate *what* the code does.

## Commands

```bash
uv sync                       # install deps into .venv
uv run python -m src.train    # train (downloads TinyShakespeare on first run, saves checkpoint.safetensors)
uv run python -m src.generate # sample from checkpoint.safetensors
```

There is no test suite. Each module has an `if __name__ == "__main__"` smoke-test block, runnable directly to sanity-check shapes:

```bash
uv run python -m src.model.attention
uv run python -m src.model.transformer
uv run python -m src.data
```

Requires an Apple Silicon Mac (MLX is Metal-backed) and Python ≥ 3.13.

## Configuration

There is no CLI/argparse. Training is configured by editing two things at the top of [src/train.py](src/train.py):

- `PRESETS` — full model+optimizer hyperparameter dicts (`char-demo` ~4.8M char-level toy, `gpt2-small` 124M, `gpt2-medium` 355M).
- `CONFIG` — a string selecting the active preset.

[src/generate.py](src/generate.py) imports `CONFIG`/`PRESETS`/`DTYPES`/`CHECKPOINT` from `train.py`, so **a checkpoint must be generated with the same `CONFIG` it was trained under** — the model is rebuilt from the preset, then weights are loaded. Changing `CONFIG` after training will mismatch shapes/tokenizer.

## Architecture

A decoder-only (GPT-2 family) transformer written from scratch on MLX, no high-level model libraries. Data flows: `DataLoader` → `Transformer` → cross-entropy (train) or autoregressive `generate` (inference).

- **[src/data.py](src/data.py)** — `DataLoader` downloads TinyShakespeare, encodes it once (BPE token ids are cached to `data/tokens_*.npy`), splits train/val, and yields `(x, y)` batches where `y` is `x` shifted one token right. `get_batch` is fully vectorized — no Python loop, no host sync. Two interchangeable tokenizers behind one interface (`CharTokenizer`, `BPETokenizer` via tiktoken's `gpt2` encoding, vocab 50257); the model/loader are tokenizer-agnostic.
- **[src/model/attention.py](src/model/attention.py)** — `MultiHeadAttention`. **Two code paths**: the default uses fused `mx.fast.scaled_dot_product_attention` with `mask="causal"` (flash-style, never materializes the T×T matrix — critical for memory at GPT-2 context lengths); the `return_weights=True` path explicitly builds the masked softmax so attention weights can be inspected (opt-in, off during training).
- **[src/model/mlp.py](src/model/mlp.py)** — position-wise feed-forward, 4× expansion, GELU.
- **[src/model/transformer.py](src/model/transformer.py)** — `Block` is **pre-norm** (`x + attn(ln1(x))`, `x + mlp(ln2(x))`). `Transformer` adds token + positional embeddings, embedding dropout, the block stack, final LayerNorm, and the LM head. The `return_weights` flag threads through `Transformer → Block → MultiHeadAttention` to collect per-layer attention maps.
- **[src/train.py](src/train.py)** — training loop, LR schedule, checkpointing. **[src/generate.py](src/generate.py)** — `generate()` does autoregressive sampling with temperature + top-k, cropping context to the last `context_length` tokens.

### Non-obvious mechanics (read before touching the training loop)

- **Weight tying:** the LM head shares the token-embedding matrix (`lm_head.weight = input_embedding.weight`). `nn.Module.set_dtype` replaces each leaf with a fresh cast array, which **breaks the tie** — both [train.py](src/train.py) and [generate.py](src/generate.py) must re-assign `model.lm_head.weight = model.input_embedding.weight` after `set_dtype`. Keep this pairing intact if you change dtype handling.
- **Gradient accumulation:** the compiled `step` takes micro-batches of shape `(accum, batch, T)`, accumulates grads over the unrolled loop, averages, clips by global norm, then does one optimizer update. Effective batch = `batch * grad_accum`. This keeps per-step activation/logit memory small while reaching a large batch.
- **`mx.compile`:** `step` is compiled with `inputs=state, outputs=state` where `state = [model.state, optimizer.state]`. The optimizer is `init`-ed before this capture. Anything mutating that state outside `step` must be reflected in `mx.eval(loss, state)`.
- **fp32 cross-entropy:** logits are upcast to fp32 before the 50k-way softmax/log-sum-exp (bf16 loses precision there); the buffer is transient and freed after backward.
- **MLX laziness:** arrays are lazy — `mx.eval(...)` forces computation. Loss/eval code accumulates on-device and syncs to host (`.item()`) as rarely as possible; preserve that pattern to avoid stalls.
- **LR schedule:** linear warmup (`warmup_frac` of `max_iters`) → cosine decay to `min_lr_frac` of peak, the standard GPT-2 schedule.
