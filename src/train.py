import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .data import DataLoader
from .generate import generate
from .model import Transformer

# --- presets -------------------------------------------------------------
# Pick one with CONFIG below. The gpt2-* presets are the real GPT-2 family
# (BPE vocab, GELU, weight tying); "char-demo" is the original tiny model.
# batch/grad_accum are tuned for a 24GB M-series Mac; effective batch =
# batch * grad_accum micro-batches per optimizer step.
PRESETS = {
    "char-demo": dict(  # original ~4.8M char model — fast smoke test
        tokenizer="char", context_length=128, embed_dim=256, num_heads=8,
        num_layers=6, dropout=0.2, dtype="float32",
        batch=32, grad_accum=1, learning_rate=3e-4, max_iters=5000,
    ),
    "gpt2-small": dict(  # 124M — the headline target, comfortable in 24GB
        # batch 8 peaks ~11GB (lots of headroom); raise toward 12-16 if free RAM
        # allows, or set grad_accum>1 for a larger effective batch (lean path).
        tokenizer="bpe", context_length=1024, embed_dim=768, num_heads=12,
        num_layers=12, dropout=0.0, dtype="bfloat16",
        batch=8, grad_accum=1, learning_rate=6e-4, max_iters=5000,
    ),
    "gpt2-medium": dict(  # 355M — fits via memory-lean grad accumulation
        tokenizer="bpe", context_length=1024, embed_dim=1024, num_heads=16,
        num_layers=24, dropout=0.0, dtype="bfloat16",
        batch=2, grad_accum=8, learning_rate=3e-4, max_iters=5000,
    ),
}

CONFIG = "gpt2-small"

# --- training knobs (shared across presets) ---
eval_interval = 250
eval_iters = 20
weight_decay = 0.1
grad_clip = 1.0
warmup_frac = 0.02       # fraction of max_iters spent warming up the LR
min_lr_frac = 0.1        # cosine decays to this fraction of the peak LR

CHECKPOINT = "checkpoint.safetensors"

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


def loss_fn(model, x, y):
    logits = model(x)
    B, T, V = logits.shape
    # Keep logits in the model dtype. At GPT-2 vocab (50257) the logits tensor
    # (B*T*V) is one of the largest in the graph, so upcasting it to fp32 would
    # roughly double peak memory; bf16 cross-entropy with stable log-sum-exp is
    # accurate enough here. (cross_entropy subtracts the max before exp.)
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


def estimate_loss(model, loader):
    """Average loss over a few batches of train and val, in eval mode.

    Each batch is evaluated and synced immediately so its activations (and the
    big logits tensor) are freed before the next one — accumulating them lazily
    would hold eval_iters forward graphs in memory at once.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_iters):
            x, y = loader.get_batch(split)
            total += loss_fn(model, x, y).item()
        out[split] = total / eval_iters
    model.train()
    return out


def build_lr_schedule(peak_lr, max_iters):
    """Linear warmup -> cosine decay, the standard GPT-2 schedule."""
    warmup_steps = max(1, int(max_iters * warmup_frac))
    warmup = optim.linear_schedule(0.0, peak_lr, warmup_steps)
    decay = optim.cosine_decay(peak_lr, max_iters - warmup_steps, peak_lr * min_lr_frac)
    return optim.join_schedules([warmup, decay], [warmup_steps])


def main():
    cfg = PRESETS[CONFIG]
    dtype = DTYPES[cfg["dtype"]]
    accum = cfg["grad_accum"]
    print(f"config={CONFIG}  dtype={cfg['dtype']}  "
          f"effective_batch={cfg['batch'] * accum}")

    loader = DataLoader(cfg["context_length"], cfg["batch"], tokenizer=cfg["tokenizer"])
    vocab_size = loader.tokenizer.vocab_size

    model = Transformer(
        cfg["num_layers"], cfg["embed_dim"], cfg["num_heads"],
        cfg["context_length"], vocab_size, dropout=cfg["dropout"], tie_weights=True,
    )
    if dtype != mx.float32:
        model.set_dtype(dtype)
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"vocab_size={vocab_size}  params={n_params/1e6:.2f}M")

    schedule = build_lr_schedule(cfg["learning_rate"], cfg["max_iters"])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=weight_decay)
    optimizer.init(model.trainable_parameters())  # populate state before capture

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    # Fast path (accum == 1): one fused, compiled forward+backward+update. The
    # whole step lives in a single compiled function with the model/optimizer
    # state captured as inputs/outputs -- the only compile-safe shape, since
    # optimizer.update *replaces* the parameter arrays (a separately-compiled
    # update would break the captured references).
    @partial(mx.compile, inputs=state, outputs=state)
    def fused_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, grad_clip)
        optimizer.update(model, grads)
        return loss

    def fast_step():
        x, y = loader.get_batch("train")
        loss = fused_step(x, y)
        mx.eval(state, loss)
        return loss.item()

    # Memory-lean path (accum > 1): accumulate grads over micro-batches,
    # evaluating after each so only one micro-batch's activations are ever
    # resident. Uncompiled (compiling across the optimizer update isn't safe),
    # which is the right trade for the big presets -- there you're memory-bound,
    # not launch-bound. Gives a large effective batch at a small memory peak.
    def accumulate_step():
        grads = None
        running = 0.0
        for _ in range(accum):
            x, y = loader.get_batch("train")
            loss, g = loss_and_grad(model, x, y)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            mx.eval(grads)               # free this micro-batch's activations
            running += loss.item()
        grads = tree_map(lambda g: g * (1.0 / accum), grads)
        grads, _ = optim.clip_grad_norm(grads, grad_clip)
        optimizer.update(model, grads)
        mx.eval(state)
        return running / accum

    train_step = fast_step if accum == 1 else accumulate_step

    model.train()
    t0 = time.time()
    for it in range(cfg["max_iters"] + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model, loader)
            dt = time.time() - t0
            peak_gb = mx.get_peak_memory() / 1e9
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {dt:.1f}s | peak {peak_gb:.2f}GB"
            )

        train_step()

    model.save_weights(CHECKPOINT)
    print(f"saved weights to {CHECKPOINT}")

    print("\n--- sample ---")
    print(generate(model, loader.tokenizer, max_new_tokens=500,
                   context_length=cfg["context_length"]))


if __name__ == "__main__":
    main()
