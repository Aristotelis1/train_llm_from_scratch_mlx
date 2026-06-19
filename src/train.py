import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .data import DataLoader
from .generate import generate
from .model import Transformer

# --- config (tuned for an M3 Air / 16GB; edit freely) ---
context_length = 128
batch_size = 32
embed_dim = 256
num_heads = 8
num_layers = 6
dropout = 0.2

learning_rate = 3e-4
max_iters = 5000
eval_interval = 250
eval_iters = 50
weight_decay = 0.1

CHECKPOINT = "checkpoint.safetensors"


def loss_fn(model, x, y):
    logits, _ = model(x)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


def estimate_loss(model, loader):
    """Average loss over a few batches of train and val, in eval mode."""
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


def main():
    loader = DataLoader(context_length, batch_size)
    vocab_size = loader.tokenizer.vocab_size

    model = Transformer(
        num_layers, embed_dim, num_heads, context_length, vocab_size, dropout=dropout
    )
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"vocab_size={vocab_size}  params={n_params/1e6:.2f}M")

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    model.train()
    t0 = time.time()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model, loader)
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {dt:.1f}s"
            )

        x, y = loader.get_batch("train")
        loss = step(x, y)
        mx.eval(model.parameters(), optimizer.state)

    model.save_weights(CHECKPOINT)
    print(f"saved weights to {CHECKPOINT}")

    print("\n--- sample ---")
    print(generate(model, loader.tokenizer, max_new_tokens=500,
                   context_length=context_length))


if __name__ == "__main__":
    main()
