import mlx.core as mx


def _top_k_filter(logits, k):
    """Keep only the k highest logits, set the rest to -inf before sampling."""
    if k is None or k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    kth = mx.sort(logits, axis=-1)[:, -k][:, None]  # k-th largest per row
    return mx.where(logits < kth, -mx.inf, logits)


def generate(model, tokenizer, prompt="\n", max_new_tokens=500,
             context_length=128, temperature=1.0, top_k=40):
    """Autoregressively sample tokens from the model."""
    model.eval()
    idx = mx.array([tokenizer.encode(prompt)])  # (1, t)

    for _ in range(max_new_tokens):
        # crop to the last context_length tokens (position embedding limit)
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :].astype(mx.float32) / temperature  # next-token logits
        logits = _top_k_filter(logits, top_k)
        next_id = mx.random.categorical(logits)   # (1,) sampled from softmax(logits)
        idx = mx.concatenate([idx, next_id[:, None]], axis=1)
        mx.eval(idx)

    return tokenizer.decode(idx[0].tolist())


def main():
    # Sample from a saved checkpoint, using the active training preset.
    from .data import DataLoader
    from .model import Transformer
    from .quantize import load_quantized
    from .train import CONFIG, PRESETS, DTYPES, CHECKPOINT

    cfg = PRESETS[CONFIG]
    loader = DataLoader(cfg["context_length"], batch_size=1, tokenizer=cfg["tokenizer"])
    model = Transformer(
        cfg["num_layers"], cfg["embed_dim"], cfg["num_heads"],
        cfg["context_length"], loader.tokenizer.vocab_size, tie_weights=True,
    )
    if DTYPES[cfg["dtype"]] != mx.float32:
        model.set_dtype(DTYPES[cfg["dtype"]])
    # load_quantized transparently handles both plain and quantized checkpoints
    # (it inspects the file metadata and rebuilds the quantized sub-modules).
    load_quantized(model, CHECKPOINT)
    print(generate(model, loader.tokenizer, max_new_tokens=1000,
                   context_length=cfg["context_length"]))


if __name__ == "__main__":
    main()
