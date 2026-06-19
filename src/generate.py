import mlx.core as mx


def generate(model, tokenizer, prompt="\n", max_new_tokens=500,
             context_length=128, temperature=1.0):
    """Autoregressively sample characters from the model."""
    model.eval()
    idx = mx.array([tokenizer.encode(prompt)])  # (1, t)

    for _ in range(max_new_tokens):
        # crop to the last context_length tokens (position embedding limit)
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # logits for the next token
        next_id = mx.random.categorical(logits)   # (1,) sampled from softmax(logits)
        idx = mx.concatenate([idx, next_id[:, None]], axis=1)
        mx.eval(idx)

    return tokenizer.decode(idx[0].tolist())


def main():
    # Sample from a saved checkpoint.
    from .data import DataLoader
    from .model import Transformer
    from .train import (context_length, embed_dim, num_heads, num_layers,
                        CHECKPOINT)

    loader = DataLoader(context_length, batch_size=1)
    model = Transformer(num_layers, embed_dim, num_heads, context_length,
                        loader.tokenizer.vocab_size)
    model.load_weights(CHECKPOINT)
    print(generate(model, loader.tokenizer, max_new_tokens=1000,
                   context_length=context_length))


if __name__ == "__main__":
    main()
