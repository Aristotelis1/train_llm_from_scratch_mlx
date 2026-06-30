import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def model_nbytes(model):
    """Resident size of a model's parameters, in bytes (forces evaluation)."""
    params = model.parameters()
    mx.eval(params)
    return sum(p.nbytes for _, p in tree_flatten(params))


def quantize_model(model, group_size=64, bits=8):
    """Replace eligible Linear/Embedding layers with quantized versions.

    nn.quantize's default predicate already skips layers whose dimensions are
    not a multiple of group_size, so weight-tied heads (input_embedding doubles
    as the LM head via as_linear) and odd-sized layers are handled correctly:
    QuantizedEmbedding.as_linear keeps the tie intact after quantization.
    """
    nn.quantize(model, group_size=group_size, bits=bits)
    return model


def save_quantized(model, path, group_size=64, bits=8):
    """Save a quantized model with the quantization params in the file metadata.

    The metadata lets load_quantized rebuild the QuantizedLinear/QuantizedEmbedding
    sub-modules with matching shapes before loading the packed weights.
    """
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(
        path, weights, metadata={"group_size": str(group_size), "bits": str(bits)}
    )


def quantization_params(path):
    """Return (group_size, bits) if the checkpoint is quantized, else None."""
    _, meta = mx.load(path, return_metadata=True)
    if meta and "bits" in meta and "group_size" in meta:
        return int(meta["group_size"]), int(meta["bits"])
    return None


def load_quantized(model, path):
    """Quantize `model` to match a saved quantized checkpoint, then load it."""
    params = quantization_params(path)
    if params is None:
        model.load_weights(path)
        return model
    group_size, bits = params
    quantize_model(model, group_size=group_size, bits=bits)
    model.load_weights(path)
    return model


def _smoke_test():
    from .model import Transformer

    mx.random.seed(0)
    model = Transformer(
        num_layers=2, embed_dim=128, num_heads=4,
        context_length=64, vocab_size=256, tie_weights=True,
    )
    model.eval()
    mx.eval(model.parameters())
    x = mx.random.randint(0, 256, (1, 32))
    ref = model(x)
    ref_bytes = model_nbytes(model)

    quantize_model(model, group_size=64, bits=8)
    q = model(x)
    q_bytes = model_nbytes(model)

    # cosine similarity between full-precision and quantized logits
    a, b = ref.reshape(-1).astype(mx.float32), q.reshape(-1).astype(mx.float32)
    cos = (a @ b) / (mx.linalg.norm(a) * mx.linalg.norm(b))

    print(f"params: {ref_bytes/1e6:.2f}MB -> {q_bytes/1e6:.2f}MB "
          f"({100*(1-q_bytes/ref_bytes):.1f}% smaller)")
    print(f"logit cosine similarity: {cos.item():.5f}")
    assert q_bytes < ref_bytes, "quantization did not shrink the model"
    assert cos.item() > 0.99, "8-bit quantization changed the logits too much"

    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "q.safetensors")
    save_quantized(model, path, group_size=64, bits=8)
    assert quantization_params(path) == (64, 8)

    reloaded = Transformer(
        num_layers=2, embed_dim=128, num_heads=4,
        context_length=64, vocab_size=256, tie_weights=True,
    )
    load_quantized(reloaded, path)
    reloaded.eval()
    r = reloaded(x)
    assert mx.allclose(r, q, atol=1e-4), "reloaded quantized model diverged"
    print("round-trip save/load OK")


if __name__ == "__main__":
    _smoke_test()
