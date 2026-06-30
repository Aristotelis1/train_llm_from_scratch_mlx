import mlx.core as mx
import mlx.nn as nn

from .attention import MultiHeadAttention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, multiply_factor=4, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, multiply_factor, dropout)

    def __call__(self, x, return_weights=False):
        if return_weights:
            attn_out, weights = self.attention(self.ln1(x), return_weights=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, weights
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, context_length, vocab_size,
                 multiply_factor=4, dropout=0.0, tie_weights=True):
        super().__init__()
        self.tie_weights = tie_weights
        self.layers = [Block(embed_dim, num_heads, multiply_factor, dropout) for _ in range(num_layers)]
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.ln_final = nn.LayerNorm(embed_dim)
        # With weight tying the token-embedding matrix doubles as the LM head, so
        # there is no separate head weight at all. This keeps a single weight in
        # the parameter tree (the optimizer can't split a shared reference into
        # two), and reclaims ~38.6M params at the GPT-2 vocab. Otherwise use a
        # dedicated linear head.
        if not tie_weights:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def _head(self, x):
        if self.tie_weights:
            return self.input_embedding.as_linear(x)  # x @ embedding.weight.T
        return self.lm_head(x)

    def __call__(self, x, return_weights=False):
        x = self.input_embedding(x) + self.position_embedding(mx.arange(x.shape[1]))
        x = self.embedding_dropout(x)

        if return_weights:
            attention_weights_all_layers = []
            for layer in self.layers:
                x, weights = layer(x, return_weights=True)
                attention_weights_all_layers.append(weights)
            return self._head(self.ln_final(x)), attention_weights_all_layers

        for layer in self.layers:
            x = layer(x)
        return self._head(self.ln_final(x))


if __name__ == "__main__":
    # Example usage
    num_layers = 2
    embed_dim = 512
    num_heads = 8
    context_length = 64
    vocab_size = 1000
    batch_size = 2
    seq_length = 10

    transformer = Transformer(num_layers, embed_dim, num_heads, context_length, vocab_size, dropout=0.1)
    transformer.train()
    x = mx.random.randint(0, vocab_size, (batch_size, seq_length))
    logits = transformer(x)
    print("Output shape:", logits.shape)

    logits, attention_weights_all_layers = transformer(x, return_weights=True)
    for layer_idx, attention_weights in enumerate(attention_weights_all_layers):
        print(f"Attention weights shape for layer {layer_idx} (B, num_heads, T, T):", attention_weights.shape)
