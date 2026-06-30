import mlx.nn as nn
import mlx.core as mx


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = self.d_k ** -0.5

        # One projection each for Q, K, V across all heads at once.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)  # on the output projection

    def _split_heads(self, t, B, T):
        # (B, T, d_model) -> (B, num_heads, T, d_k)
        return t.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, x, return_weights=False):
        B, T, C = x.shape

        Q = self._split_heads(self.W_q(x), B, T)
        K = self._split_heads(self.W_k(x), B, T)
        V = self._split_heads(self.W_v(x), B, T)

        if return_weights:
            # Explicit path: materialize the (B, num_heads, T, T) attention
            # weights so they can be inspected/visualized. Costly in memory at
            # long context, so it's opt-in and off during training.
            scores = mx.matmul(Q, mx.swapaxes(K, -2, -1)) * self.scale
            mask = mx.tril(mx.ones((T, T)))
            scores = mx.where(mask == 0, -mx.inf, scores)
            weights = mx.softmax(scores, axis=-1)
            out = mx.matmul(weights, V)
        else:
            # Fused, flash-style attention. Never materializes the T x T tensor,
            # which is the single biggest memory term at GPT-2 context lengths.
            out = mx.fast.scaled_dot_product_attention(
                Q, K, V, scale=self.scale, mask="causal"
            )
            weights = None

        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)  # merge heads
        out = self.dropout(self.W_o(out))
        return (out, weights) if return_weights else out


if __name__ == "__main__":
    # Example usage
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_length = 10

    attention_layer = MultiHeadAttention(d_model, num_heads)
    x = mx.random.normal((batch_size, seq_length, d_model))
    output = attention_layer(x)
    print("Output shape:", output.shape)

    output, attention_weights = attention_layer(x, return_weights=True)
    print("Attention weights shape (B, num_heads, T, T):", attention_weights.shape)
