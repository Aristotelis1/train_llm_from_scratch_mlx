import mlx.nn as nn
import mlx.core as mx

class MLP(nn.Module):
    def __init__(self, embed_dim, multiply_factor=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * multiply_factor)
        self.fc2 = nn.Linear(embed_dim * multiply_factor, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Example usage
    embed_dim = 512
    batch_size = 2
    seq_length = 10

    mlp_layer = MLP(embed_dim)
    x = mx.random.normal((batch_size, seq_length, embed_dim))
    output = mlp_layer(x)

    print("Output shape:", output.shape)