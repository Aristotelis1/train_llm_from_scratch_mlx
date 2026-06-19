import os
import urllib.request

import mlx.core as mx

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "input.txt")


def download():
    """Download TinyShakespeare once and return its full text."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(INPUT_PATH):
        print(f"Downloading TinyShakespeare to {INPUT_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, INPUT_PATH)
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Character-level tokenizer: builds int<->char maps from the corpus."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class DataLoader:
    """Loads the corpus, encodes it, splits train/val, and yields batches."""

    def __init__(self, context_length, batch_size, val_frac=0.1):
        text = download()
        self.tokenizer = CharTokenizer(text)
        self.context_length = context_length
        self.batch_size = batch_size

        data = mx.array(self.tokenizer.encode(text), dtype=mx.int32)
        n = int(len(data) * (1 - val_frac))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        """Return (x, y) where y is x shifted one token to the right."""
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.context_length - 1
        ix = mx.random.randint(0, max_start, (self.batch_size,)).tolist()
        x = mx.stack([data[i : i + self.context_length] for i in ix])
        y = mx.stack([data[i + 1 : i + 1 + self.context_length] for i in ix])
        return x, y


if __name__ == "__main__":
    loader = DataLoader(context_length=128, batch_size=4)
    print("vocab_size:", loader.tokenizer.vocab_size)
    print("train tokens:", loader.train_data.size, "val tokens:", loader.val_data.size)
    x, y = loader.get_batch("train")
    print("x shape:", x.shape, "y shape:", y.shape)
    print("decoded x[0]:", repr(loader.tokenizer.decode(x[0].tolist())[:60]))
    print("decoded y[0]:", repr(loader.tokenizer.decode(y[0].tolist())[:60]))
