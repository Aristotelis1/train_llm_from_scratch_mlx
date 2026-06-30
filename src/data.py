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


class BPETokenizer:
    """GPT-2 byte-pair tokenizer (vocab 50257) via tiktoken.

    Same interface as CharTokenizer so the DataLoader/model don't care which
    one is in use. This is the real GPT-2 vocabulary, so a model trained with
    it is genuinely a GPT-2-family model rather than a char-level toy.
    """

    def __init__(self, _text=None):
        import tiktoken

        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab  # 50257

    def encode(self, s):
        # allow the special <|endoftext|> token through if present
        return self._enc.encode(s, allowed_special="all")

    def decode(self, ids):
        return self._enc.decode([int(i) for i in ids])


TOKENIZERS = {"char": CharTokenizer, "bpe": BPETokenizer}


class DataLoader:
    """Loads the corpus, encodes it, splits train/val, and yields batches.

    Token ids are pre-encoded once (and cached to disk for BPE, which is the
    slow part) so every training step is a cheap on-device gather.
    """

    def __init__(self, context_length, batch_size, tokenizer="bpe", val_frac=0.1):
        text = download()
        tok_name = tokenizer
        self.tokenizer = TOKENIZERS[tok_name](text)
        self.context_length = context_length
        self.batch_size = batch_size

        ids = self._encode_cached(text, tok_name)
        data = mx.array(ids, dtype=mx.int32)
        n = int(len(data) * (1 - val_frac))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def _encode_cached(self, text, tok_name):
        """Encode the corpus once, caching the token ids to a .npy file.

        BPE encoding of ~1M chars takes a moment; caching makes re-runs instant.
        The cache key includes the tokenizer name so char/bpe don't collide.
        """
        cache = os.path.join(DATA_DIR, f"tokens_{tok_name}.npy")
        if os.path.exists(cache):
            return mx.load(cache)
        print(f"Tokenizing corpus with '{tok_name}' tokenizer ...")
        ids = mx.array(self.tokenizer.encode(text), dtype=mx.int32)
        mx.save(cache, ids)
        return ids

    def get_batch(self, split):
        """Return (x, y) where y is x shifted one token to the right.

        Fully vectorized: sample B start offsets, build a (B, T+1) index matrix
        with broadcasting, gather once, then split into inputs/targets. No
        Python loop and no host sync (unlike a list-comprehension + .tolist()).
        """
        data = self.train_data if split == "train" else self.val_data
        T = self.context_length
        max_start = data.size - T - 1
        starts = mx.random.randint(0, max_start, (self.batch_size, 1))
        idx = starts + mx.arange(T + 1)[None, :]  # (B, T+1)
        chunk = data[idx]                          # (B, T+1)
        return chunk[:, :-1], chunk[:, 1:]


if __name__ == "__main__":
    loader = DataLoader(context_length=128, batch_size=4, tokenizer="bpe")
    print("vocab_size:", loader.tokenizer.vocab_size)
    print("train tokens:", loader.train_data.size, "val tokens:", loader.val_data.size)
    x, y = loader.get_batch("train")
    print("x shape:", x.shape, "y shape:", y.shape)
    print("decoded x[0]:", repr(loader.tokenizer.decode(x[0].tolist())[:60]))
    print("decoded y[0]:", repr(loader.tokenizer.decode(y[0].tolist())[:60]))
