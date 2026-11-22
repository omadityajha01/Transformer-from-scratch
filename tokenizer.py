"""Byte-level Byte Pair Encoding (BPE) tokeniser, written from scratch.

Running this file trains the tokeniser on data/input.txt, saves the merges
to data/tokenizer.json and writes the encoded dataset to data/train.npy and
data/val.npy (90/10 split).
"""

import json

import numpy as np


def get_pair_counts(ids):
    """Count how often each adjacent pair of token ids occurs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (id, id) -> new id, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    def train(self, text, vocab_size=512, verbose=False):
        """Learn `vocab_size - 256` merges from `text`."""
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            counts = get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (new_id - 255) % 32 == 0:
                print(f"merge {new_id - 255}/{vocab_size - 256}: {pair} -> {new_id} {self.vocab[new_id]!r}")

    def encode(self, text):
        """Convert a string to a list of token ids."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # Apply the earliest-learned merge that is present in the sequence.
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """Convert a list of token ids back to a string."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        with open(path, "w") as f:
            json.dump([[a, b, new_id] for (a, b), new_id in self.merges.items()], f)

    def load(self, path):
        with open(path) as f:
            merges = json.load(f)
        self.__init__()
        for a, b, new_id in merges:
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        return self


if __name__ == "__main__":
    with open("data/input.txt", encoding="utf-8") as f:
        text = f.read()

    tok = BPETokenizer()
    tok.train(text, vocab_size=512, verbose=True)
    tok.save("data/tokenizer.json")

    ids = np.array(tok.encode(text), dtype=np.uint16)
    n = int(0.9 * len(ids))
    np.save("data/train.npy", ids[:n])
    np.save("data/val.npy", ids[n:])
    print(f"{len(text):,} characters -> {len(ids):,} tokens "
          f"(compression {len(text) / len(ids):.2f}x); train {n:,}, val {len(ids) - n:,}")
