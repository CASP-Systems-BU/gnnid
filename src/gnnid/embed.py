"""Word2Vec sentence encoding with sinusoidal positional encoding (FLASH §4.2).

Train once on all TRAIN-split sentences. Sub-min_count tokens are mapped to a
per-namespace `unk:<ns>` before training so each namespace's unk gets a real
vector; unseen tokens at inference map there too — a never-before-seen exfil
domain becomes the benign-rare `unk:dst-ext` and shifts the embedding rather
than vanishing.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def token_namespace(tok: str) -> str:
    """`p:/cart/*` -> 'p'; `dst:ext:foo.com` -> 'dst-ext'. The unk bucket key."""
    head, _, rest = tok.partition(":")
    if head == "dst" and rest.startswith("ext:"):
        return "dst-ext"
    return head or "misc"


def unk_token(tok: str) -> str:
    return f"unk:{token_namespace(tok)}"


def _positional_encoding(n: int, dim: int) -> np.ndarray:
    """Standard sinusoidal PE, shape [n, dim]."""
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)
    pos = np.arange(n)[:, None]
    i = np.arange(dim)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / dim)
    pe = np.zeros((n, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


class SentenceEmbedder:
    def __init__(self, kv: KeyedVectors, dim: int):
        self.kv = kv
        self.dim = dim
        self._pe_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------- train
    @classmethod
    def train(cls, sentences: list[list[str]], cfg) -> "SentenceEmbedder":
        dim = int(cfg.dotted_get("w2v.dim", 64))
        min_count = int(cfg.dotted_get("w2v.min_count", 2))
        counts = Counter(t for s in sentences for t in s)
        mapped = [[t if counts[t] >= min_count else unk_token(t) for t in s]
                  for s in sentences]
        model = Word2Vec(
            sentences=mapped, vector_size=dim,
            sg=int(cfg.dotted_get("w2v.sg", 1)),
            window=int(cfg.dotted_get("w2v.window", 5)),
            negative=int(cfg.dotted_get("w2v.negative", 10)),
            min_count=1,  # already thresholded via unk mapping
            epochs=int(cfg.dotted_get("w2v.epochs", 15)),
            workers=int(cfg.dotted_get("w2v.workers", 4)),
            seed=int(cfg.dotted_get("seed", 17)))
        return cls(model.wv, dim)

    # ------------------------------------------------------------------ encode
    def _pe(self, n: int) -> np.ndarray:
        if n not in self._pe_cache:
            self._pe_cache[n] = _positional_encoding(n, self.dim)
        return self._pe_cache[n]

    def _vec(self, tok: str) -> np.ndarray:
        if tok in self.kv:
            return self.kv[tok]
        u = unk_token(tok)
        if u in self.kv:
            return self.kv[u]
        return np.zeros(self.dim, dtype=np.float32)

    def encode(self, tokens: list[str]) -> np.ndarray:
        """Token list -> one dim-vector: (token vectors + PE), mean-pooled."""
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        mat = np.stack([self._vec(t) for t in tokens]).astype(np.float32)
        mat = mat + self._pe(len(tokens))
        return mat.mean(axis=0)

    def encode_many(self, sentences: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return {eid: self.encode(toks) for eid, toks in sentences.items()}

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.kv.save(str(path))

    @classmethod
    def load(cls, path: str | Path, dim: int) -> "SentenceEmbedder":
        return cls(KeyedVectors.load(str(path)), dim)
