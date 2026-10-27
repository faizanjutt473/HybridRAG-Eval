"""BM25 keyword search with light normalisation (lowercase, stop-words, Porter stemming)."""
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")  # keeps decimals such as 28.4
_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    return _stemmer.stem(token)


def tokenize(text: str):
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25Index:
    def __init__(self, bm25: BM25Okapi):
        self.bm25 = bm25

    @classmethod
    def build(cls, texts) -> "BM25Index":
        return cls(BM25Okapi([tokenize(t) for t in texts]))

    def search(self, query: str, k: int):
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return cls(pickle.load(f))
