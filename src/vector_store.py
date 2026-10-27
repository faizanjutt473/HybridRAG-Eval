"""FAISS inner-product index over normalised embeddings."""
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, index: "faiss.Index"):
        self.index = index

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "VectorStore":
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index)

    def search(self, query_emb: np.ndarray, k: int):
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_emb.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        return cls(faiss.read_index(str(path)))
