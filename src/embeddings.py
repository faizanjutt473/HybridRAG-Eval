"""Sentence-transformer embeddings (loaded lazily, cached)."""
from functools import lru_cache

import numpy as np

from src.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Return L2-normalised float32 embeddings (cosine similarity == inner product)."""
    emb = get_embedder().encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype="float32")
