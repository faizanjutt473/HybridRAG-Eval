"""Cross-encoder reranking of retrieved candidates."""
from functools import lru_cache

import numpy as np

from src.config import RERANK_MODEL


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank(query: str, candidates, top_k: int):
    """Score (query, chunk) pairs with the cross-encoder and return the best top_k."""
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = np.asarray(get_reranker().predict(pairs, batch_size=16, show_progress_bar=False))
    order = np.argsort(scores)[::-1][:top_k]
    results = []
    for new_rank, i in enumerate(order, start=1):
        chunk = dict(candidates[i])
        chunk["retrieval_rank"] = chunk.get("rank")
        chunk["score"] = float(scores[i])
        chunk["rank"] = new_rank
        results.append(chunk)
    return results
