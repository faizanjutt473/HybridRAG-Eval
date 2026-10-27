"""Vector, BM25 and hybrid (Reciprocal Rank Fusion) retrieval over one index bundle."""
from src.config import RRF_K, TOP_K_CANDIDATES
from src.embeddings import embed_texts
from src.indexing import IndexBundle


def reciprocal_rank_fusion(rankings, k: int = RRF_K):
    """rankings: list of ranked lists of chunk indices -> [(index, fused_score)] best first."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    MODES = ("vector", "bm25", "hybrid")

    def __init__(self, bundle: IndexBundle):
        self.bundle = bundle

    def _vector(self, query: str, k: int):
        return self.bundle.vectors.search(embed_texts([query])[0], k)

    def _bm25(self, query: str, k: int):
        return self.bundle.bm25.search(query, k)

    def search(self, query: str, mode: str, k: int):
        """Return the top-k chunk dicts (each with a 'score') for the given mode."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        pool = max(k, TOP_K_CANDIDATES)

        if mode == "vector":
            hits = self._vector(query, k)
        elif mode == "bm25":
            hits = self._bm25(query, k)
        else:
            vec = [i for i, _ in self._vector(query, pool)]
            kw = [i for i, _ in self._bm25(query, pool)]
            hits = reciprocal_rank_fusion([vec, kw])[:k]

        results = []
        for rank, (idx, score) in enumerate(hits, start=1):
            chunk = dict(self.bundle.chunks[idx])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results
