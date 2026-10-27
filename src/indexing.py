"""Build and load the per-strategy indexes (chunks + FAISS + BM25).

Run:  python -m src.indexing            (build both strategies)
      python -m src.indexing --reingest (re-read raw documents first)
"""
import argparse
import json
from dataclasses import dataclass
from functools import lru_cache

from src.chunking import chunk_documents
from src.config import DOCUMENTS_JSON, INDEX_DIR, STRATEGIES
from src.embeddings import embed_texts
from src.ingestion import ingest_all, load_documents, save_documents
from src.keyword_search import BM25Index
from src.vector_store import VectorStore


@dataclass
class IndexBundle:
    strategy: str
    chunks: list
    vectors: VectorStore
    bm25: BM25Index


def _dir(strategy: str):
    return INDEX_DIR / strategy


def build_index(strategy: str, records) -> IndexBundle:
    chunks = chunk_documents(records, strategy)
    texts = [c["text"] for c in chunks]
    print(f"[{strategy}] {len(chunks)} chunks | avg {sum(c['n_words'] for c in chunks) / len(chunks):.0f} words")

    vectors = VectorStore.build(embed_texts(texts, show_progress=True))
    bm25 = BM25Index.build(texts)

    out = _dir(strategy)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    vectors.save(out / "faiss.index")
    bm25.save(out / "bm25.pkl")
    return IndexBundle(strategy, chunks, vectors, bm25)


@lru_cache(maxsize=None)
def load_bundle(strategy: str) -> IndexBundle:
    out = _dir(strategy)
    if not (out / "faiss.index").exists():
        raise FileNotFoundError(f"No index for '{strategy}'. Run:  python -m src.indexing")
    chunks = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
    return IndexBundle(
        strategy,
        chunks,
        VectorStore.load(out / "faiss.index"),
        BM25Index.load(out / "bm25.pkl"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reingest", action="store_true", help="re-read raw documents")
    args = parser.parse_args()

    if args.reingest or not DOCUMENTS_JSON.exists():
        print("Ingesting raw documents...")
        save_documents(ingest_all())
    records = load_documents()
    print(f"{len({r['source'] for r in records})} documents, {len(records)} pages\n")

    for strategy in STRATEGIES:
        build_index(strategy, records)
    print("\nIndexes saved in", INDEX_DIR)


if __name__ == "__main__":
    main()
