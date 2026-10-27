"""End-to-end RAG pipeline: retrieve (vector / BM25 / hybrid) -> rerank -> generate with citations.

CLI:
  python -m src.pipeline --question "How many attention heads does the Transformer use?"
  python -m src.pipeline --strategy fixed --mode vector --no-rerank --question "..."
"""
import argparse

from src.config import STRATEGIES, TOP_K_FINAL, TOP_K_RERANK_INPUT
from src.generator import generate_answer
from src.hybrid_retriever import HybridRetriever
from src.indexing import load_bundle
from src.reranker import rerank


class RAGPipeline:
    def __init__(
        self,
        strategy: str = "recursive",
        mode: str = "hybrid",
        use_reranker: bool = True,
        top_k: int = TOP_K_FINAL,
    ):
        self.strategy = strategy
        self.mode = mode
        self.use_reranker = use_reranker
        self.top_k = top_k
        self.retriever = HybridRetriever(load_bundle(strategy))

    def retrieve(self, query: str):
        if self.use_reranker:
            candidates = self.retriever.search(query, self.mode, TOP_K_RERANK_INPUT)
            return rerank(query, candidates, self.top_k)
        return self.retriever.search(query, self.mode, self.top_k)

    def answer(self, query: str, llm=None) -> dict:
        chunks = self.retrieve(query)
        return {
            "question": query,
            "answer": generate_answer(query, chunks, llm),
            "contexts": chunks,
        }


def _main():
    parser = argparse.ArgumentParser(description="Ask the RAG system a question")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--mode", choices=("vector", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    pipe = RAGPipeline(args.strategy, args.mode, not args.no_rerank)
    result = pipe.answer(args.question)

    print("\nANSWER:\n" + result["answer"])
    print("\nRETRIEVED CHUNKS:")
    for c in result["contexts"]:
        preview = c["text"][:110].replace("\n", " ")
        print(f"  #{c['rank']} [{c['source']}, page {c['page']}] score={c['score']:.3f}  {preview}...")


if __name__ == "__main__":
    _main()
