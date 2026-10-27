"""Run the full evaluation: retrieval metrics for every config + answer quality for the best config.

Usage (from the project root):
  python -m evaluation.run_evaluation --skip-answers      # retrieval only, no API calls
  python -m evaluation.run_evaluation                     # + answers, judge, citations (Gemini)
  python -m evaluation.run_evaluation --max-questions 10  # quick trial run
  python -m evaluation.run_evaluation --fresh             # ignore cached answers
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    answer_similarity,
    citation_scores,
    judge_answer,
    retrieval_scores,
)
from src.config import EVAL_SET_PATH, RESULTS_DIR, STRATEGIES  # noqa: E402
from src.generator import NOT_FOUND, generate_answer  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

# (label, retrieval mode, use reranker)
CONFIGS = [
    ("Vector only", "vector", False),
    ("BM25 only", "bm25", False),
    ("Hybrid (RRF)", "hybrid", False),
    ("Hybrid + Rerank", "hybrid", True),
]
BEST_CONFIG = "Hybrid + Rerank"
CACHE_PATH = RESULTS_DIR / "answers_cache.json"


# ------------------------------------------------------------------ helpers
def fmt(v):
    if isinstance(v, float):
        return "-" if pd.isna(v) else f"{v:.3f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ retrieval stage
def run_retrieval(strategy, items):
    """Returns (per-question rows, {question_id: top-k chunks of the best config})."""
    rows, best_contexts = [], {}
    for label, mode, use_rerank in CONFIGS:
        pipe = RAGPipeline(strategy, mode, use_rerank)
        for item in tqdm(items, desc=f"{strategy:9s} | {label:16s}", leave=False):
            ranked = pipe.retrieve(item["question"])
            rows.append(
                {"strategy": strategy, "config": label, "id": item["id"], "type": item["type"],
                 **retrieval_scores(ranked, item)}
            )
            if label == BEST_CONFIG:
                best_contexts[item["id"]] = ranked
    return rows, best_contexts


# ------------------------------------------------------------------ answer stage
def run_answers(strategy, items, contexts, cache, sleep, llm):
    for item in tqdm(items, desc=f"{strategy:9s} | answers", leave=False):
        key = f"{strategy}::{item['id']}"
        if key in cache:
            continue
        chunks = contexts[item["id"]]
        try:
            answer = generate_answer(item["question"], chunks, llm)
            time.sleep(sleep)
            judge = judge_answer(item["question"], item["ground_truth_answer"], answer, llm)
            time.sleep(sleep)
        except Exception as exc:  # stop cleanly on quota errors; results so far stay cached
            print(f"\n  LLM error on Q{item['id']}: {str(exc)[:200]}")
            print("  Partial results are cached - re-run the same command later to continue.")
            return False
        cache[key] = {
            "strategy": strategy,
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth_answer"],
            "answer": answer,
            "judge_score": judge["score"],
            "judge_reason": judge["reason"],
            "similarity": answer_similarity(answer, item["ground_truth_answer"]),
            "abstained": float(NOT_FOUND.lower() in answer.lower()),
            **citation_scores(answer, item, chunks),
            "contexts": [{"source": c["source"], "page": c["page"], "rank": c["rank"]} for c in chunks],
        }
        save_json(CACHE_PATH, cache)
    return True


def summarise_answers(cache, items):
    ids = {i["id"] for i in items}
    rows = []
    for strategy in STRATEGIES:
        entries = [v for v in cache.values() if v["strategy"] == strategy and v["id"] in ids]
        if not entries:
            continue
        n = len(entries)
        rows.append(
            {
                "strategy": strategy,
                "config": BEST_CONFIG,
                "answers_graded": n,
                "correctness": sum(e["judge_score"] for e in entries) / n,
                "fully_correct": sum(e["judge_score"] == 1.0 for e in entries) / n,
                "answer_similarity": sum(e["similarity"] for e in entries) / n,
                "citation_correct": sum(e["citation_correct"] for e in entries) / n,
                "citation_grounded": sum(e["citation_grounded"] for e in entries) / n,
                "abstain_rate": sum(e["abstained"] for e in entries) / n,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ reporting
def make_chart(summary: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in zip(axes, ("hit@5", "mrr"), ("Hit rate @5", "MRR @5")):
        pivot = summary.pivot(index="config", columns="strategy", values=metric)
        pivot = pivot.reindex([c[0] for c in CONFIGS])
        pivot.plot.bar(ax=ax, rot=20, width=0.75)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="chunking")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-answers", action="store_true", help="retrieval metrics only (no Gemini calls)")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between LLM calls")
    parser.add_argument("--fresh", action="store_true", help="ignore cached answers")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    args = parser.parse_args()

    items = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    if args.max_questions:
        items = items[: args.max_questions]
    print(f"Evaluating {len(items)} questions | strategies: {args.strategies}\n")

    # 1) retrieval metrics for every strategy x config
    per_q, contexts = [], {}
    for strategy in args.strategies:
        rows, best = run_retrieval(strategy, items)
        per_q += rows
        contexts[strategy] = best
    per_q_df = pd.DataFrame(per_q)
    per_q_df.to_csv(RESULTS_DIR / "per_question_retrieval.csv", index=False)

    metric_cols = ["hit@1", "hit@3", "hit@5", "doc_hit@5", "mrr"]
    summary = per_q_df.groupby(["strategy", "config"], sort=False)[metric_cols].mean().reset_index()

    by_type = per_q_df.groupby(["strategy", "config", "type"], sort=False)[["hit@5", "mrr"]].mean().reset_index()
    by_type.to_csv(RESULTS_DIR / "retrieval_by_question_type.csv", index=False)

    # 2) answer quality for the Hybrid + Rerank configuration
    answers_df = pd.DataFrame()
    if not args.skip_answers:
        if args.fresh and CACHE_PATH.exists():
            CACHE_PATH.unlink()
        cache = load_json(CACHE_PATH, {})
        llm = get_llm()
        for strategy in args.strategies:
            if not run_answers(strategy, items, contexts[strategy], cache, args.sleep, llm):
                break
        answers_df = summarise_answers(cache, items)
        for strategy in args.strategies:
            entries = [v for v in cache.values() if v["strategy"] == strategy]
            save_json(RESULTS_DIR / f"answers_{strategy}.json", entries)

    # 3) final results table
    final = summary.copy()
    if not answers_df.empty:
        final = final.merge(answers_df, on=["strategy", "config"], how="left")
    final.insert(0, "chunking", final.pop("strategy"))
    final = final.rename(columns={"config": "retrieval", "hit@1": "Hit@1", "hit@3": "Hit@3",
                                  "hit@5": "Hit@5", "doc_hit@5": "DocHit@5", "mrr": "MRR@5"})
    final.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    (RESULTS_DIR / "results_table.md").write_text(df_to_markdown(final) + "\n", encoding="utf-8")
    make_chart(summary, RESULTS_DIR / "comparison_chart.png")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== RESULTS ===")
    print(final.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== Hit@5 by question type ===")
    print(by_type.pivot_table(index=["strategy", "config"], columns="type", values="hit@5", sort=False)
          .to_string(float_format=lambda v: f"{v:.3f}"))

    best_rows = summary[summary["config"] == BEST_CONFIG].set_index("strategy")
    if len(best_rows) > 1:
        winner = best_rows["mrr"].idxmax()
        print(f"\nBetter chunking on {BEST_CONFIG} (by MRR@5): {winner}")
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
