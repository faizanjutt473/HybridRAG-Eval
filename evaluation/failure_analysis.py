"""Explain the retrieval misses and show where the two chunking strategies disagree.

Run after `python -m evaluation.run_evaluation`:
    python -m evaluation.failure_analysis
Output is printed and saved to evaluation/results/failure_analysis.md
"""
import json

import pandas as pd

from evaluation.metrics import is_relevant
from src.config import EVAL_SET_PATH, RESULTS_DIR, STRATEGIES
from src.indexing import load_bundle
from src.pipeline import RAGPipeline

CONFIG = "Hybrid + Rerank"


def evidence_locations(strategy, item, limit=8):
    """(source, page) pairs where some chunk really contains the evidence."""
    found = {(c["source"], c["page"]) for c in load_bundle(strategy).chunks if is_relevant(c, item)}
    found = sorted(found)
    more = f" (+{len(found) - limit} more)" if len(found) > limit else ""
    return ", ".join(f"{s} p{p}" for s, p in found[:limit]) + more if found else "NOT FOUND IN ANY CHUNK"


def main():
    items = {i["id"]: i for i in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))}
    df = pd.read_csv(RESULTS_DIR / "per_question_retrieval.csv")
    df = df[df["config"] == CONFIG]
    out = [f"# Failure analysis ({CONFIG})", ""]

    # 1) questions missed at Hit@5
    for strategy in STRATEGIES:
        pipe = RAGPipeline(strategy, "hybrid", True)
        missed = df[(df["strategy"] == strategy) & (df["hit@5"] == 0)]
        out.append(f"## {strategy}: {len(missed)} question(s) missed at Hit@5")
        for qid in missed["id"]:
            item = items[qid]
            out += [
                "",
                f"### Q{qid} [{item['type']}] {item['question']}",
                f"- expected: {item['source_doc']} | evidence: {item['evidence']}",
                f"- evidence is actually in: {evidence_locations(strategy, item)}",
                "- retrieved top-5:",
            ]
            for c in pipe.retrieve(item["question"]):
                out.append(f"  - #{c['rank']} {c['source']} p{c['page']}: {c['text'][:90]!r}")
        out.append("")

    # 2) where the strategies disagree
    wide = df.pivot(index="id", columns="strategy", values=["hit@5", "mrr"])
    out.append("## Fixed vs recursive: questions where the outcome differs")
    a, b = STRATEGIES[0], STRATEGIES[1]
    mrr_diff = wide["mrr"][b] - wide["mrr"][a]
    better_b, better_a = int((mrr_diff > 0).sum()), int((mrr_diff < 0).sum())
    ties = int((mrr_diff == 0).sum())
    out.append(f"- by reciprocal rank: {b} better on {better_b}, {a} better on {better_a}, tie on {ties} questions")
    out.append("")
    out.append(f"| Q | type | {a} hit@5 | {b} hit@5 | {a} RR | {b} RR | question |")
    out.append("|---|---|---|---|---|---|---|")
    for qid in mrr_diff[mrr_diff != 0].index:
        item = items[qid]
        out.append(
            f"| {qid} | {item['type']} | {wide['hit@5'][a][qid]:.0f} | {wide['hit@5'][b][qid]:.0f} | "
            f"{wide['mrr'][a][qid]:.2f} | {wide['mrr'][b][qid]:.2f} | {item['question'][:70]} |"
        )

    text = "\n".join(out)
    (RESULTS_DIR / "failure_analysis.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nSaved to {RESULTS_DIR / 'failure_analysis.md'}")


if __name__ == "__main__":
    main()
