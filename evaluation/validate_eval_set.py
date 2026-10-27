"""Check that every question's source file exists and its evidence string is really in the text.

Run this BEFORE the evaluation:   python -m evaluation.validate_eval_set
If an item fails, open that document, find the sentence that answers the question and
edit the `evidence` list (short, distinctive phrases) in evaluation/eval_set.json.
"""
import json
import sys
from collections import defaultdict

from evaluation.metrics import as_list, normalize
from src.config import EVAL_SET_PATH
from src.ingestion import load_documents


def main() -> int:
    items = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    pages = defaultdict(list)
    for rec in load_documents():
        pages[rec["source"]].append(rec["text"])
    full_text = {src: normalize(" ".join(texts)) for src, texts in pages.items()}

    problems, seen_ids = [], set()
    required = {"id", "type", "question", "ground_truth_answer", "source_doc", "evidence"}
    for item in items:
        missing = required - set(item)
        if missing:
            problems.append((item.get("id"), f"missing keys: {sorted(missing)}"))
            continue
        if item["id"] in seen_ids:
            problems.append((item["id"], "duplicate id"))
        seen_ids.add(item["id"])

        sources = as_list(item["source_doc"])
        unknown = [s for s in sources if s not in full_text]
        if unknown:
            problems.append((item["id"], f"source file not found in corpus: {unknown}"))
            continue
        if not any(normalize(e) in full_text[s] for s in sources for e in item["evidence"]):
            problems.append((item["id"], f"evidence not found in {sources}: {item['evidence']}"))

    print(f"{len(items)} questions checked, {len(problems)} problem(s)")
    for qid, msg in problems:
        print(f"  [Q{qid}] {msg}")
    if len(items) < 30:
        print("WARNING: fewer than 30 questions.")
    if not problems:
        print("Eval set is consistent with the corpus.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
