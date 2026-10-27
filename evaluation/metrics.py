"""Retrieval and answer-quality metrics.

Retrieval relevance is chunk-level and strategy-independent: a retrieved chunk counts as
relevant when it comes from a ground-truth source document AND contains at least one of the
item's `evidence` strings. This lets us compare chunking strategies fairly, even though
their chunk IDs and boundaries differ.
"""
import re

from src.config import TOP_K_FINAL
from src.embeddings import embed_texts
from src.generator import extract_citations, strip_citations

KS = (1, 3, 5)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def as_list(value):
    return value if isinstance(value, list) else [value]


def is_doc_match(chunk, item) -> bool:
    return chunk["source"] in as_list(item["source_doc"])


def is_relevant(chunk, item) -> bool:
    if not is_doc_match(chunk, item):
        return False
    text = normalize(chunk["text"])
    return any(normalize(e) in text for e in item["evidence"])


def retrieval_scores(ranked, item, ks=KS) -> dict:
    """Hit@k, DocHit@k and reciprocal rank for one question."""
    ranked = ranked[:TOP_K_FINAL]
    relevant = [is_relevant(c, item) for c in ranked]
    doc_ok = [is_doc_match(c, item) for c in ranked]
    out = {}
    for k in ks:
        out[f"hit@{k}"] = float(any(relevant[:k]))
        out[f"doc_hit@{k}"] = float(any(doc_ok[:k]))
    out["mrr"] = next((1.0 / r for r, ok in enumerate(relevant, start=1) if ok), 0.0)
    return out


# ---------------------------------------------------------------- answer quality
JUDGE_PROMPT = """You are grading the answer of a question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score the system answer against the reference answer:
1   = correct: contains the key facts of the reference, no contradictions
0.5 = partially correct: contains some key facts but misses or gets wrong an important part
0   = incorrect, contradicts the reference, or states that the information was not found

Ignore citation markup like [source: ..., page ...] and differences in wording or length.
Reply in exactly this format:
SCORE: <0, 0.5 or 1>
REASON: <one short sentence>"""

_SCORE_RE = re.compile(r"SCORE:\s*(1(?:\.0)?|0\.5|0)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE | re.DOTALL)


def judge_answer(question: str, reference: str, answer: str, llm) -> dict:
    """LLM-as-judge: returns {'score': 0 | 0.5 | 1, 'reason': str}."""
    if not answer.strip():
        return {"score": 0.0, "reason": "empty answer"}
    reply = llm.generate(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer))
    m = _SCORE_RE.search(reply)
    reason = _REASON_RE.search(reply)
    return {
        "score": float(m.group(1)) if m else 0.0,
        "reason": reason.group(1).strip() if reason else ("unparsable judge reply" if not m else ""),
    }


def answer_similarity(answer: str, reference: str) -> float:
    """Cosine similarity between answer (citations stripped) and the reference answer."""
    a, r = strip_citations(answer), reference
    if not a:
        return 0.0
    emb = embed_texts([a, r])
    return float(emb[0] @ emb[1])


def citation_scores(answer: str, item, contexts) -> dict:
    """citation_correct: cites at least one ground-truth source.
    citation_grounded: every cited (source, page) really was among the retrieved chunks."""
    cites = extract_citations(answer)
    retrieved = {(c["source"], c["page"]) for c in contexts}
    correct = any(src in as_list(item["source_doc"]) for src, _ in cites)
    grounded = bool(cites) and all((src, pg) in retrieved for src, pg in cites)
    return {"citation_correct": float(correct), "citation_grounded": float(grounded)}
