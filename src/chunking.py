"""Two chunking strategies, applied page by page so every chunk keeps an exact page number.

A. fixed     - sliding window of N words with M words of overlap (ignores sentence boundaries)
B. recursive - paragraph -> sentence -> word fallback; packs whole sentences up to N words
               and carries the last sentence over as overlap (respects text structure)
"""
import re

from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    RECURSIVE_OVERLAP_SENTENCES,
    STRATEGIES,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def fixed_size_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def _to_units(text: str, max_words: int):
    """Split text into sentence-sized units (lists of words); hard-split very long ones."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        for sentence in _SENT_SPLIT.split(para):
            words = sentence.split()
            if not words:
                continue
            if len(words) > max_words:
                units.extend(words[i : i + max_words] for i in range(0, len(words), max_words))
            else:
                units.append(words)
    return units


def recursive_chunks(
    text: str,
    max_words: int = CHUNK_WORDS,
    overlap_sentences: int = RECURSIVE_OVERLAP_SENTENCES,
):
    units = _to_units(text, max_words)
    chunks, current, count = [], [], 0
    for unit in units:
        if current and count + len(unit) > max_words:
            chunks.append(" ".join(w for u in current for w in u))
            current = current[-overlap_sentences:] if overlap_sentences else []
            count = sum(len(u) for u in current)
            if count + len(unit) > max_words:  # overlap would overflow -> drop it
                current, count = [], 0
        current.append(unit)
        count += len(unit)
    if current:
        chunks.append(" ".join(w for u in current for w in u))
    return chunks


CHUNKERS = {"fixed": fixed_size_chunks, "recursive": recursive_chunks}
assert set(CHUNKERS) == set(STRATEGIES)


def chunk_documents(records, strategy: str):
    """records: page-level dicts from ingestion -> list of chunk dicts."""
    chunker = CHUNKERS[strategy]
    chunks = []
    for rec in records:
        for text in chunker(rec["text"]):
            chunks.append(
                {
                    "chunk_id": f"{strategy}-{len(chunks):05d}",
                    "strategy": strategy,
                    "source": rec["source"],
                    "page": rec["page"],
                    "text": text,
                    "n_words": len(text.split()),
                }
            )
    return chunks
