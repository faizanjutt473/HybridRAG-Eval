"""Answer generation with mandatory source citations."""
import re

from src.llm import LLMClient, get_llm

NOT_FOUND = "Not found in documents."

SYSTEM_PROMPT = f"""You are a careful question-answering assistant for a collection of documents.
Rules:
1. Answer ONLY from the numbered context passages. Never use outside knowledge.
2. After every factual statement, cite the passage it came from in exactly this format:
   [source: <filename>, page <number>]
3. If several passages support a statement, cite each of them.
4. If the passages do not contain the answer, reply exactly: {NOT_FOUND}
5. Be concise: 1-4 sentences."""

# Matches "file.pdf, page 3" anywhere (also inside combined citations).
CITATION_RE = re.compile(r"([\w\-.]+\.(?:pdf|docx|txt))\s*,\s*pages?\s*(\d+)", re.IGNORECASE)
CITATION_MARKUP_RE = re.compile(r"\[source:[^\]]*\]", re.IGNORECASE)


def build_context(chunks) -> str:
    blocks = [
        f"[{i}] (source: {c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks, llm: LLMClient | None = None) -> str:
    llm = llm or get_llm()
    prompt = f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
    return llm.generate(prompt, system=SYSTEM_PROMPT)


def extract_citations(answer: str):
    """Return a list of (source_filename, page_int) cited in the answer."""
    return [(m.group(1), int(m.group(2))) for m in CITATION_RE.finditer(answer)]


def strip_citations(answer: str) -> str:
    return CITATION_MARKUP_RE.sub("", answer).strip()
