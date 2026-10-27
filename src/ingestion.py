"""Load PDF / DOCX / TXT files into page-level records with metadata."""
import json
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from src.config import DOCUMENTS_JSON, RAW_DIR

# TXT / DOCX have no real pages, so we group paragraphs into ~3000-char "pages"
# to keep citations uniform: [source: file, page X].
PSEUDO_PAGE_CHARS = 3000


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_pseudo_pages(paragraphs, max_chars=PSEUDO_PAGE_CHARS):
    pages, current, size = [], [], 0
    for para in paragraphs:
        if size + len(para) > max_chars and current:
            pages.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para)
    if current:
        pages.append("\n\n".join(current))
    return pages


def load_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": i, "text": text})
    return pages


def load_docx(path: Path):
    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    texts = _split_into_pseudo_pages(paragraphs)
    return [{"page": i, "text": clean_text(t)} for i, t in enumerate(texts, start=1)]


def load_txt(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    texts = _split_into_pseudo_pages(paragraphs)
    return [{"page": i, "text": clean_text(t)} for i, t in enumerate(texts, start=1)]


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}


def ingest_all(raw_dir: Path = RAW_DIR):
    records = []
    for path in sorted(raw_dir.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"  SKIP {path.name} (unsupported type)")
            continue
        try:
            pages = loader(path)
        except Exception as exc:  # keep going if one file is broken
            print(f"  FAIL {path.name}: {exc}")
            continue
        for p in pages:
            records.append(
                {
                    "source": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "page": p["page"],
                    "text": p["text"],
                }
            )
        print(f"  OK   {path.name:45s} {len(pages):3d} pages")
    return records


def save_documents(records, path: Path = DOCUMENTS_JSON):
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(path: Path = DOCUMENTS_JSON):
    """Return page records; run ingestion first if the cache file is missing."""
    if not path.exists():
        print("documents.json not found - ingesting raw documents...")
        save_documents(ingest_all(), path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("Ingesting documents...")
    recs = ingest_all()
    save_documents(recs)
    n_docs = len({r["source"] for r in recs})
    n_chars = sum(len(r["text"]) for r in recs)
    print(f"\nDocuments: {n_docs} | Pages: {len(recs)} | Characters: {n_chars:,}")
    print(f"Saved to {DOCUMENTS_JSON}")
