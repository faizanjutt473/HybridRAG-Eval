"""Central configuration for the HybridRAG-Eval project."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- Paths ----
RAW_DIR = ROOT / "data" / "raw_docs"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "indexes"
DOCUMENTS_JSON = PROCESSED_DIR / "documents.json"

EVAL_DIR = ROOT / "evaluation"
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RESULTS_DIR = EVAL_DIR / "results"

for _d in (PROCESSED_DIR, INDEX_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Models ----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# gemini-2.5-* models were shut down on 16 Oct 2026, so we default to 3.5 Flash.
# You can override this from .env with GEMINI_MODEL=...
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MAX_RETRIES = 5

# ---- Chunking (sizes are in words) ----
STRATEGIES = ("fixed", "recursive")
CHUNK_WORDS = 150          # max chunk size for both strategies
CHUNK_OVERLAP_WORDS = 30   # fixed strategy: word overlap between windows
RECURSIVE_OVERLAP_SENTENCES = 1  # recursive strategy: sentences carried over

# ---- Retrieval ----
TOP_K_CANDIDATES = 20      # candidates taken from each retriever before fusion
TOP_K_RERANK_INPUT = 20    # candidates passed to the cross-encoder
TOP_K_FINAL = 5            # chunks given to the LLM / used for Hit@k
RRF_K = 60                 # reciprocal-rank-fusion constant
