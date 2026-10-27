"""Thin Gemini wrapper with retry/backoff for rate limits (429) and temporary errors."""
import time
from functools import lru_cache

from src.config import GEMINI_API_KEY, LLM_MAX_RETRIES, LLM_MODEL


class LLMClient:
    def __init__(self, model: str = LLM_MODEL):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not found. Put it in the .env file (see .env.example).")
        from google import genai

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        delay = 8
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                return (response.text or "").strip()
            except Exception as exc:
                code = getattr(exc, "code", None)
                msg = str(exc)
                retryable = code in (429, 500, 502, 503, 504) or any(
                    s in msg for s in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
                )
                if not retryable or attempt == LLM_MAX_RETRIES:
                    raise
                print(f"  [LLM] {code or 'error'} - retry {attempt}/{LLM_MAX_RETRIES - 1} in {delay}s")
                time.sleep(delay)
                delay *= 2
        return ""


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
