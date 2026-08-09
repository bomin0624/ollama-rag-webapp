from functools import lru_cache
from typing import Protocol

import ollama

from src.config import (
    GENERATE_MODEL,
    LLM_BACKEND,
    LLM_BASE_URL,
    LLM_TIMEOUT,
)


class LLMClient(Protocol):
    """The one call the RAG flow needs from a chat backend."""

    def chat(self, prompt: str) -> str:
        """Return the assistant reply to a one-shot user ``prompt``."""
        ...


class OllamaClient:
    """``LLMClient`` backed by the native Ollama chat API."""

    def __init__(self, base_url: str, timeout: float, model: str):
        self.client = ollama.Client(host=base_url, timeout=timeout)
        self.model = model

    def chat(self, prompt: str) -> str:
        result = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return result["message"]["content"]


LLM_CLIENT_CLASSES = {
    "ollama": OllamaClient,
}


# Built lazily so that importing this module never reads backend
# configuration, and cached so connection setup is paid only once.
@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Gets or creates the cached client for ``LLM_BACKEND``."""
    try:
        client_class = LLM_CLIENT_CLASSES[LLM_BACKEND]
    except KeyError as e:
        allowed = ", ".join(LLM_CLIENT_CLASSES)
        raise ValueError(
            f"Invalid LLM_BACKEND: {LLM_BACKEND!r}. Choose one of: {allowed}"
        ) from e

    return client_class(
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        model=GENERATE_MODEL,
    )
