from functools import lru_cache
from typing import Protocol

import ollama
from openai import OpenAI

from src.config import (
    GENERATE_MODEL,
    LLM_API_KEY,
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


class VLLMClient:
    """``LLMClient`` backed by vLLM's OpenAI-compatible ``/v1`` server."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        model: str,
        api_key: str = LLM_API_KEY,
    ):
        self.client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model

    def chat(self, prompt: str) -> str:
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = result.choices[0]
        if choice.message.content is None:
            # Happens when the model emits no text at all, e.g. it hit the
            # token limit first. Returning "" here would look like an answer.
            raise ValueError(
                f"{self.model} returned no content "
                f"(finish_reason={choice.finish_reason!r})."
            )
        return choice.message.content


LLM_CLIENT_CLASSES = {
    "ollama": OllamaClient,
    "vllm": VLLMClient,
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
