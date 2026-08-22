from functools import lru_cache
from typing import Protocol

from openai import OpenAI

from src.config import (
    GENERATE_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TIMEOUT,
)


class LLMClient(Protocol):
    """The one call the RAG flow needs from a chat backend."""

    def chat(self, prompt: str) -> str:
        """Return the assistant reply to a one-shot user ``prompt``."""
        ...


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
        if not result.choices:
            raise ValueError(f"{self.model} returned no choices.")

        choice = result.choices[0]
        content = choice.message.content
        if not content:
            raise ValueError(
                f"{self.model} returned no content "
                f"(finish_reason={choice.finish_reason!r})."
            )
        return content


# Cached so connection setup is paid only once. The configuration itself is
# read at import, so changing the environment after that has no effect.
@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Gets or creates the cached vLLM client."""
    return VLLMClient(
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        model=GENERATE_MODEL,
    )
