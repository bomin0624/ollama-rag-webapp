from types import SimpleNamespace

import pytest

from src.model import VLLMClient, get_llm_client


class FakeCompletions:
    """Stands in for OpenAI().chat.completions."""

    def __init__(
        self,
        content: str | None = "canned answer",
        finish_reason: str = "stop",
        choice_count: int = 1,
    ):
        self.content = content
        self.finish_reason = finish_reason
        self.choice_count = choice_count
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            finish_reason=self.finish_reason,
        )
        return SimpleNamespace(choices=[choice] * self.choice_count)


def build_vllm_client(completions: FakeCompletions, **kwargs) -> VLLMClient:
    """A VLLMClient whose transport is replaced by ``completions``."""
    client = VLLMClient(
        **{
            "base_url": "http://localhost:8000",
            "timeout": 5.0,
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            **kwargs,
        }
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client


@pytest.fixture(autouse=True)
def clear_llm_client_cache():
    """Keep the cached client from leaking between tests."""
    get_llm_client.cache_clear()
    yield
    get_llm_client.cache_clear()


def test_get_llm_client_builds_a_vllm_client():
    assert isinstance(get_llm_client(), VLLMClient)


def test_get_llm_client_caches_the_client():
    assert get_llm_client() is get_llm_client()


def test_vllm_client_sends_the_prompt_and_returns_the_content():
    completions = FakeCompletions("Statins are safe.")
    client = build_vllm_client(completions)

    answer = client.chat("Do statins cause cancer?")

    assert answer == "Statins are safe."
    assert completions.calls[0]["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert completions.calls[0]["messages"] == [
        {"role": "user", "content": "Do statins cause cancer?"}
    ]


@pytest.mark.parametrize("content", [None, ""], ids=["null", "empty-string"])
def test_vllm_client_raises_when_the_model_returns_no_content(content):
    client = build_vllm_client(
        FakeCompletions(content, finish_reason="length")
    )

    with pytest.raises(ValueError, match="returned no content"):
        client.chat("Do statins cause cancer?")


def test_vllm_client_raises_when_the_response_has_no_choices():
    client = build_vllm_client(FakeCompletions(choice_count=0))

    with pytest.raises(ValueError, match="returned no choices"):
        client.chat("Do statins cause cancer?")


@pytest.mark.parametrize("base_url", ["http://vllm:8000", "http://vllm:8000/"])
def test_vllm_client_targets_the_openai_compatible_path(base_url):
    client = VLLMClient(base_url=base_url, timeout=5.0, model="any-model")

    assert str(client.client.base_url).rstrip("/") == "http://vllm:8000/v1"
