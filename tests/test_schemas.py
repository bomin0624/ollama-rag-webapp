import pytest
from pydantic import ValidationError

from src.schemas import QueryRequest


def test_query_request_accepts_valid_query():
    request = QueryRequest(query="What causes high cholesterol?")

    assert request.query == "What causes high cholesterol?"


def test_query_request_strips_surrounding_whitespace():
    request = QueryRequest(query="  What causes high cholesterol?  ")

    assert request.query == "What causes high cholesterol?"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"query": ""}, "empty"),
        ({"query": "   "}, "whitespace only, stripped to empty"),
        ({"query": "a" * 1001}, "longer than max_length"),
        (
            {"query": "valid", "top_k": 5},
            "unknown field, extra is forbidden",
        ),
    ],
    ids=["empty", "whitespace_only", "too_long", "extra_field"],
)
def test_query_request_rejects_invalid_payload(kwargs: dict, reason: str):
    with pytest.raises(ValidationError):
        QueryRequest(**kwargs)
