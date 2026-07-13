from src.schemas import QueryRequest


def test_query_request_accepts_valid_query():
    request = QueryRequest(query="What causes high cholesterol?")

    assert request.query == "What causes high cholesterol?"
