from langchain_core.documents import Document

from src.retriever.utils import build_bm25_documents, rerank_documents


class FakeReranker:
    """Stands in for the CrossEncoder, scoring pairs in a fixed order."""

    def __init__(self, scores: list[float]):
        self.scores = scores

    def predict(self, pairs, batch_size: int) -> list[float]:
        return self.scores[: len(pairs)]


def test_build_bm25_documents_keeps_metadata_id():
    collection = {
        "ids": ["chunk-1", "chunk-2"],
        "documents": ["Statins and breast cancer.", "Leucine intake."],
        "metadatas": [
            {"id": "MED-335", "title": "Statins"},
            {"id": "MED-336", "title": "Leucine"},
        ],
    }

    documents = build_bm25_documents(collection)

    assert len(documents) == 2
    assert documents[0].page_content == "Statins and breast cancer."
    assert documents[0].metadata["id"] == "MED-335"
    assert documents[1].metadata["id"] == "MED-336"


def test_build_bm25_documents_falls_back_to_collection_id():
    collection = {
        "ids": ["chunk-1", "chunk-2"],
        "documents": ["Statins and breast cancer.", "Leucine intake."],
        "metadatas": [
            {"title": "Statins"},
            {"id": None, "title": "Leucine"},
        ],
    }

    documents = build_bm25_documents(collection)

    assert [doc.metadata["id"] for doc in documents] == ["chunk-1", "chunk-2"]


def test_build_bm25_documents_skips_empty_text():
    collection = {
        "ids": ["chunk-1", "chunk-2", "chunk-3"],
        "documents": ["Statins.", "", None],
        "metadatas": [
            {"id": "MED-335"},
            {"id": "MED-336"},
            {"id": "MED-337"},
        ],
    }

    documents = build_bm25_documents(collection)

    assert [doc.metadata["id"] for doc in documents] == ["MED-335"]


def test_rerank_documents_returns_top_n_by_score():
    chunks = [
        Document(page_content="low", metadata={"id": "MED-1"}),
        Document(page_content="high", metadata={"id": "MED-2"}),
        Document(page_content="middle", metadata={"id": "MED-3"}),
    ]
    reranker = FakeReranker(scores=[0.1, 0.9, 0.5])

    reranked = rerank_documents(
        query="statins",
        retrieved_chunks=chunks,
        reranker_model=reranker,
        top_n=2,
    )

    assert [doc.metadata["id"] for doc in reranked] == ["MED-2", "MED-3"]
