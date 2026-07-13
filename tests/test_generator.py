from langchain_core.documents import Document

from src.generator import build_prompt


class FakeRetriever:
    """Stands in for the RAGRetriever, returning a fixed result set."""

    def __init__(self, documents: list[Document]):
        self.documents = documents

    def retrieve_and_rerank(self, query: str) -> list[Document]:
        return self.documents


def test_build_prompt_includes_query_and_retrieved_documents():
    documents = [
        Document(
            page_content="Statins are well tolerated.",
            metadata={"id": "MED-335"},
        ),
        Document(
            page_content="Leucine restriction extends lifespan.",
            metadata={"id": "MED-336"},
        ),
    ]

    prompt, retrieved_docs = build_prompt(
        "Do statins cause cancer?", retriever=FakeRetriever(documents)
    )

    assert "Do statins cause cancer?" in prompt
    assert "Document MED-335:\nStatins are well tolerated." in prompt
    assert "Document MED-336:\nLeucine restriction extends lifespan." in prompt
    assert retrieved_docs == documents


def test_build_prompt_reports_when_nothing_is_retrieved():
    prompt, retrieved_docs = build_prompt(
        "Do statins cause cancer?", retriever=FakeRetriever([])
    )

    assert "No relevant documents found." in prompt
    assert retrieved_docs == []


def test_build_prompt_falls_back_to_unknown_id():
    documents = [Document(page_content="Untagged chunk.", metadata={})]

    prompt, _ = build_prompt(
        "Do statins cause cancer?", retriever=FakeRetriever(documents)
    )

    assert "Document unknown_id:\nUntagged chunk." in prompt
