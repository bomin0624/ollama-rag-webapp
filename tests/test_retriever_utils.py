from src.retriever.utils import build_bm25_documents


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
