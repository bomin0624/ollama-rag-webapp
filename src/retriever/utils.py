from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import EMBED_TRUNCATE_DIM


def build_bm25_documents(collection: dict) -> list[Document]:
    """Build BM25 documents with a stable metadata id on every document."""
    documents = collection["documents"]
    collection_ids = collection["ids"]
    metadatas = collection["metadatas"] or []

    bm25_documents = []
    for index, text in enumerate(documents):
        if not text:
            continue

        metadata = (
            dict(metadatas[index] or {}) if index < len(metadatas) else {}
        )
        if "id" not in metadata or metadata["id"] is None:
            metadata["id"] = collection_ids[index]

        bm25_documents.append(Document(page_content=text, metadata=metadata))

    return bm25_documents


def rerank_documents(
    query: str,
    retrieved_chunks: list[Document],
    reranker_model: CrossEncoder,
    top_n: int,
) -> list[Document]:
    """
    Rerank retrieved chunks and keep the top chunk for each unique document.
    """
    if not retrieved_chunks:
        return []
    pairs = [(query, chunk.page_content) for chunk in retrieved_chunks]
    scores = reranker_model.predict(pairs)
    # List of tuples [(score, Document), (score, Document), ...]
    scored_docs = sorted(
        zip(scores, retrieved_chunks, strict=False),
        key=lambda x: x[0],
        reverse=True,
    )

    unique_docs = []
    seen_ids = set()
    for _, doc in scored_docs:
        if doc.metadata["id"] not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc.metadata["id"])
        if len(unique_docs) >= top_n:
            break

    return unique_docs


def build_embedding(model_name: str) -> HuggingFaceEmbeddings:
    model_kwargs = {}
    if EMBED_TRUNCATE_DIM is not None:
        model_kwargs["truncate_dim"] = EMBED_TRUNCATE_DIM
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

