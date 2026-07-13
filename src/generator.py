from functools import lru_cache

import ollama
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL,
    GENERATE_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    RERANKER_MODEL,
    RETRIEVER_TYPE,
    SEARCH_K,
    VECTOR_DB_DIR,
)
from src.retriever.retriever import HybridRetriever, RAGRetriever
from src.retriever.utils import initialize_vector_database

DB_DIRECTORY = str(VECTOR_DB_DIR)
RETRIEVER_CLASSES = {
    "vector": RAGRetriever,
    "hybrid": HybridRetriever,
}
ollama_client = ollama.Client(
    host=OLLAMA_URL,
    timeout=OLLAMA_TIMEOUT,
)


# Using LRU cache to store the retriever instance
# for efficient reuse across multiple calls
@lru_cache(maxsize=1)
def get_retriever(db_directory: str) -> RAGRetriever:
    """Gets or creates a cached RAGRetriever instance."""
    try:
        retriever_class = RETRIEVER_CLASSES[RETRIEVER_TYPE]
    except KeyError as e:
        allowed = ", ".join(RETRIEVER_CLASSES)
        raise ValueError(
            f"Invalid RETRIEVER_TYPE: {RETRIEVER_TYPE!r}. "
            f"Choose one of: {allowed}"
        ) from e

    initialize_vector_database(DB_DIRECTORY)

    return retriever_class(
        db_directory=db_directory,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        search_k=SEARCH_K,
    )


def build_prompt(
    query: str, retriever: RAGRetriever | None = None
) -> tuple[str, list[Document]]:
    """Build the prompt for ``query``.

    ``retriever`` defaults to the shared cached one; pass an explicit
    retriever to build a prompt without touching the vector database.
    """
    prompt = (
        f"\nBased on the following query: {query} and "
        "the context provided below to give the user answer.\n"
    )
    # Use the shared cached retriever when one is not injected.
    if retriever is None:
        retriever = get_retriever(DB_DIRECTORY)
    retrieved_docs: list[Document] = retriever.retrieve_and_rerank(query)

    if not retrieved_docs:
        prompt += "\nNo relevant documents found.\n"
        return prompt, retrieved_docs

    for doc in retrieved_docs:
        metadata = doc.metadata or {}
        doc_id = metadata.get("id")
        document_id = str(doc_id) if doc_id is not None else "unknown_id"
        prompt += f"\nDocument {document_id}:\n{doc.page_content}\n"

    return prompt, retrieved_docs


def generate_response_with_sources(query: str) -> tuple[str, list[Document]]:
    prompt, retrieved_docs = build_prompt(query)
    result = ollama_client.chat(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return result["message"]["content"], retrieved_docs
