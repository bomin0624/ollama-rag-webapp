import os
from functools import lru_cache

import ollama
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL,
    GENERATE_MODEL,
    RERANKER_MODEL,
    SEARCH_K,
)
from src.retriever import RAGRetriever, initialize_vector_database

DB_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "vectordatabase")


# Using LRU cache to store the retriever instance
# for efficient reuse across multiple calls
@lru_cache(maxsize=1)
def get_retriever(db_directory: str) -> RAGRetriever:
    """Gets or creates a cached RAGRetriever instance."""
    return RAGRetriever(
        db_directory=db_directory,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        search_k=SEARCH_K,
    )


def build_prompt(query: str) -> tuple[str, list[Document]]:
    prompt = (
        f"\nBased on the following query: {query} and "
        "the context provided below to give the user answer.\n"
    )
    # Ensure the database is initialized (idempotent check)
    initialize_vector_database(DB_DIRECTORY)
    retriever = get_retriever(DB_DIRECTORY)
    retrieved_docs = retriever.retrieve_and_rerank(query)

    if not retrieved_docs:
        prompt += "\nNo relevant documents found.\n"
        return prompt, retrieved_docs

    for doc in retrieved_docs:
        metadata = doc.metadata or {}
        document_id = metadata.get("id", "unknown_id")
        prompt += f"\nDocument {document_id}:\n{doc.page_content}\n"

    return prompt, retrieved_docs


def generate_response_with_sources(query: str) -> tuple[str, list[Document]]:
    prompt, retrieved_docs = build_prompt(query)
    result = ollama.chat(
        model=GENERATE_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return result["message"]["content"], retrieved_docs
