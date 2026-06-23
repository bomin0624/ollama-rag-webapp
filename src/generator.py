import os
from functools import lru_cache

import ollama

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


def generate_prompt_stream(query: str) -> str:
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
        return prompt
    else:
        for _idx, doc in enumerate(retrieved_docs):
            # print(f"\n--- Document {idx + 1} ---")
            # print(f"Content: {doc.page_content[:250]}...")
            # print(f"Metadata: {doc.metadata}")
            prompt += f"\nDocument {doc.metadata['id']}:\n{doc.page_content}\n"
    return prompt


def generate_response(query: str) -> str:
    prompt = generate_prompt_stream(query)
    result = ollama.chat(
        model=GENERATE_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return result["message"]["content"]


# if __name__ == "__main__":
#     query = input("Please enter your query: ")
#     response = generate_response(query)
#     print(response)
