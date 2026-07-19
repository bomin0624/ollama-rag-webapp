import ollama
from langchain_core.documents import Document
from langsmith import traceable

from src.config import (
    GENERATE_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
)
from src.retriever.retriever import (
    RAGRetriever,
)

ollama_client = ollama.Client(
    host=OLLAMA_URL,
    timeout=OLLAMA_TIMEOUT,
)


def build_prompt(
    query: str, retriever: RAGRetriever
) -> tuple[str, list[Document]]:
    """Build the prompt for ``query`` using the provided retriever."""
    prompt = (
        f"\nBased on the following query: {query} and "
        "the context provided below to give the user answer.\n"
    )
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


@traceable(name="rag-query", run_type="chain")
def generate_response_with_sources(
    query: str, retriever: RAGRetriever
) -> tuple[str, list[Document]]:
    prompt, retrieved_docs = build_prompt(query, retriever=retriever)
    result = ollama_client.chat(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return result["message"]["content"], retrieved_docs
