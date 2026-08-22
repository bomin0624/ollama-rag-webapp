from langchain_core.documents import Document
from langsmith import traceable

from src.config import GENERATE_MODEL
from src.model import LLMClient, get_llm_client
from src.retriever.retriever import (
    RAGRetriever,
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


@traceable(
    name="llm-chat",
    run_type="llm",
    metadata={"model": GENERATE_MODEL},
    process_inputs=lambda inputs: {"prompt": inputs["prompt"]},
)
def _chat(prompt: str, client: LLMClient) -> str:
    return client.chat(prompt)


@traceable(
    name="rag-query",
    run_type="chain",
    process_inputs=lambda inputs: {"query": inputs["query"]},
)
def generate_response_with_sources(
    query: str,
    retriever: RAGRetriever,
    client: LLMClient | None = None,
) -> tuple[str, list[Document]]:
    prompt, retrieved_docs = build_prompt(query, retriever=retriever)
    llm_client = get_llm_client() if client is None else client
    return _chat(prompt, llm_client), retrieved_docs
