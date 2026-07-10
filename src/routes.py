import logging

from fastapi import APIRouter, HTTPException, status
from langchain_core.documents import Document

from src.config import GENERATE_MODEL
from src.generator import generate_response_with_sources
from src.schemas import QueryRequest, QueryResponse, SourceDocument

router = APIRouter()
logger = logging.getLogger(__name__)


def build_source_document(doc: Document) -> SourceDocument:
    metadata = doc.metadata or {}
    doc_id = metadata.get("id")

    return SourceDocument(
        id=str(doc_id) if doc_id is not None else "unknown_id",
        title=metadata.get("title"),
        content=doc.page_content,
    )


@router.get("/health", include_in_schema=False)
def health_check():
    """Health check endpoint to verify the server is running."""
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Endpoint to handle user queries and return generated responses."""
    try:
        answer, retrieved_docs = generate_response_with_sources(request.query)
        sources = [build_source_document(doc) for doc in retrieved_docs]

        return QueryResponse(
            answer=answer,
            sources=sources,
            model=GENERATE_MODEL,
        )

    except Exception as e:
        logger.exception("Error occurred while processing query")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
