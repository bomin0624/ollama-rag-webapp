import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.generator import generate_response

router = APIRouter()
logger = logging.getLogger(__name__)


# FastAPI data model used to define the request body structure
class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="User question for the RAG system. "
        "Must be between 1 and 1000 characters.",
    )


@router.get("/health", include_in_schema=False)
def health_check():
    """Health check endpoint to verify the server is running."""
    return {"status": "ok"}


@router.post("/query")
def query(request: QueryRequest):
    """Endpoint to handle user queries and return generated responses."""
    try:
        response = generate_response(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error occurred while processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
