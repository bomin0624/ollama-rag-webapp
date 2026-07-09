import logging

from fastapi import APIRouter, HTTPException, status

from src.generator import generate_response
from src.schemas import QueryRequest

router = APIRouter()
logger = logging.getLogger(__name__)


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
