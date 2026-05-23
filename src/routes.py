from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.generator import generate_response



router = APIRouter()


# FastAPI data model used to define the request body structure
class QueryRequest(BaseModel):
    query: str


@router.get("/health")
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
        raise HTTPException(status_code=500, detail=str(e))