import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # for defining request body models

from generator import DB_DIRECTORY, generate_response, initialize_vector_database

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
log_dir = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "webapp.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path, mode="a"), logging.StreamHandler()],
)




@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Phase ---
    logging.info("Server starting: Initializing vector database...")
    initialize_vector_database(DB_DIRECTORY)
    logging.info("Vector database is ready!")
    
    yield  # Yield control; FastAPI starts receiving and processing API requests
    
    # --- Shutdown Phase ---
    logging.info("Server shutting down: Cleaning up resources...")

app = FastAPI(lifespan=lifespan)


@app.get("/health")

def health_check():
    """Health check endpoint to verify the server is running."""
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query: str




def main():
    print("Hello from rag!")


if __name__ == "__main__":
    main()
