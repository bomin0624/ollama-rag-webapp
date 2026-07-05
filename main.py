"""FastAPI entry point for the RAG web app.

Configures logging, initializes and warms the retriever during the lifespan
startup phase, and exposes the API routes.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.generator import (
    DB_DIRECTORY,
    get_retriever,
    initialize_vector_database,
)
from src.routes import router


def configure_logging() -> None:
    """Log to both the console and log/webapp.log."""
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "webapp.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="a"),
            logging.StreamHandler(),
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown.

    Startup runs before ``yield``: build the vector database if needed, then
    warm the retriever so the first query does not pay the cost of loading the
    embedding model, reranker, and BM25 index on demand. Shutdown runs after
    ``yield``; uvicorn already drains in-flight requests, and this app holds no
    resources that need explicit teardown.
    """
    # --- Startup ---
    logging.info("Server starting: initializing vector database...")
    initialize_vector_database(DB_DIRECTORY)
    logging.info("Vector database is ready.")

    logging.info("Warming up retriever (models + BM25 index)...")
    get_retriever(DB_DIRECTORY)
    logging.info("Retriever is ready.")

    yield

    # --- Shutdown ---
    logging.info("Server shutting down.")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="RAG Web App", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
