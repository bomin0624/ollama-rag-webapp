import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.generator import DB_DIRECTORY, initialize_vector_database
from src.routes import router


def configure_logging() -> None:
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
    # --- Startup Phase ---
    logging.info("Server starting: Initializing vector database...")
    initialize_vector_database(DB_DIRECTORY)
    logging.info("Vector database is ready!")
    # Yield control; FastAPI starts receiving and processing API requests
    yield

    # --- Shutdown Phase ---
    logging.info("Server shutting down: Cleaning up resources...")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()

# TO:DO
# add post 查詢路由 input: query string, output: generated response
# front-end
# Evaluation: RAGAS


def main():
    print("Hello from rag!")


if __name__ == "__main__":
    main()
