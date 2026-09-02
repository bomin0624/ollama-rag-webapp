"""Evaluate faithfulness and answer relevancy of generated responses."""

import asyncio
import logging
import os
import random
from datetime import datetime

import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src.config import (
    DATASET_URL,
    DATASETS_DIR,
    LOG_DIR,
    RANDOM_SEED,
    VECTOR_DB_DIR,
)
from src.generator import generate_response_with_sources
from src.retriever.retriever import get_retriever
from src.retriever.utils import initialize_vector_database

load_dotenv()


def configure_logging() -> None:
    """Log to the console and a timestamped file for this evaluation run."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = LOG_DIR / f"ragas_evaluation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def setting_evaluation():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the project's .env file."
        )

    client = AsyncOpenAI(api_key=openai_api_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    judge_llm = llm_factory(
        "gpt-4o-mini-2024-07-18",
        provider="openai",
        client=client,
        max_tokens=4096,
    )

    judge_embeddings = HuggingFaceEmbeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
        device=device,
    )
    faithfulness = Faithfulness(llm=judge_llm)

    # Use the judge LLM and embeddings to evaluate answer relevancy
    relevancy = AnswerRelevancy(
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return faithfulness, relevancy


async def main():
    data_path = util.download_and_unzip(DATASET_URL, str(DATASETS_DIR))
    _, queries, _ = GenericDataLoader(data_path).load(split="test")

    random.seed(RANDOM_SEED)

    random_queries = random.sample(list(queries.items()), 10)
    db_directory = str(VECTOR_DB_DIR)
    initialize_vector_database(db_directory)
    retriever = get_retriever(str(VECTOR_DB_DIR))
    faithfulness, relevancy = setting_evaluation()
    for query_id, query_text in random_queries:
        answer, documents = generate_response_with_sources(
            query_text,
            retriever=retriever,
        )

        contexts = [document.page_content for document in documents]

        faithfulness_result, relevancy_result = await asyncio.gather(
            faithfulness.ascore(
                user_input=query_text,
                response=answer,
                retrieved_contexts=contexts,
            ),
            relevancy.ascore(
                user_input=query_text,
                response=answer,
            ),
        )
        logging.info("-------------------------------------")
        logging.info("Query ID: %s", query_id)
        logging.info("Query text: %s", query_text)
        logging.info("Faithfulness score: %s", faithfulness_result.value)
        logging.info("Relevancy score: %s", relevancy_result.value)
        logging.info("-------------------------------------")


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
