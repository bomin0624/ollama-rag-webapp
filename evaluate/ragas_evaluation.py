"""Evaluate faithfulness and answer relevancy of generated responses."""

import asyncio
import os
import random

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src.config import DATASET_URL, DATASETS_DIR, RANDOM_SEED, VECTOR_DB_DIR
from src.generator import generate_response_with_sources
from src.retriever.retriever import get_retriever
from src.retriever.utils import initialize_vector_database

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to the project's .env file or "
        "export it before running this module."
    )

client = AsyncOpenAI(api_key=openai_api_key)

judge_llm = llm_factory(
    "gpt-4o-mini-2024-07-18", provider="openai", client=client
)

judge_embeddings = HuggingFaceEmbeddings(
    model="mixedbread-ai/mxbai-embed-large-v1",
    device="cuda",
)


faithfulness = Faithfulness(llm=judge_llm)

relevancy = AnswerRelevancy(
    llm=judge_llm,
    embeddings=judge_embeddings,
)


async def main():
    # using nfcorpus test dataset's query
    data_path = util.download_and_unzip(DATASET_URL, str(DATASETS_DIR))
    _, queries, _ = GenericDataLoader(data_path).load(split="test")

    random.seed(RANDOM_SEED)

    random_queries = random.sample(list(queries.items()), 10)
    db_directory = str(VECTOR_DB_DIR)
    initialize_vector_database(db_directory)
    retriever = get_retriever(str(VECTOR_DB_DIR))
    for query_id, query_text in enumerate(random_queries):
        answer, documents = generate_response_with_sources(
            query_text,
            retriever=retriever,
        )

        contexts = [document.page_content for document in documents]

        faithfulness_result = await faithfulness.ascore(
            user_input=query_text,
            response=answer,
            retrieved_contexts=contexts,
        )

        relevancy_result = await relevancy.ascore(
            user_input=query_text,
            response=answer,
        )
        print("-------------------------------------")
        print("Query ID:", query_id)
        print("Query text:", query_text)
        print("Faithfulness score:", faithfulness_result.value)
        print("Relevancy score:", relevancy_result.value)
        print("-------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
