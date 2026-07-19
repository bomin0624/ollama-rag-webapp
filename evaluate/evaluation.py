import argparse
import logging
import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from evaluate.utils import (
    build_retriever,
    chunks_to_initial_results,
    rerank_all_queries,
    run_retrieval,
    setup_logging,
)
from src.config import (
    DATASET_URL,
    DATASETS_DIR,
    DEFAULT_WORKERS,
    EMBED_TRUNCATE_DIM,
    RERANK_BATCH_SIZE,
    RETRIEVER_TYPE,
    SEARCH_K,
    VECTOR_DB_DIR,
)
from src.retriever.utils import initialize_vector_database


def evaluate_retriever(
    retriever_type: str = RETRIEVER_TYPE,
    split: str = "dev",
    workers: int = DEFAULT_WORKERS,
    rerank_batch_size: int = RERANK_BATCH_SIZE,
) -> None:
    """
    Evaluate the retriever performance using Recall@30
    for initial retrieval and NDCG@3 for reranked results.
    """
    setup_logging(retriever_type, split)

    data_path = util.download_and_unzip(DATASET_URL, str(DATASETS_DIR))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

    # print(queries.items()) # {'PLAIN-2':
    # 'Do Cholesterol Statin Drugs Cause Breast Cancer?'}
    # Ensure the vector database is initialized before evaluation
    db_directory = str(VECTOR_DB_DIR)
    initialize_vector_database(db_directory)

    ragretriever = build_retriever(retriever_type, db_directory)

    # Warm up the retriever so the first query's cold-start cost (lazy model
    # loading, CUDA kernel compilation) is excluded from the timed loop and
    # lazy initialization happens before worker threads start.
    warmup_query = next(iter(queries.values()))
    ragretriever.retriever.invoke(warmup_query)

    logging.info(f"--- Evaluating Initial Retrieval (Top {SEARCH_K}) ---")
    retrieval_start = time.perf_counter()
    retrieved = run_retrieval(ragretriever, queries, workers=workers)
    retrieval_time = time.perf_counter() - retrieval_start

    initial_results = {
        query_id: chunks_to_initial_results(chunks, SEARCH_K)
        for query_id, chunks in retrieved.items()
    }

    # Evaluate initial retrieval using Recall@30
    evaluator = EvaluateRetrieval()
    k_values_initial = [1, 5, 10, 30]
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, initial_results, k_values_initial
    )
    # print(evaluator.evaluate(qrels, initial_results, k_values_initial))
    # ({'NDCG@1': 0.41796, 'NDCG@5': 0.3661, 'NDCG@10': 0.2739,
    # 'NDCG@30': 0.20669},
    # {'MAP@1': 0.05611, 'MAP@5': 0.10866, 'MAP@10': 0.10866,
    # 'MAP@30': 0.10866},
    # {'Recall@1': 0.05611, 'Recall@5': 0.12886, 'Recall@10': 0.12886,
    # 'Recall@30': 0.12886},
    # {'P@1': 0.43963, 'P@5': 0.31641, 'P@10': 0.1582, 'P@30': 0.05273})
    logging.info("Initial Retrieval Metrics:")
    logging.info(f"NDCG@10: {ndcg['NDCG@10']}")
    logging.info(f"Recall@30: {recall['Recall@30']}")

    # Warm up the reranker for the same cold-start reason as the retriever.
    ragretriever.reranker.predict(
        [(warmup_query, warmup_query)], batch_size=rerank_batch_size
    )

    rerank_start = time.perf_counter()
    reranked_results = rerank_all_queries(
        ragretriever, queries, retrieved, batch_size=rerank_batch_size
    )
    rerank_time = time.perf_counter() - rerank_start

    # Evaluate reranked results using NDCG@3
    k_values_reranked = [1, 3, 5, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, reranked_results, k_values_reranked
    )
    logging.info("Reranked Retrieval Metrics:")
    logging.info(f"NDCG@3: {ndcg['NDCG@3']}")

    num_queries = len(queries)
    logging.info("--- Embedding Dimension & Timing ---")
    logging.info(f"Embedding truncate dim: {EMBED_TRUNCATE_DIM}")
    logging.info(
        f"Initial retrieval time: {retrieval_time:.2f}s total, "
        f"{retrieval_time / num_queries * 1000:.2f}ms/query throughput "
        f"({num_queries} queries, {workers} workers; per-query latency "
        f"only when workers=1)"
    )
    logging.info(
        f"Reranking time: {rerank_time:.2f}s total, "
        f"{rerank_time / num_queries * 1000:.2f}ms/query throughput "
        f"({num_queries} queries, batched across queries)"
    )
    logging.info(f"Total time: {retrieval_time + rerank_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Retriever Models")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Choose the retriever to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        default="dev",
        help="Choose the dataset split to evaluate on",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of concurrent retrieval workers; the bottleneck is "
            "the Ollama server, so also check OLLAMA_NUM_PARALLEL. "
            "Use 1 for sequential behaviour."
        ),
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=RERANK_BATCH_SIZE,
        help=(
            "Batch size for the cross-encoder reranker's predict() call. "
            "Larger values use more GPU memory but keep the GPU better "
            f"saturated. Defaults to RERANK_BATCH_SIZE ({RERANK_BATCH_SIZE})."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.rerank_batch_size < 1:
        parser.error("--rerank-batch-size must be >= 1")

    evaluate_retriever(
        retriever_type=args.retriever,
        split=args.split,
        workers=args.workers,
        rerank_batch_size=args.rerank_batch_size,
    )
