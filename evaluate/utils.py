import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL,
    LOG_DIR,
    RERANKER_MODEL,
    SEARCH_K,
)
from src.retriever.retriever import (
    HybridRetriever,
    RAGRetriever,
)


def setup_logging(retriever_type: str, split: str) -> None:
    """Configure logging to both a per-run file and stdout."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = str(LOG_DIR / f"nfcorpus_{retriever_type}_{split}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def build_retriever(retriever_type: str, db_directory: str) -> RAGRetriever:
    """Instantiate the retriever under evaluation."""
    if retriever_type == "hybrid":
        retriever_class = HybridRetriever
    elif retriever_type == "vector":
        retriever_class = RAGRetriever
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    return retriever_class(
        db_directory=db_directory,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        search_k=SEARCH_K,
    )


def run_retrieval(
    ragretriever: RAGRetriever,
    queries: dict[str, str],
    workers: int,
) -> dict[str, list[Document]]:
    """
    Retrieve chunks for every query, running up to `workers` queries
    concurrently. Query embedding (torch, releases the GIL) and the
    Chroma search overlap across threads; the BM25 part is pure Python
    and stays serialized by the GIL. `workers=1` reproduces sequential
    behaviour.
    """

    def _retrieve(item: tuple[str, str]) -> tuple[str, list[Document]]:
        query_id, query_text = item
        return query_id, ragretriever.retriever.invoke(query_text)

    retrieved: dict[str, list[Document]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for query_id, chunks in tqdm(
            executor.map(_retrieve, queries.items()),
            total=len(queries),
            desc="Initial Retriever",
        ):
            retrieved[query_id] = chunks
    return retrieved


def chunks_to_initial_results(
    retrieved_chunks: list[Document], search_k: int = SEARCH_K
) -> dict[str, float]:
    """
    Convert ranked chunks into the doc_id -> score mapping BEIR expects,
    using reciprocal rank as the score and keeping the first occurrence
    of each document.
    """
    query_results: dict[str, float] = {}
    for rank, chunk in enumerate(retrieved_chunks):
        doc_id = chunk.metadata["id"]
        if doc_id not in query_results:
            query_results[doc_id] = 1.0 / (rank + 1)
    return dict(
        sorted(query_results.items(), key=lambda x: x[1], reverse=True)[
            :search_k
        ]
    )


def rerank_all_queries(
    ragretriever: RAGRetriever,
    queries: dict[str, str],
    retrieved: dict[str, list[Document]],
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """
    Rerank every query's chunks in one flattened pass so the
    cross-encoder always sees full `batch_size` batches, then scatter
    the scores back per query.
    """
    pairs: list[tuple[str, str]] = []
    offsets: dict[str, tuple[int, int]] = {}
    for query_id, query_text in queries.items():
        chunks = retrieved[query_id]
        start = len(pairs)
        pairs.extend((query_text, chunk.page_content) for chunk in chunks)
        offsets[query_id] = (start, len(pairs))

    scores = ragretriever.reranker.predict(
        pairs, batch_size=batch_size, show_progress_bar=True
    )

    reranked_results: dict[str, dict[str, float]] = {}
    for query_id, (start, end) in offsets.items():
        chunks = retrieved[query_id]
        if not chunks:
            reranked_results[query_id] = {}
            continue
        scored_chunks = sorted(
            zip(scores[start:end], chunks, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
        query_results: dict[str, float] = {}
        for score, chunk in scored_chunks:
            doc_id = chunk.metadata["id"]
            if doc_id not in query_results:
                query_results[doc_id] = float(score)
        reranked_results[query_id] = query_results
    return reranked_results
