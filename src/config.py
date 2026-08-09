import os
from pathlib import Path

# Project paths, all derived from the repository root so callers never have to
# count ".." / parents[N] levels relative to their own file location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = PROJECT_ROOT / "vectordatabase"
DATASETS_DIR = PROJECT_ROOT / "datasets"
LOG_DIR = PROJECT_ROOT / "log"

DATASET = "nfcorpus"
DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
    f"{DATASET}.zip"
)

# Number of candidates fetched by the initial retriever. Kept large for recall.
SEARCH_K = 200

# Batch size for the cross-encoder reranker's predict() call. Larger values use
# more GPU memory but keep the GPU better saturated.
RERANK_BATCH_SIZE = 64

# Number of final documents returned after reranking.
RERANK_RETURN_N = 3

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Chat backend used for answer generation. Backends are registered in
# LLM_CLIENT_CLASSES in src/model.py, which also validates this value.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

# Model name for generation, spelled the way the backend knows it.
GENERATE_MODEL = os.getenv("GENERATE_MODEL", "llama3.1")

RETRIEVER_TYPE = "hybrid"  # Options: "hybrid" or "vector"

# Dimensionality to truncate embeddings to for
# efficiency in storage and retrieval. Overridable via the EMBED_TRUNCATE_DIM
# environment variable (e.g. `make evaluate DIM=256`).
EMBED_TRUNCATE_DIM = int(os.environ.get("EMBED_TRUNCATE_DIM", "512"))

# Chunking parameters used when building the vector database.
# max_length * 4 = chunk_size; chunk_overlap = chunk_size * 0.10 ~ 0.25
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 300
