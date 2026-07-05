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

SEARCH_K = 200

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_URL = "http://localhost:11434"  # Ollama server URL

GENERATE_MODEL = "llama3.1"  # Model name for generation

RETRIEVER_TYPE = "hybrid"  # Options: "hybrid" or "vector"

# Dimensionality to truncate embeddings to for
# efficiency in storage and retrieval
EMBED_TRUNCATE_DIM = 256
