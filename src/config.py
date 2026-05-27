DATASET = "nfcorpus"
DATASET_URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"

SEARCH_K = 200

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_URL = "http://localhost:11434"  # Ollama server URL

GENERATE_MODEL = "llama3.1"  # Model name for generation

RETRIEVER_TYPE = "hybrid"  # Options: "hybrid" or "vector"
