dataset = "nfcorpus"
DATASET_URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

search_k = 200

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_URL = "http://localhost:11434"  # Ollama server URL

generate_model = "llama3.1"  # Model name for generation

retriever_type = "hybrid"  # Options: "hybrid" or "vector"
