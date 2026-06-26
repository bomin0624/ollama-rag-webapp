# RAG Web App

This is a RAG web app that exposes a query API with FastAPI. The system builds a Chroma vector database from a BEIR dataset, retrieves relevant documents for each query, formats the retrieved context into a prompt, and uses Ollama to call a local LLM for answer generation.

## Architecture Overview

```mermaid
flowchart TD
		Client[Client] --> API[FastAPI /query]
		API --> Generator[src/generator.py]
		Generator --> Retriever[src/retriever.py]
		Retriever --> Chroma[(Chroma vector database)]
		Retriever --> Reranker[CrossEncoder reranker]
		Generator --> Ollama[Ollama local LLM]
		Ollama --> API

		Dataset[BEIR dataset] --> InitDB[initialize_vector_database]
		InitDB --> Splitter[RecursiveCharacterTextSplitter]
		Splitter --> Embedding[HuggingFace embeddings]
		Embedding --> Chroma
```

## Main Modules

- `main.py`: FastAPI app entry point. It configures logging and ensures the vector database is initialized during the lifespan startup phase.
- `src/routes.py`: API route definitions. It currently provides `GET /health` and `POST /query`.
- `src/generator.py`: RAG generation flow. It gets the retriever, builds the prompt, and calls `GENERATE_MODEL` through Ollama.
- `src/retriever.py`: Retrieval core. It builds the Chroma vector store, runs vector search, supports BM25 hybrid retrieval, and applies CrossEncoder reranking.
- `src/evaluation.py`: Retriever evaluation with BEIR qrels, including initial retrieval and reranked retrieval metrics.
- `src/config.py`: Central configuration for the dataset, embedding model, reranker model, Ollama URL, generation model, and retriever type.
- `docker-compose.yml`: Starts the Ollama container and mounts `./ollama` to persist model data.
- `Makefile`: Common development commands for installation, formatting, linting, CI checks, and starting the API server.

## Query Flow

1. The user calls `POST /query` with a `query` field in the request body.
2. `src/routes.py` calls `generate_response(query)`.
3. `src/generator.py` ensures `vectordatabase/` is initialized and gets the cached retriever.
4. The retriever fetches candidate chunks from Chroma.
5. The `CrossEncoder` reranker reorders the candidate chunks and keeps the top documents.
6. The generator combines the query and retrieved documents into a prompt.
7. Ollama uses `GENERATE_MODEL` to produce the final answer.

## Vector Database Initialization

`initialize_vector_database()` runs when `vectordatabase/` does not exist or is empty:

1. Download the BEIR dataset from `DATASET_URL`.
2. Read `corpus.jsonl`.
3. Combine each title and text field into a LangChain `Document`.
4. Split documents into chunks with `RecursiveCharacterTextSplitter`.
5. Generate embeddings with `EMBEDDING_MODEL`.
6. Write chunks to Chroma with cosine distance.

If the embedding model or embedding dimension changes, delete or rebuild `vectordatabase/`. Chroma collections require a consistent vector dimension.

## Retriever Types

Two retriever types are currently supported:

- `vector`: Uses Chroma vector search only.
- `hybrid`: Combines Chroma vector search with BM25 and merges ranked results through `EnsembleRetriever`.

Configure this with `RETRIEVER_TYPE` in `src/config.py`.

## Running the App

Install dependencies:

```bash
make install
```

Start Ollama:

```bash
docker compose up -d ollama
```

Start the FastAPI app:

```bash
make run
```

By default, the API server runs at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Query API:

```bash
curl -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"query":"Living Longer by Reducing Leucine Intake"}'
```

## Evaluation

Run retriever evaluation:

```bash
uv run python -m src.evaluation --retriever hybrid --split dev
```

Available retrievers:

- `hybrid`
- `vector`

Available splits:

- `train`
- `dev`
- `test`

Evaluation logs are written to `log/`.

## TODO

- Add frontend for the RAG web app.
- Add RAGAS evaluation.
- Add Matryoshka embedding support, including configurable truncate dimension and vector database rebuild guidance.
- Add tests for retriever initialization and query route behavior.
