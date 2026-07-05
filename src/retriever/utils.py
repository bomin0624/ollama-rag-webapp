import json
from pathlib import Path

from beir import util
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET_URL,
    DATASETS_DIR,
    EMBED_TRUNCATE_DIM,
    EMBEDDING_MODEL,
)


def build_bm25_documents(collection: dict) -> list[Document]:
    """Build BM25 documents with a stable metadata id on every document."""
    documents = collection["documents"]
    collection_ids = collection["ids"]
    metadatas = collection["metadatas"] or []

    bm25_documents = []
    for index, text in enumerate(documents):
        if not text:
            continue

        metadata = (
            dict(metadatas[index] or {}) if index < len(metadatas) else {}
        )
        if "id" not in metadata or metadata["id"] is None:
            metadata["id"] = collection_ids[index]

        bm25_documents.append(Document(page_content=text, metadata=metadata))

    return bm25_documents


def rerank_documents(
    query: str,
    retrieved_chunks: list[Document],
    reranker_model: CrossEncoder,
    top_n: int,
) -> list[Document]:
    """
    Rerank retrieved chunks and keep the top chunk for each unique document.
    """
    if not retrieved_chunks:
        return []
    pairs = [(query, chunk.page_content) for chunk in retrieved_chunks]
    scores = reranker_model.predict(pairs)
    # List of tuples [(score, Document), (score, Document), ...]
    scored_docs = sorted(
        zip(scores, retrieved_chunks, strict=False),
        key=lambda x: x[0],
        reverse=True,
    )

    unique_docs = []
    seen_ids = set()
    for _, doc in scored_docs:
        if doc.metadata["id"] not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc.metadata["id"])
        if len(unique_docs) >= top_n:
            break

    return unique_docs


def build_embedding(model_name: str) -> HuggingFaceEmbeddings:
    model_kwargs = {}
    if EMBED_TRUNCATE_DIM is not None:
        model_kwargs["truncate_dim"] = EMBED_TRUNCATE_DIM
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def _stored_embedding_dim(db_directory: str) -> int | None:
    """Return the persisted collection's vector dim, or None if empty."""
    store = Chroma(
        persist_directory=db_directory,
        embedding_function=build_embedding(EMBEDDING_MODEL),
    )
    got = store.get(limit=1, include=["embeddings"])
    embeddings = got.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    return len(embeddings[0])


def initialize_vector_database(db_directory: str) -> None:
    """Build the vector database if missing.

    If it already exists, verify the persisted vector dimension matches the
    configured ``EMBED_TRUNCATE_DIM``. A mismatch means the embedding model or
    truncate dimension changed since the database was built, which would fail
    with an opaque Chroma error at query time; fail fast with rebuild guidance
    instead.
    """
    db_path = Path(db_directory)
    if db_path.exists() and any(db_path.iterdir()):
        if EMBED_TRUNCATE_DIM is not None:
            stored_dim = _stored_embedding_dim(db_directory)
            if stored_dim is not None and stored_dim != EMBED_TRUNCATE_DIM:
                raise RuntimeError(
                    f"Vector database dimension ({stored_dim}) does not "
                    f"match the configured EMBED_TRUNCATE_DIM "
                    f"({EMBED_TRUNCATE_DIM}). The embedding model or truncate "
                    f"dimension changed since the database was built. "
                    f"Rebuild it with:\n"
                    f"    rm -rf {db_directory} && make run"
                )
        return

    print("Vector database not found. Creating new database...")
    data_path = util.download_and_unzip(DATASET_URL, str(DATASETS_DIR))
    # corpus, queries, qrels = GenericDataLoader(data_path).load("test")
    corpus_path = Path(data_path) / "corpus.jsonl"
    documents = []

    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            documents.append(
                Document(
                    page_content=data["title"] + "." + " " + data["text"],
                    metadata={
                        "title": data.get("title", ""),
                        "id": data["_id"],
                    },
                )
            )

    # max_length * 4 = chunk_size
    # chunk_overlap = chunk_size * 0.10 ~ 0.25
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")

    embedding = build_embedding(EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=db_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store created and persisted to {db_directory}")
