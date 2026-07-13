from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import RERANK_RETURN_N
from src.retriever.utils import (
    build_bm25_documents,
    build_embedding_model,
    rerank_documents,
)


class RAGRetriever:
    def __init__(
        self,
        db_directory: str,
        embedding_model: str,
        reranker_model: str,
        search_k: int,
    ):
        self.embedding = build_embedding_model(embedding_model)
        self.vector_store = Chroma(
            persist_directory=db_directory,
            embedding_function=self.embedding,
        )
        # https://zenn.dev/pipon_tech_blog/articles/8cdb27830236c5
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": search_k}
        )
        print(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)

    def retrieve_and_rerank(
        self, query: str, top_n: int = RERANK_RETURN_N
    ) -> list[Document]:
        initial_docs = self.retriever.invoke(query)
        reranked_docs = rerank_documents(
            query, initial_docs, self.reranker, top_n=top_n
        )
        return reranked_docs


class HybridRetriever(RAGRetriever):
    """
    A retriever that combines both vector search and sparse search (BM25).
    """

    def __init__(
        self,
        db_directory: str,
        embedding_model: str,
        reranker_model: str,
        search_k: int,
    ):
        super().__init__(
            db_directory, embedding_model, reranker_model, search_k
        )

        # Get the raw collection data to use for BM25.
        # TODO: This loads the entire collection into memory.
        # Consider Elasticsearch or Weaviate.
        collection = self.vector_store.get()
        documents = build_bm25_documents(collection)

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = search_k

        self.retriever = EnsembleRetriever(
            retrievers=[self.retriever, bm25_retriever], weights=[0.7, 0.3]
        )  # Reciprocal Rank Fusion (RRF) Algorithm
