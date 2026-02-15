import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from config import embedding_model, reranker_model, url


def rerank_documents(
    query: str, documents: list[Document], reranker_model:CrossEncoder, top_n: int
) -> list[Document]:
    """Using CrossEncoder to rerank the retrieved documents."""
    if not documents:
        return []
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker_model.predict(pairs)
    # List of tuples [(score, Document), (score, Document), ...]
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    unique_docs = []
    seen_ids = set()
    for _, doc in scored_docs:
        if doc.metadata["id"] not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc.metadata["id"])
        if len(unique_docs) >= top_n:
            break

    return unique_docs


class RAGRetriever:

    def __init__(self, db_directory: str, embedding_model: str, reranker_model: str, search_k: int = 30):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = Chroma(
            persist_directory=db_directory,
            embedding_function=self.embedding,
        )
        # https://zenn.dev/pipon_tech_blog/articles/8cdb27830236c5
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": search_k})
        print(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
    
    def retrieve_and_rerank(self, query: str, top_n: int = 3) -> list[Document]:
        initial_docs = self.retriever.invoke(query)
        reranked_docs = rerank_documents(query, initial_docs, self.reranker, top_n=top_n)
        return reranked_docs


def initialize_vector_database(db_directory: str):
    """Initialize the vector database if it does not exist."""
    if not os.path.exists(db_directory) or not os.listdir(db_directory):
        print("Vector database not found. Creating new database...")
        data_path = util.download_and_unzip(url, os.path.join(os.path.dirname(__file__), "..", "datasets"))
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        documents = []

        for doc_id, content in corpus.items():
            documents.append(Document(page_content=content["text"], 
                                    metadata={"title": content["title"], 
                                                "id": doc_id}))

        # max_length * 4 = chunk_size
        # chunk_overlap = chunk_size * 0.10 ~ 0.25
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=300,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Number of chunks: {len(chunks)}")
    

        embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=db_directory,
        )
        print(f"Vector store created and persisted to {db_directory}")



if __name__ == "__main__":

    db_directory = os.path.join(os.path.dirname(__file__), "..", "vectordatabase")
    initialize_vector_database(db_directory)
    # if not os.path.exists(db_directory) or not os.listdir(db_directory):
    #     print("Vector database not found. Creating new database...")
    #     data_path = util.download_and_unzip(url, os.path.join(os.path.dirname(__file__), "..", "datasets"))
    #     corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    #     documents = []

    #     for doc_id, content in corpus.items():
    #         documents.append(Document(page_content=content["text"], 
    #                                 metadata={"title": content["title"], 
    #                                             "id": doc_id}))

    #     # max_length * 4 = chunk_size
    #     # chunk_overlap = chunk_size * 0.10 ~ 0.25
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=2048,
    #         chunk_overlap=300,
    #     )

    #     chunks = text_splitter.split_documents(documents)
    #     print(f"Number of chunks: {len(chunks)}")
    

    #     embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    #     vector_store = Chroma.from_documents(
    #         documents=chunks,
    #         embedding=embedding,
    #         persist_directory=db_directory,
    #     )
    #     print(f"Vector store created and persisted to {db_directory}")
    # else:
    #     print(f"Using existing vector database at {db_directory}\n")

    # retriever = RAGRetriever(
    #     db_directory=db_directory,
    #     embedding_model=embedding_model,
    #     reranker_model=reranker_model,
    # )

    # query = "Living Longer by Reducing Leucine Intake"

    # retrieved_docs = retriever.retrieve_and_rerank(query)
    # print(f"Searching for query: {query}")
    # print("-----------------------------------------")
    # print(f"Found {len(retrieved_docs)} documents:")

    # for idx, doc in enumerate(retrieved_docs):
    #     print(f"\n--- Document {idx + 1} ---")
    #     print(f"Content: {doc.page_content[:250]}...")
    #     print(f"Metadata: {doc.metadata}")