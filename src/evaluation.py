import argparse
import logging
import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

from config import embedding_model, reranker_model, url
from retriever import HybridRetriever, RAGRetriever, initialize_vector_database


def evaluate_retriever(retriever_type: str = "hybrid"):
    """
    Evaluate the retriever performance using Recall@30 for initial retrieval and NDCG@3 for reranked results.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "..", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"evaluation_nfcorpus_{retriever_type}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler()],
    )


    data_path = util.download_and_unzip(url, os.path.join(os.path.dirname(__file__), "..", "datasets"))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # print(queries.items()) # {'PLAIN-2': 'Do Cholesterol Statin Drugs Cause Breast Cancer?'}

    db_directory = os.path.join(os.path.dirname(__file__), "..", "vectordatabase")
    initialize_vector_database(db_directory)

    search_k = 100
    if retriever_type == "hybrid":
        ragretriever = HybridRetriever(
            db_directory=db_directory,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            search_k=search_k
        )
    elif retriever_type == "vector":
        ragretriever = RAGRetriever(
            db_directory=db_directory,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            search_k=search_k
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
        
    logging.info(f"--- Evaluating Initial Retrieval (Top {search_k}) ---")
    initial_results = {}
    for query_id, query_text in tqdm(queries.items(), desc="Initial Retriever"):
        initial_docs = ragretriever.retriever.invoke(query_text) # will return the list that already sorted by the retriever
        query_results ={}
        for rank, doc in enumerate(initial_docs):
            # print(f"\n=== Rank: {rank} ===")
            # print(f"Doc ID: {doc.metadata.get('id')}")
            # print(f"Doc Content preview: {doc.page_content[:100]}...")
            # print(f"Doc Metadata: {doc.metadata}")
            
            if doc.metadata["id"] not in query_results:
                query_results[doc.metadata["id"]] = 1.0 / (rank + 1) # for BEIR to calculate the metrics, we need to assign a score to each retrieved document.

        initial_results[query_id] = query_results

    # Evaluate initial retrieval using Recall@30
    evaluator = EvaluateRetrieval()
    k_values_initial = [1,5,10,30]
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, initial_results, k_values_initial)
    # print(evaluator.evaluate(qrels, initial_results, k_values_initial))  
    # ({'NDCG@1': 0.41796, 'NDCG@5': 0.3661, 'NDCG@10': 0.2739, 'NDCG@30': 0.20669}, 
    # {'MAP@1': 0.05611, 'MAP@5': 0.10866, 'MAP@10': 0.10866, 'MAP@30': 0.10866}, 
    # {'Recall@1': 0.05611, 'Recall@5': 0.12886, 'Recall@10': 0.12886, 'Recall@30': 0.12886}, 
    # {'P@1': 0.43963, 'P@5': 0.31641, 'P@10': 0.1582, 'P@30': 0.05273})  
    logging.info("Initial Retrieval Metrics:")
    logging.info(f"Recall@30: {recall['Recall@30']}") # Vector: 0.12886,  Hybrid: 0.12886


    reranked_results = {}
    for query_id, query_text in tqdm(queries.items(), desc="Reranking"):
        initial_docs = ragretriever.retriever.invoke(query_text) # will use the init paramaters of the retriever, which is search_k
         # print(initial_docs)
        if not initial_docs:
            reranked_results[query_id] = {}
            continue
        pairs = [(query_text, doc.page_content) for doc in initial_docs]
        scores = ragretriever.reranker.predict(pairs)
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)

        query_results = {}
        seen_ids = set()
        for score, doc in scored_docs:
            if doc.metadata["id"] not in seen_ids:
                query_results[doc.metadata["id"]] = float(score)
                seen_ids.add(doc.metadata["id"])
        reranked_results[query_id] = query_results

    # Evaluate reranked results using NDCG@3
    k_values_reranked = [1, 3, 5, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, reranked_results, k_values_reranked)
    logging.info("Reranked Retrieval Metrics:")
    logging.info(f"NDCG@3: {ndcg['NDCG@3']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Retriever Models")
    parser.add_argument("--retriever", type=str, choices=["vector", "hybrid"], default="hybrid", help="Choose the retriever to evaluate")
    args = parser.parse_args()
    evaluate_retriever(retriever_type=args.retriever)