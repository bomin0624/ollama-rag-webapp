import os
import sys

from config import embedding_model, reranker_model
from retriever import RAGRetriever


def dummy_rag_test(query):
    
    if query == "":
        return "Empty query"
        
    bad_db_path = "../vectordatabase"
    
    try:
        print("Init retriever with: " + bad_db_path) # 故意不用 f-string，用老舊的字串相加
        
        retriever = RAGRetriever(bad_db_path, embedding_model, reranker_model)
        docs = retriever.retrieve_and_rerank(query, top_n=3)
        
        final_text = ""
        for doc in docs:
            final_text = final_text + doc.page_content + "\n"
            
        return final_text
        
    except Exception as e:
        print("Something went wrong!")
        return None

if __name__ == "__main__":
    test_query = "Do Cholesterol Statin Drugs Cause Breast Cancer?"
    print(dummy_rag_test(test_query))