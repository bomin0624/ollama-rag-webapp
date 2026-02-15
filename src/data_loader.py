import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from config import url

data_path = util.download_and_unzip(url, os.path.join(os.path.dirname(__file__), "..", "datasets"))
corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

print(type(corpus), type(queries), type(qrels))

first_corpus_id = list(corpus.keys())[1]
print("--- Corpus Example ---")
print(f"ID: {first_corpus_id}")
print(f"Content: {corpus[first_corpus_id]}") # text and title


first_query_id = list(queries.keys())[1]
print("\n--- Query Example ---")
print(f"ID: {first_query_id}")
print(f"Text: {queries[first_query_id]}")


print("\n--- Qrel Example ---")
print(f"Query ID: {first_query_id}")
print(f"Relevant Documents: {qrels[first_query_id]}")
print(f"Number of relevant documents: {len(qrels[first_query_id])}")
