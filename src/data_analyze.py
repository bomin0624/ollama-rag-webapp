import logging

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer

from src.config import DATASET_URL, DATASETS_DIR, EMBEDDING_MODEL, LOG_DIR


def analyze_token_length(texts, tokenizer, title):
    logging.info(f"Tokenizing {len(texts)} texts for '{title}'...")
    token_length = [len(tokenizer.tokenize(text)) for text in texts]

    logging.info("---------------------------------")
    logging.info(f"Statistics for '{title}':")
    logging.info(f"  Min: {np.min(token_length)}")
    logging.info(f"  Max: {np.max(token_length)}")
    logging.info(f"  Mean: {np.mean(token_length):.2f}")
    logging.info(f"  Median: {np.median(token_length)}")
    logging.info("----------------------------------")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = str(LOG_DIR / "data_analyze_nfcorpus.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file_path,
        filemode="w",
    )

    data_path = util.download_and_unzip(DATASET_URL, str(DATASETS_DIR))
    corpus, queries, _ = GenericDataLoader(data_path).load(split="dev")

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    texts = [content["text"] for content in corpus.values()]
    analyze_token_length(texts, tokenizer, "Corpus")

    query_texts = list(queries.values())
    analyze_token_length(query_texts, tokenizer, "Queries")


if __name__ == "__main__":
    main()
