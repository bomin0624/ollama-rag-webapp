import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer

from config import embedding_model, url


def analyze_token_length(texts, tokenizer, title):
    logging.info(f"Tokenizing {len(texts)} texts for '{title}'...")
    token_length = [len(tokenizer.tokenize(text)) for text in texts]
    # plt.figure(figsize=(10, 6))
    # plt.hist(token_length, bins=50, color='blue', alpha=0.7)
    # plt.title('Token Length Distribution')
    # plt.ylabel('Frequency')
    # plt.xlabel('Token Length')
    # plt.grid(True)
 
    logging.info("---------------------------------")
    logging.info(f"Statistics for '{title}':")
    logging.info(f"  Min: {np.min(token_length)}")
    logging.info(f"  Max: {np.max(token_length)}")
    logging.info(f"  Mean: {np.mean(token_length):.2f}")
    logging.info(f"  Median: {np.median(token_length)}")
    logging.info("----------------------------------")

    # output_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    # os.makedirs(output_dir, exist_ok=True)
    
    # save_path = os.path.join(output_dir, f"{title}_token_length_distribution.png")
    # plt.savefig(save_path)
    # logging.info(f"Histogram saved to {save_path}")
    # plt.close()


def main():
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')
    log_file_path = os.path.join(log_dir, 'data_analyze_nfcorpus.log')
    
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'
        )
    
    data_path = util.download_and_unzip(url, os.path.join(os.path.dirname(__file__), "..", "datasets"))
    corpus, queries, _ = GenericDataLoader(data_path).load(split="dev")

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    texts = [content["text"] for content in corpus.values()]
    analyze_token_length(texts, tokenizer, "Corpus")

    query_texts = list(queries.values())
    analyze_token_length(query_texts, tokenizer, "Queries")


if __name__ == "__main__":
    main()