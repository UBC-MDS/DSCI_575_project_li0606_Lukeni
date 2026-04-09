"""
BM25 retriever implementation (Milestone 1).

This file is intentionally a scaffold in the repository initialization step.
You will implement BM25 retrieval logic here during Milestone 1.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils import load_corpus, tokenize_for_bm25


class BM25Retriever:
    def __init__(self, corpus_df: pd.DataFrame, text_col: str = "retrieval_text") -> None:
        self.corpus_df = corpus_df.reset_index(drop=True).copy()
        self.text_col = text_col
        self.tokenized_corpus = [
            tokenize_for_bm25(text) for text in self.corpus_df[self.text_col].fillna("")
        ]
        self.index = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_corpus_file(
        cls,
        corpus_path: str | Path,
        text_col: str = "retrieval_text",
    ) -> "BM25Retriever":
        corpus_df = load_corpus(corpus_path)
        return cls(corpus_df=corpus_df, text_col=text_col)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_tokens = tokenize_for_bm25(query)
        scores = np.array(self.index.get_scores(query_tokens))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.corpus_df.iloc[top_indices].copy()
        results["bm25_score"] = scores[top_indices]

        return results.sort_values("bm25_score", ascending=False).reset_index(drop=True)

    def save(self, index_path: str | Path, tokens_path: str | Path) -> None:
        index_path = Path(index_path)
        tokens_path = Path(tokens_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        tokens_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(tokens_path, "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

    @staticmethod
    def load_saved(
        corpus_path: str | Path,
        index_path: str | Path,
        tokens_path: str | Path,
        text_col: str = "retrieval_text",
    ) -> "BM25Retriever":
        corpus_df = load_corpus(corpus_path)

        retriever = BM25Retriever.__new__(BM25Retriever)
        retriever.corpus_df = corpus_df.reset_index(drop=True).copy()
        retriever.text_col = text_col

        with open(index_path, "rb") as f:
            retriever.index = pickle.load(f)

        with open(tokens_path, "rb") as f:
            retriever.tokenized_corpus = pickle.load(f)

        return retriever
