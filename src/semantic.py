"""Semantic retriever (sentence-transformers + FAISS)."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import pandas as pd

from src.utils import load_corpus


class SemanticRetriever:
    def __init__(
        self,
        corpus_df: pd.DataFrame,
        text_col: str = "retrieval_text",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.corpus_df = corpus_df.reset_index(drop=True).copy()
        self.text_col = text_col
        self.model_name = model_name
        self.index = None

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    @classmethod
    def from_corpus_file(
        cls,
        corpus_path: str | Path,
        text_col: str = "retrieval_text",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "SemanticRetriever":
        corpus_df = load_corpus(corpus_path)
        return cls(corpus_df=corpus_df, text_col=text_col, model_name=model_name)

    def build_index(self, batch_size: int = 64) -> None:
        texts = self.corpus_df[self.text_col].fillna("").tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.index is None:
            raise ValueError("Semantic index has not been built yet. Call build_index() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        result_indices = indices[0]
        result_scores = scores[0]

        results = self.corpus_df.iloc[result_indices].copy()
        results["semantic_score"] = result_scores

        return results.sort_values("semantic_score", ascending=False).reset_index(drop=True)

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if self.index is None:
            raise ValueError("Semantic index has not been built yet. Call build_index() first.")

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        metadata = {
            "corpus_df": self.corpus_df,
            "text_col": self.text_col,
            "model_name": self.model_name,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load_saved(index_path: str | Path, metadata_path: str | Path) -> "SemanticRetriever":
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        retriever = SemanticRetriever(
            corpus_df=metadata["corpus_df"],
            text_col=metadata["text_col"],
            model_name=metadata["model_name"],
        )
        retriever.index = faiss.read_index(str(index_path))

        return retriever