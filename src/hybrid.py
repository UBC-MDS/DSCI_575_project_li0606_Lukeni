from collections import defaultdict

from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = None):
        k = top_k or self.top_k

        bm25_docs = self.bm25_retriever.search(query, top_k=k)
        semantic_docs = self.semantic_retriever.search(query, top_k=k)

        scores = defaultdict(float)
        rows = {}

        for rank, (_, row) in enumerate(bm25_docs.iterrows(), start=1):
            doc_id = row["doc_id"]
            scores[doc_id] += 1.0 / (self.rrf_k + rank)
            rows[doc_id] = row

        for rank, (_, row) in enumerate(semantic_docs.iterrows(), start=1):
            doc_id = row["doc_id"]
            scores[doc_id] += 1.0 / (self.rrf_k + rank)
            rows[doc_id] = row

        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        result_rows = []
        for doc_id in ranked_ids:
            row = rows[doc_id].copy()
            row["hybrid_score"] = scores[doc_id]
            result_rows.append(row)

        import pandas as pd
        return pd.DataFrame(result_rows).reset_index(drop=True)