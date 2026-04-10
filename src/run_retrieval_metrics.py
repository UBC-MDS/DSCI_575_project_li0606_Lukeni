"""
Compute Precision@k, Recall@k, and MRR for BM25 and semantic retrievers
using labeled doc_ids in data/processed/ground_truth.csv.

Requires the same sample artifacts as ``make eval`` (aligned 1k corpus).

From the repository root: ``make metrics``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    processed = PROJECT_ROOT / "data" / "processed"
    gt_path = processed / "ground_truth.csv"
    faiss_path = processed / "faiss_sample.index"
    meta_path = processed / "semantic_sample_metadata.pkl"
    summary_path = processed / "retrieval_metrics_summary.csv"
    per_query_path = processed / "retrieval_metrics_per_query.csv"

    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing {gt_path}")
    if not faiss_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing semantic sample artifacts under {processed}."
        )

    from src.bm25 import BM25Retriever
    from src.retrieval_metrics import (
        mean_precision_at_k,
        mean_recall_at_k,
        mean_reciprocal_rank,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )
    from src.semantic import SemanticRetriever
    from src.utils import load_ground_truth, parse_relevant_doc_ids

    semantic = SemanticRetriever.load_saved(faiss_path, meta_path)
    corpus_df = semantic.corpus_df
    bm25 = BM25Retriever(corpus_df)

    gt = load_ground_truth(gt_path)
    top_k = 10

    per_rows: list[dict[str, object]] = []
    bm25_ranked: list[list[str]] = []
    sem_ranked: list[list[str]] = []
    rel_sets: list[set[str]] = []

    for _, r in gt.iterrows():
        qid = str(r["query_id"])
        query = str(r["query"])
        rel = parse_relevant_doc_ids(r.get("relevant_doc_ids"))
        rel_sets.append(rel)

        bm = bm25.search(query, top_k=top_k)
        se = semantic.search(query, top_k=top_k)
        bm_ids = bm["doc_id"].astype(str).tolist()
        se_ids = se["doc_id"].astype(str).tolist()
        bm25_ranked.append(bm_ids)
        sem_ranked.append(se_ids)

        if not rel:
            continue

        for method, ids in ("bm25", bm_ids), ("semantic", se_ids):
            per_rows.append(
                {
                    "query_id": qid,
                    "method": method,
                    "P@5": precision_at_k(ids, rel, 5),
                    "P@10": precision_at_k(ids, rel, 10),
                    "R@5": recall_at_k(ids, rel, 5),
                    "R@10": recall_at_k(ids, rel, 10),
                    "RR": reciprocal_rank(ids, rel),
                    "num_relevant_labeled": len(rel),
                }
            )

    labeled_bm25 = [b for b, rel in zip(bm25_ranked, rel_sets) if rel]
    labeled_sem = [s for s, rel in zip(sem_ranked, rel_sets) if rel]
    labeled_rel = [rel for rel in rel_sets if rel]

    summary_rows = [
        {
            "method": "bm25",
            "queries_with_labels": len(labeled_rel),
            "P@5": mean_precision_at_k(labeled_bm25, labeled_rel, 5),
            "P@10": mean_precision_at_k(labeled_bm25, labeled_rel, 10),
            "R@5": mean_recall_at_k(labeled_bm25, labeled_rel, 5),
            "R@10": mean_recall_at_k(labeled_bm25, labeled_rel, 10),
            "MRR": mean_reciprocal_rank(labeled_bm25, labeled_rel),
        },
        {
            "method": "semantic",
            "queries_with_labels": len(labeled_rel),
            "P@5": mean_precision_at_k(labeled_sem, labeled_rel, 5),
            "P@10": mean_precision_at_k(labeled_sem, labeled_rel, 10),
            "R@5": mean_recall_at_k(labeled_sem, labeled_rel, 5),
            "R@10": mean_recall_at_k(labeled_sem, labeled_rel, 10),
            "MRR": mean_reciprocal_rank(labeled_sem, labeled_rel),
        },
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(per_rows).to_csv(per_query_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {per_query_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
