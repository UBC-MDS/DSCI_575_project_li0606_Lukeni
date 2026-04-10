"""
Run BM25 and semantic retrieval for each query in ground_truth.csv.

Uses the same document subset as the saved semantic (FAISS) sample index so scores
are comparable. Requires artifacts from notebooks or build steps:

  - data/processed/faiss_sample.index
  - data/processed/semantic_sample_metadata.pkl
  - data/processed/ground_truth.csv

BM25 is rebuilt on semantic.corpus_df (same rows as the FAISS index).

From the repository root (preferred): ``make eval``

Output: ``data/processed/qualitative_eval_runs.csv``
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
    out_path = processed / "qualitative_eval_runs.csv"

    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing {gt_path}. Create or restore ground_truth.csv.")
    if not faiss_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing semantic sample artifacts under {processed}. "
            "Build faiss_sample.index and semantic_sample_metadata.pkl (see notebook)."
        )

    from src.bm25 import BM25Retriever
    from src.semantic import SemanticRetriever
    from src.utils import format_topk_for_eval, load_ground_truth

    semantic = SemanticRetriever.load_saved(faiss_path, meta_path)
    corpus_df = semantic.corpus_df
    bm25 = BM25Retriever(corpus_df)

    gt = load_ground_truth(gt_path)
    top_k = 10

    rows: list[dict[str, str | int]] = []
    for _, r in gt.iterrows():
        qid = str(r["query_id"])
        query = str(r["query"])
        difficulty = str(r["difficulty"]) if pd.notna(r.get("difficulty")) else ""
        bm = bm25.search(query, top_k=top_k)
        se = semantic.search(query, top_k=top_k)

        bm25_summary = format_topk_for_eval(
            bm, k=5, score_col="bm25_score" if "bm25_score" in bm.columns else None
        )
        sem_summary = format_topk_for_eval(
            se, k=5, score_col="semantic_score" if "semantic_score" in se.columns else None
        )

        rows.append(
            {
                "query_id": qid,
                "difficulty": difficulty,
                "query": query,
                "bm25_top5_summary": bm25_summary,
                "semantic_top5_summary": sem_summary,
                "bm25_top10_doc_ids": "|".join(bm["doc_id"].astype(str).tolist()),
                "semantic_top10_doc_ids": "|".join(se["doc_id"].astype(str).tolist()),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} queries, top_k={top_k}).")


if __name__ == "__main__":
    main()
