"""
Offline evaluation: qualitative comparison CSV and/or retrieval metrics.

Uses the same **notebook sample** bundle as the Streamlit app (``src.retrieval.discover_bundle``).

Examples::

    make eval      # qualitative only
    make metrics   # metrics only
    PYTHONPATH=. python -m src.evaluation all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def run_qualitative(processed: Path | None = None, top_k: int = 10) -> Path:
    from src.retrieval import discover_bundle, load_retrievers
    from src.utils import format_topk_for_eval, load_ground_truth

    _ensure_project_path()
    processed = processed or Path("data/processed")
    gt_path = processed / "ground_truth.csv"
    out_path = processed / "qualitative_eval_runs.csv"

    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing {gt_path}")

    bundle = discover_bundle(processed)
    bm25, semantic = load_retrievers(bundle)
    gt = load_ground_truth(gt_path)

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
    print(f"Wrote {out_path} ({len(out_df)} queries, top_k={top_k}, bundle={bundle.label}).")
    return out_path


def run_metrics(processed: Path | None = None, top_k: int = 10) -> tuple[Path, Path]:
    from src.retrieval import discover_bundle, load_retrievers
    from src.retrieval_metrics import (
        mean_precision_at_k,
        mean_recall_at_k,
        mean_reciprocal_rank,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )
    from src.utils import load_ground_truth, parse_relevant_doc_ids

    _ensure_project_path()
    processed = processed or Path("data/processed")
    gt_path = processed / "ground_truth.csv"
    summary_path = processed / "retrieval_metrics_summary.csv"
    per_query_path = processed / "retrieval_metrics_per_query.csv"

    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing {gt_path}")

    bundle = discover_bundle(processed)
    bm25, semantic = load_retrievers(bundle)
    gt = load_ground_truth(gt_path)

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
    print(f"(bundle={bundle.label})")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return summary_path, per_query_path


def main() -> None:
    _ensure_project_path()
    p = argparse.ArgumentParser(description="Retrieval evaluation (qualitative + metrics).")
    p.add_argument(
        "command",
        choices=("qualitative", "metrics", "all"),
        help="qualitative: CSV of BM25 vs semantic runs; metrics: P@k/R@k/MRR; all: both",
    )
    p.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Override processed data directory (default: PROCESSED_DATA_DIR or data/processed)",
    )
    args = p.parse_args()
    processed = args.processed

    if args.command in ("qualitative", "all"):
        run_qualitative(processed)
    if args.command in ("metrics", "all"):
        run_metrics(processed)


if __name__ == "__main__":
    main()
