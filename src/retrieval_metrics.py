"""
Binary-relevance retrieval metrics for ranked doc_id lists.

Assumes each query has a set of relevant document IDs (from ground truth).
"""

from __future__ import annotations


def precision_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Fraction of the top-k positions that are relevant: |rel ∩ top_k| / k."""
    if k <= 0:
        return 0.0
    top = ranked_doc_ids[:k]
    hits = sum(1 for d in top if d in relevant_doc_ids)
    return hits / k


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Fraction of all relevant docs that appear in top-k: |rel ∩ top_k| / |rel|."""
    if not relevant_doc_ids:
        return 0.0
    top = set(ranked_doc_ids[:k])
    hits = len(relevant_doc_ids & top)
    return hits / len(relevant_doc_ids)


def reciprocal_rank(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document (1-indexed), or 0.0 if none."""
    for i, doc in enumerate(ranked_doc_ids, start=1):
        if doc in relevant_doc_ids:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """MRR over multiple queries; skips queries with empty relevant_sets."""
    rr_vals: list[float] = []
    for ranked, rel in zip(ranked_lists, relevant_sets):
        if not rel:
            continue
        rr_vals.append(reciprocal_rank(ranked, rel))
    if not rr_vals:
        return 0.0
    return sum(rr_vals) / len(rr_vals)


def mean_precision_at_k(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
    k: int,
) -> float:
    """Mean P@k over queries with non-empty relevant_sets."""
    vals: list[float] = []
    for ranked, rel in zip(ranked_lists, relevant_sets):
        if not rel:
            continue
        vals.append(precision_at_k(ranked, rel, k))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def mean_recall_at_k(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
    k: int,
) -> float:
    """Mean R@k over queries with non-empty relevant_sets."""
    vals: list[float] = []
    for ranked, rel in zip(ranked_lists, relevant_sets):
        if not rel:
            continue
        vals.append(recall_at_k(ranked, rel, k))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
