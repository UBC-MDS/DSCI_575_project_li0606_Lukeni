"""Unit tests for retrieval metric definitions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval_metrics import (  # noqa: E402
    mean_precision_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert precision_at_k(ranked, rel, 4) == 0.5


def test_recall_at_k():
    ranked = ["x", "b", "c"]
    rel = {"a", "b", "c"}
    assert recall_at_k(ranked, rel, 3) == 2 / 3


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mean_aggregates():
    r1 = ["a", "b"]
    r2 = ["b", "a"]
    rel1 = {"a"}
    rel2 = {"b"}
    assert mean_precision_at_k([r1, r2], [rel1, rel2], 1) == 1.0
    assert mean_reciprocal_rank([r1, r2], [rel1, rel2]) == 1.0


if __name__ == "__main__":
    test_precision_at_k()
    test_recall_at_k()
    test_reciprocal_rank()
    test_mean_aggregates()
    print("ok")
