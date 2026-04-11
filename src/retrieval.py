"""
Index discovery, BM25 + FAISS loading, and hybrid (RRF) search.

``discover_bundle()`` loads only the **notebook sample** artifact set (see README). The Streamlit app
and ``python -m src.evaluation`` use that bundle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


@dataclass
class RetrievalBundle:
    label: str
    corpus_path: Path
    bm25_index: Path
    bm25_tokens: Path
    faiss_index: Path
    semantic_meta: Path


def _processed_dir() -> Path:
    return Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))


def _first_existing(processed: Path, names: list[str]) -> Path | None:
    for name in names:
        p = processed / name
        if p.is_file():
            return p
    return None


def discover_bundle(processed: Path | None = None) -> RetrievalBundle:
    """
    Load the **notebook sample bundle** only — filenames match
    ``notebooks/milestone1_exploration.ipynb`` (representative sample corpus, BM25, semantic).

    A separate full-corpus pipeline exists in ``src/build_retrievers.py`` for local experiments;
    it is **not** loaded here, so remote deploy only needs the notebook artifacts.
    """
    processed = processed or _processed_dir()
    if not processed.is_dir():
        raise FileNotFoundError(
            f"Processed data directory does not exist: {processed}. "
            "Set PROCESSED_DATA_DIR in .env or create the notebook sample bundle first."
        )

    corpus = _first_existing(
        processed,
        [
            "video_games_corpus_sample.parquet",
            "video_games_corpus_sample.csv",
        ],
    )
    bundle = RetrievalBundle(
        label="sample",
        corpus_path=corpus if corpus is not None else processed / "__missing__.csv",
        bm25_index=processed / "bm25_sample_index.pkl",
        bm25_tokens=processed / "bm25_sample_tokens.pkl",
        faiss_index=processed / "faiss_sample.index",
        semantic_meta=processed / "semantic_sample_metadata.pkl",
    )

    required: list[tuple[str, Path | None]] = [
        ("corpus (video_games_corpus_sample.parquet or .csv)", corpus),
        ("bm25_sample_index.pkl", bundle.bm25_index),
        ("bm25_sample_tokens.pkl", bundle.bm25_tokens),
        ("faiss_sample.index", bundle.faiss_index),
        ("semantic_sample_metadata.pkl", bundle.semantic_meta),
    ]
    missing = [f"{name} -> {p}" for name, p in required if p is None or not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            "No complete notebook sample bundle under "
            f"{processed.resolve()}. "
            "Run `notebooks/milestone1_exploration.ipynb` through the sample corpus, BM25, and semantic "
            "sections (or obtain these files from a teammate) and ensure PROCESSED_DATA_DIR points at "
            "the folder that contains them. "
            f"Missing: {'; '.join(missing)}"
        )

    assert corpus is not None
    bundle.corpus_path = corpus
    return bundle


def load_retrievers(
    bundle: RetrievalBundle,
) -> tuple[BM25Retriever, SemanticRetriever]:
    """
    Load FAISS + metadata first, then BM25. If saved BM25 rows do not match the
    semantic corpus (common with notebook workflows), rebuild BM25 from ``semantic.corpus_df``.
    """
    semantic = SemanticRetriever.load_saved(bundle.faiss_index, bundle.semantic_meta)
    try:
        bm25 = BM25Retriever.load_saved(
            bundle.corpus_path,
            bundle.bm25_index,
            bundle.bm25_tokens,
        )
        if len(bm25.corpus_df) != len(semantic.corpus_df):
            raise ValueError("BM25 / semantic corpus size mismatch")
    except Exception:
        bm25 = BM25Retriever(semantic.corpus_df)
    return bm25, semantic


def reciprocal_rank_fusion(
    ranked_doc_ids_a: list[str],
    ranked_doc_ids_b: list[str],
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(ranked_doc_ids_a, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    for rank, doc_id in enumerate(ranked_doc_ids_b, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    ordered = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [(d, scores[d]) for d in ordered]


def hybrid_search(
    bm25: BM25Retriever,
    semantic: SemanticRetriever,
    corpus_df: pd.DataFrame,
    query: str,
    top_k: int = 10,
    depth: int = 30,
) -> pd.DataFrame:
    b = bm25.search(query, top_k=depth)
    s = semantic.search(query, top_k=depth)
    fused = reciprocal_rank_fusion(
        b["doc_id"].astype(str).tolist(),
        s["doc_id"].astype(str).tolist(),
    )[:top_k]

    rows: list[dict] = []
    for doc_id, hscore in fused:
        m = corpus_df["doc_id"].astype(str) == doc_id
        if not m.any():
            continue
        row = corpus_df.loc[m].iloc[0].to_dict()
        row["hybrid_score"] = hscore
        rows.append(row)
    return pd.DataFrame(rows)
