"""Utility functions for building the retrieval corpus and query helpers."""
from __future__ import annotations

import ast
import gzip
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """
    Load a .jsonl or .jsonl.gz file into a pandas DataFrame.
    """
    path = Path(path)
    records = []

    open_func = gzip.open if path.suffix == ".gz" else open

    with open_func(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return pd.DataFrame(records)


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)

    if isinstance(value, str):
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return " ".join(str(v) for v in parsed if v is not None)
            except (ValueError, SyntaxError):
                pass

        return value

    return str(value)


def clean_text(text: Any) -> str:
    text = _safe_text(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_bm25(text: Any) -> list[str]:
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def pick_join_key(reviews_df: pd.DataFrame, meta_df: pd.DataFrame) -> str:
    for key in ["parent_asin", "asin"]:
        if key in reviews_df.columns and key in meta_df.columns:
            return key
    raise KeyError("No shared join key found. Expected 'parent_asin' or 'asin'.")


def build_retrieval_text(row: pd.Series) -> str:
    parts = [
        row.get("product_title", ""),
        row.get("categories", ""),
        row.get("features", ""),
        row.get("description", ""),
        row.get("review_title", ""),
        row.get("text", ""),
    ]
    return clean_text(" ".join(_safe_text(x) for x in parts if x is not None))


def build_corpus(
    reviews_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    min_review_chars: int = 20,
) -> pd.DataFrame:
    join_key = pick_join_key(reviews_df, meta_df)

    reviews = reviews_df.copy()
    meta = meta_df.copy()

    if "title" in reviews.columns:
        reviews = reviews.rename(columns={"title": "review_title"})
    if "title" in meta.columns:
        meta = meta.rename(columns={"title": "product_title"})

    meta_cols = [
        col for col in [join_key, "product_title", "description", "features", "categories", "price"]
        if col in meta.columns
    ]
    meta = meta[meta_cols].drop_duplicates(subset=[join_key])

    merged = reviews.merge(meta, on=join_key, how="left")

    if "text" not in merged.columns:
        raise KeyError("Reviews data must contain a 'text' column.")

    merged["text"] = merged["text"].fillna("")
    merged = merged[merged["text"].str.len() >= min_review_chars].copy()

    if "review_title" not in merged.columns:
        merged["review_title"] = ""
    if "product_title" not in merged.columns:
        merged["product_title"] = ""

    merged["retrieval_text"] = merged.apply(build_retrieval_text, axis=1)
    merged = merged[merged["retrieval_text"].str.len() > 0].copy()

    merged = merged.reset_index(drop=True)
    merged["doc_id"] = [f"doc_{i}" for i in range(len(merged))]

    preferred_cols = [
        "doc_id",
        join_key,
        "product_title",
        "review_title",
        "text",
        "rating",
        "price",
        "retrieval_text",
        "description",
        "features",
        "categories",
    ]
    keep_cols = [col for col in preferred_cols if col in merged.columns]

    return merged[keep_cols].copy()


def save_corpus(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
            return
        except Exception:
            fallback = output_path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return

    df.to_csv(output_path, index=False)


def load_corpus(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """
    Load query set. Expected columns at minimum: query_id, query.
    Optional: difficulty, relevant_doc_ids (pipe-separated doc_id values).
    """
    path = Path(path)
    df = pd.read_csv(path)
    required = {"query_id", "query"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground_truth CSV missing required columns: {sorted(missing)}")
    return df


def format_hit_line(
    row: pd.Series,
    title_col: str = "product_title",
    text_col: str = "text",
    rating_col: str = "rating",
    max_title: int = 72,
    max_text: int = 120,
) -> str:
    """One-line summary of a single retrieval hit for qualitative notes."""
    title = str(row.get(title_col, "") or "")[:max_title]
    text = str(row.get(text_col, "") or "").replace("\n", " ")[:max_text]
    rating = row.get(rating_col, "")
    return f"{title} | r={rating} | {text}"


def format_topk_for_eval(
    hits: pd.DataFrame,
    k: int = 5,
    title_col: str = "product_title",
    text_col: str = "text",
    score_col: str | None = None,
) -> str:
    """Compact multi-hit string for CSV / discussion tables (semicolon-separated rows)."""
    lines = []
    take = hits.head(k)
    for _, row in take.iterrows():
        part = format_hit_line(row, title_col=title_col, text_col=text_col)
        if score_col and score_col in row.index:
            part = f"{part} | score={row[score_col]:.4f}" if pd.notna(row[score_col]) else part
        lines.append(part)
    return " ;; ".join(lines)
