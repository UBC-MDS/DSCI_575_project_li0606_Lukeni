"""Fetch `*_final` artifacts from a release URL when local files are missing (hosted Streamlit, etc.)."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

ARTIFACTS = (
    "video_games_corpus_final.csv",
    "bm25_final_index.pkl",
    "bm25_final_tokens.pkl",
    "faiss_final.index",
    "semantic_final_metadata.pkl",
)


def _fetch_enabled() -> bool:
    v = (os.getenv("FETCH_REMOTE_ARTIFACTS") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _base() -> str | None:
    u = (os.getenv("REMOTE_ARTIFACTS_BASE_URL") or "").strip()
    if not u:
        return None
    return u if u.endswith("/") else u + "/"


def _ok(processed: Path) -> bool:
    if not processed.is_dir():
        return False
    from src.retrieval import discover_bundle

    try:
        discover_bundle(processed)
        return True
    except FileNotFoundError:
        return False


def needs_remote_download(processed: Path) -> bool:
    return (not _ok(processed)) and _fetch_enabled() and bool(_base())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    r = urllib.request.Request(url, headers={"User-Agent": "DSCI575-App/1.0"})
    try:
        with urllib.request.urlopen(r, timeout=600) as resp, open(part, "wb") as f:
            while b := resp.read(1024 * 512):
                f.write(b)
        part.replace(dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def ensure_remote_artifacts(processed: Path) -> bool:
    processed = processed.resolve()
    processed.mkdir(parents=True, exist_ok=True)
    if _ok(processed):
        return True
    if not _fetch_enabled() or not _base():
        return _ok(processed)
    base = _base() or ""
    for name in ARTIFACTS:
        dest = processed / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        url = f"{base.rstrip('/')}/{name}"
        try:
            _download(url, dest)
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(f"HTTP {e.code} fetching {url}") from e
        except urllib.error.URLError as e:
            raise FileNotFoundError(f"Network error: {url} — {e}") from e
    if not _ok(processed):
        raise FileNotFoundError(
            f"Bundle still incomplete under {processed}. If you use only Parquet for the corpus, add it to the same URL prefix."
        )
    return True
