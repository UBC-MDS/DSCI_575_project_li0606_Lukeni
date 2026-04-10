"""Streamlit retrieval UI — modern wide-layout search (BM25 / Semantic / Hybrid RRF)."""

from __future__ import annotations

import csv
import html
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import discover_bundle, hybrid_search, load_retrievers  # noqa: E402

TEXT_PREVIEW = 200
TOP_K = 10

MODE_INTERNAL = {"BM25": "bm25", "Semantic": "semantic", "Hybrid": "hybrid"}

_CARD_CSS = """
<style>
  .search-hero { text-align: center; margin-bottom: 0.25rem; }
  .search-hero h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.35rem; }
  .status-line { margin: 0.35rem 0 1rem 0; padding: 0.45rem 0.75rem; border-left: 3px solid #4a90d9;
    background: linear-gradient(90deg, rgba(74,144,217,0.08), transparent); color: #333; font-size: 0.95rem; }
  .rating-line { margin: 0.35rem 0 0.35rem 0; font-size: 0.95rem; }
  .review-body { color: #5a5a5a; font-size: 0.92rem; line-height: 1.45; margin: 0; }
  .bundle-hint { text-align: center; font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
</style>
"""


def _feedback_path() -> Path:
    base = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    return Path(os.getenv("FEEDBACK_LOG_PATH", str(base / "app_feedback.csv")))


def _truncate_ellipsis(s: object, n: int = TEXT_PREVIEW) -> str:
    t = str(s) if s is not None else ""
    t = t.replace("\n", " ").strip()
    if len(t) <= n:
        return t
    return t[:n] + "..."


def _rating_display(rating: object) -> str:
    try:
        x = float(rating)
        n = max(0, min(5, int(round(x))))
        stars = "⭐" * n + "☆" * (5 - n)
        return html.escape(f"{stars} ({x:g})")
    except (TypeError, ValueError):
        return "—"


def _append_feedback(
    *,
    query: str,
    mode: str,
    product_title: str,
    feedback: int,
    doc_id: str,
) -> None:
    path = _feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "query",
        "mode",
        "product_title",
        "feedback",
        "doc_id",
    ]
    new_file = not path.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "product_title": product_title[:500],
        "feedback": feedback,
        "doc_id": doc_id,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


@st.cache_resource(show_spinner="Loading retrieval indices…")
def _cached_retrievers():
    bundle = discover_bundle()
    bm25, semantic = load_retrievers(bundle)
    return bundle, bm25, semantic


def _show_toast(msg: str) -> None:
    try:
        st.toast(msg, icon="✓")
    except Exception:
        st.success(msg)


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="Video Games Search",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    try:
        bundle, bm25, semantic = _cached_retrievers()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Retrieval indices were not found. Place the notebook sample bundle under `data/processed/` "
            "or set `PROCESSED_DATA_DIR`. See the README section on retrieval artifacts."
        )
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown('<div class="search-hero"><h1>Video Games Search</h1></div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="bundle-hint">Index bundle: <strong>{html.escape(bundle.label)}</strong></p>',
            unsafe_allow_html=True,
        )

        with st.form("search_form", clear_on_submit=False):
            query = st.text_input(
                "Query",
                placeholder="Search reviews and products…",
                label_visibility="collapsed",
                key="query_input",
            )
            mode_label = st.radio(
                "Search mode",
                ["BM25", "Semantic", "Hybrid"],
                horizontal=True,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    if submitted and query.strip():
        q = query.strip()
        corpus = bm25.corpus_df
        internal = MODE_INTERNAL[mode_label]
        with st.spinner("Searching…"):
            if internal == "bm25":
                hits = bm25.search(q, top_k=TOP_K)
                score_col = "bm25_score"
            elif internal == "semantic":
                hits = semantic.search(q, top_k=TOP_K)
                score_col = "semantic_score"
            else:
                hits = hybrid_search(bm25, semantic, corpus, q, top_k=TOP_K)
                score_col = "hybrid_score"
        st.session_state["hits"] = hits
        st.session_state["score_col"] = score_col
        st.session_state["last_query"] = q
        st.session_state["last_mode_label"] = mode_label
    elif submitted and not query.strip():
        st.warning("Enter a query to search.")
        for k in ("hits", "score_col", "last_query", "last_mode_label"):
            st.session_state.pop(k, None)

    hits = st.session_state.get("hits")
    score_col = st.session_state.get("score_col", "bm25_score")
    last_q = st.session_state.get("last_query", "")
    last_mode = st.session_state.get("last_mode_label", "")

    if hits is not None and not hits.empty and last_q:
        st.markdown(
            f'<div class="status-line"><em>Showing top results for '
            f'<strong>{html.escape(last_q)}</strong> using '
            f'<strong>{html.escape(last_mode)}</strong> mode.</em></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Results")

        for rank, (_, row) in enumerate(hits.iterrows()):
            title = str(row.get("product_title", "") or "(no title)")
            text_raw = row.get("text", "")
            rating = row.get("rating", "")
            doc_id = str(row.get("doc_id", ""))
            score = row.get(score_col, float("nan"))
            try:
                score_s = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_s = str(score)

            review_e = html.escape(_truncate_ellipsis(text_raw))
            rating_html = _rating_display(rating)
            uid = f"{rank}_{doc_id}"[:80]

            with st.container(border=True):
                head_l, head_r = st.columns([5, 1])
                with head_l:
                    st.markdown(f"**{title}**")
                    st.caption(f"Score: {score_s}")
                with head_r:
                    u_col, d_col = st.columns(2)
                    with u_col:
                        if st.button("👍", key=f"up_{uid}", help="Upvote"):
                            _append_feedback(
                                query=last_q,
                                mode=last_mode,
                                product_title=title,
                                feedback=1,
                                doc_id=doc_id,
                            )
                            _show_toast("Feedback recorded!")
                    with d_col:
                        if st.button("👎", key=f"dn_{uid}", help="Downvote"):
                            _append_feedback(
                                query=last_q,
                                mode=last_mode,
                                product_title=title,
                                feedback=-1,
                                doc_id=doc_id,
                            )
                            _show_toast("Feedback recorded!")
                st.markdown(f'<div class="rating-line">{rating_html}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="review-body">{review_e}</p>', unsafe_allow_html=True)

    elif hits is not None and hits.empty:
        st.warning("No results for this query.")


if __name__ == "__main__":
    main()
