"""Streamlit UI — Search (BM25 / Semantic / Hybrid) and RAG (semantic or hybrid + LLM)."""

from __future__ import annotations

import os

# Before importing `src` (sentence-transformers / tokenizers): Streamlit reruns can fork; HF warns
# unless parallelism is disabled. See https://github.com/huggingface/tokenizers/issues/363
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import html
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Upstream huggingface_hub deprecation (pulled in by transformers / sentence-transformers).
warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import discover_bundle, hybrid_search, load_retrievers  # noqa: E402

TEXT_PREVIEW = 200
ANSWER_MAX_CHARS = 8000
CUSTOM_SYSTEM_PROMPT_MAX_CHARS = 6000
TOP_K = 10
RAG_TOP_K = 5

MODE_INTERNAL = {"BM25": "bm25", "Semantic": "semantic", "Hybrid": "hybrid"}
RAG_MODE_INTERNAL = {"Semantic RAG": "semantic", "Hybrid RAG": "hybrid"}

# Fallback if `src.rag_pipeline` is missing SYSTEM_PROMPT_V3 (e.g. unsynced branch); keep in sync with rag_pipeline.py.
_SYSTEM_PROMPT_V3_FALLBACK = """
You are an Amazon reviews analyst.
Answer the user only from the retrieved reviews and product metadata.
Prefer direct evidence from the retrieved context, and avoid unsupported claims.
"""


def _prompt_map():
    """Preset strings from ``src.rag_pipeline``; V3 uses getattr so older modules still load."""
    import importlib

    rp = importlib.import_module("src.rag_pipeline")
    v1 = rp.SYSTEM_PROMPT_V1
    v2 = rp.SYSTEM_PROMPT_V2
    v3 = getattr(rp, "SYSTEM_PROMPT_V3", None) or _SYSTEM_PROMPT_V3_FALLBACK
    return {"V1": v1, "V2": v2, "V3": v3}

_CARD_CSS = """
<style>
  .search-hero { text-align: center; margin-bottom: 0.25rem; }
  .search-hero h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.35rem; }
  .status-line { margin: 0.35rem 0 1rem 0; padding: 0.45rem 0.75rem; border-left: 3px solid #4a90d9;
    background: linear-gradient(90deg, rgba(74,144,217,0.08), transparent); color: #333; font-size: 0.95rem; }
  .rating-line { margin: 0.35rem 0 0.35rem 0; font-size: 0.95rem; }
  .review-body { color: #5a5a5a; font-size: 0.92rem; line-height: 1.45; margin: 0; }
  .bundle-hint { text-align: center; font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
  .rag-answer { margin: 0.75rem 0 1rem 0; padding: 0.75rem 1rem; border-radius: 8px;
    background: #f8fafc; border: 1px solid #e2e8f0; line-height: 1.55; white-space: pre-wrap; }
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


def _truncate_answer(s: object, n: int = ANSWER_MAX_CHARS) -> tuple[str, bool]:
    t = str(s) if s is not None else ""
    if len(t) <= n:
        return t, False
    return t[:n] + "\n\n…(truncated)", True


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


@st.cache_resource(show_spinner="Loading RAG pipelines…")
def _cached_rag_pipelines(
    corpus_path: str,
    bm25_index: str,
    bm25_tokens: str,
    faiss_index: str,
    semantic_meta: str,
):
    from src.rag_pipeline import HybridRAGPipeline, SemanticRAGPipeline

    semantic = SemanticRAGPipeline(
        corpus_path=corpus_path,
        faiss_index_path=faiss_index,
        metadata_path=semantic_meta,
        top_k=RAG_TOP_K,
    )
    hybrid = HybridRAGPipeline(
        corpus_path=corpus_path,
        bm25_index_path=bm25_index,
        bm25_tokens_path=bm25_tokens,
        faiss_index_path=faiss_index,
        metadata_path=semantic_meta,
        top_k=RAG_TOP_K,
    )
    return semantic, hybrid


def _show_toast(msg: str) -> None:
    try:
        st.toast(msg, icon="✓")
    except Exception:
        st.success(msg)


def _score_column_for_row(row, rag_internal: str) -> tuple[str, str]:
    if rag_internal == "hybrid" and "hybrid_score" in row.index:
        return "hybrid_score", "Hybrid (RRF)"
    if "semantic_score" in row.index:
        return "semantic_score", "Semantic"
    if "bm25_score" in row.index:
        return "bm25_score", "BM25"
    return "", ""


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="Video Games Search & RAG",
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

    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown('<div class="search-hero"><h1>Video Games Search & RAG</h1></div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="bundle-hint">Index bundle: <strong>{html.escape(bundle.label)}</strong></p>',
            unsafe_allow_html=True,
        )

        tab_search, tab_rag = st.tabs(["Search", "RAG"])

        with tab_search:
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

        with tab_rag:
            if not groq_key:
                st.error(
                    "RAG mode requires **GROQ_API_KEY** in your environment. "
                    "Copy `.env.example` to `.env` or export the variable. See README (Milestone 2 LLM setup)."
                )
            else:
                try:
                    semantic_pipe, hybrid_pipe = _cached_rag_pipelines(
                        str(bundle.corpus_path),
                        str(bundle.bm25_index),
                        str(bundle.bm25_tokens),
                        str(bundle.faiss_index),
                        str(bundle.semantic_meta),
                    )
                except Exception as e:
                    st.error("Could not load RAG pipelines.")
                    st.exception(e)
                    semantic_pipe, hybrid_pipe = None, None

                if semantic_pipe is not None and hybrid_pipe is not None:
                    prompts = _prompt_map()

                    st.caption("Retrieval + Groq LLM. Answers use the top-5 retrieved reviews as context.")

                    st.markdown(
                        """
**How to ask (your query in the box):** Write in plain English, as you would to another shopper—name a **genre**, **platform**, or **product need** (e.g. *“relaxing story-driven games”*, *“wireless controller with good battery”*, *“Is this headset good for footsteps in FPS?”*).  
Specific questions and short comparisons work best. The assistant only uses **retrieved review text** from this project’s sample index; if the corpus has little on your topic, answers may be vague or say the context is insufficient—that is expected.

**System prompt:** Use a **preset** (V1–V3, same strings as in `src/rag_pipeline.py`) or choose **Custom** to write your own instructions. The app always appends the same *Context* and *Question* blocks after your system text. Custom prompts are capped for safety (see form).
                        """
                    )
                    with st.expander("View full system prompt texts (V1, V2, V3)"):
                        t1, t2, t3 = st.tabs(["V1", "V2", "V3"])
                        with t1:
                            st.code(prompts["V1"].strip(), language=None)
                        with t2:
                            st.code(prompts["V2"].strip(), language=None)
                        with t3:
                            st.code(prompts["V3"].strip(), language=None)

                    with st.form("rag_form", clear_on_submit=False):
                        rag_query = st.text_input(
                            "RAG query",
                            placeholder="Ask a question about products and reviews…",
                            label_visibility="collapsed",
                            key="rag_query_input",
                        )
                        rag_mode_label = st.radio(
                            "RAG retrieval",
                            ["Semantic RAG", "Hybrid RAG"],
                            horizontal=True,
                            label_visibility="collapsed",
                        )
                        prompt_source = st.radio(
                            "System prompt",
                            ["Preset", "Custom"],
                            horizontal=True,
                            help="Preset: V1–V3 from code. Custom: your own instructions (still grounded on retrieved reviews).",
                        )
                        if prompt_source == "Preset":
                            preset_variant = st.radio(
                                "Preset variant",
                                ["V1", "V2", "V3"],
                                index=0,
                                horizontal=True,
                                help="V1: shopping assistant. V2: concise, admit if context is thin. V3: analyst, evidence-first.",
                            )
                            custom_system_text = ""
                        else:
                            preset_variant = "V1"
                            custom_system_text = st.text_area(
                                "Custom system prompt",
                                height=220,
                                placeholder=(
                                    "Describe how the assistant should behave. The app still appends "
                                    "retrieved review context and your question after this text.\n\n"
                                    "Example: You answer only from the provided reviews. If the context "
                                    "does not support an answer, say so briefly."
                                ),
                                help=(
                                    f"Maximum {CUSTOM_SYSTEM_PROMPT_MAX_CHARS} characters. "
                                    "You are responsible for asking the model to stay grounded in context."
                                ),
                                key="rag_custom_system_prompt",
                            )
                        rag_submitted = st.form_submit_button(
                            "Generate answer", type="primary", use_container_width=True
                        )

                    if rag_submitted and rag_query.strip():
                        q = rag_query.strip()
                        internal_rag = RAG_MODE_INTERNAL[rag_mode_label]
                        pipe = semantic_pipe if internal_rag == "semantic" else hybrid_pipe

                        sp: str | None = None
                        disp_prompt: str
                        if prompt_source == "Preset":
                            sp = prompts[preset_variant]
                            disp_prompt = preset_variant
                        else:
                            ct = (custom_system_text or "").strip()
                            if not ct:
                                st.warning("Enter a custom system prompt, or switch to Preset.")
                                st.session_state.pop("rag_result", None)
                                disp_prompt = ""
                            elif len(ct) > CUSTOM_SYSTEM_PROMPT_MAX_CHARS:
                                st.error(
                                    f"Custom system prompt is too long ({len(ct)} chars). "
                                    f"Maximum is {CUSTOM_SYSTEM_PROMPT_MAX_CHARS}."
                                )
                                st.session_state.pop("rag_result", None)
                                disp_prompt = ""
                            else:
                                sp = ct
                                disp_prompt = "Custom"

                        if sp is not None:
                            with st.spinner("Generating answer…"):
                                try:
                                    out = pipe.answer(q, system_prompt=sp)
                                    st.session_state["rag_result"] = out
                                    st.session_state["rag_last_query"] = q
                                    st.session_state["rag_mode_label"] = rag_mode_label
                                    st.session_state["rag_prompt_key"] = disp_prompt
                                except Exception as e:
                                    st.session_state.pop("rag_result", None)
                                    st.error(
                                        "The LLM request failed. Check your API key, network, and rate limits."
                                    )
                                    st.exception(e)
                    elif rag_submitted and not rag_query.strip():
                        st.warning("Enter a question to run RAG.")
                        st.session_state.pop("rag_result", None)

                    rag_out = st.session_state.get("rag_result")
                    rag_q = st.session_state.get("rag_last_query", "")
                    rag_mode_disp = st.session_state.get("rag_mode_label", "")
                    rag_prompt_disp = st.session_state.get("rag_prompt_key", "V1")

                    if rag_out is not None and rag_q:
                        st.markdown(
                            f'<div class="status-line"><em>RAG answer for '
                            f'<strong>{html.escape(rag_q)}</strong> — '
                            f'<strong>{html.escape(rag_mode_disp)}</strong> — '
                            f'<strong>{html.escape(rag_prompt_disp)}</strong></em></div>',
                            unsafe_allow_html=True,
                        )

                        ans = rag_out.get("answer", "")
                        ans_show, truncated = _truncate_answer(ans, ANSWER_MAX_CHARS)
                        st.markdown("### Answer")
                        st.markdown(
                            f'<div class="rag-answer">{html.escape(ans_show)}</div>',
                            unsafe_allow_html=True,
                        )
                        if truncated:
                            st.caption(f"Answer truncated to {ANSWER_MAX_CHARS} characters.")

                        docs = rag_out.get("docs")
                        if docs is not None and not docs.empty:
                            st.markdown("### Sources (retrieved reviews)")
                            r_internal = RAG_MODE_INTERNAL.get(rag_mode_disp, "semantic")
                            for i, (_, row) in enumerate(docs.iterrows(), start=1):
                                title = str(row.get("product_title", "") or "(no title)")
                                text_raw = row.get("text", "")
                                rating = row.get("rating", "")
                                doc_id = str(row.get("doc_id", ""))
                                sc_name, sc_label = _score_column_for_row(row, r_internal)
                                score = row.get(sc_name, float("nan")) if sc_name else float("nan")
                                try:
                                    score_s = f"{float(score):.4f}" if sc_name else "—"
                                except (TypeError, ValueError):
                                    score_s = str(score)
                                review_e = html.escape(_truncate_ellipsis(text_raw))
                                rating_html = _rating_display(rating)

                                with st.container(border=True):
                                    st.markdown(f"**[{i}]** {title}")
                                    if sc_label:
                                        st.caption(f"{sc_label} score: {score_s}")
                                    st.markdown(f'<div class="rating-line">{rating_html}</div>', unsafe_allow_html=True)
                                    st.markdown(f'<p class="review-body">{review_e}</p>', unsafe_allow_html=True)
                                    st.caption(f"`doc_id`: `{html.escape(doc_id)}`")


if __name__ == "__main__":
    main()
