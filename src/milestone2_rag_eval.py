"""
Milestone 2 qualitative RAG eval: run hybrid RAG on fixed queries, write JSON for reports.

Output: ``results/milestone2_rag_eval_runs.json`` (by default).

Requires ``GROQ_API_KEY`` and the same notebook sample bundle as the Streamlit app.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed query set for ``results/milestone2_discussion.md`` (diverse intents; Video Games domain).
MILESTONE2_RAG_QUERIES: list[str] = [
    "best racing game with fun tracks",
    "story-rich scary game with dark atmosphere",
    "good wireless PS5 controller with long battery life",
    "relaxing cozy game for stress relief",
    "competitive online FPS with a large player base",
    "Steam Deck games that work well for travel",
    "family-friendly party multiplayer on Nintendo Switch",
    "Is Minecraft good for creative kids?",
    "soulslike or very hard action RPG recommendations",
    "best gaming headset under $50 with clear mic for Discord",
]


def _ensure_project_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def run_milestone2_rag_eval(
    *,
    out_path: Path | None = None,
    top_k: int = 5,
) -> Path:
    """
    Load hybrid RAG, run ``MILESTONE2_RAG_QUERIES``, write JSON with answers and retrieval rows.

    Each hit is a **review-level** row; ``product_title`` may repeat across different ``doc_id``s.
    """
    _ensure_project_path()
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("GROQ_API_KEY", "").strip():
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env (see .env.example) to run Milestone 2 RAG eval."
        )

    from src.retrieval import discover_bundle
    from src.rag_pipeline import HybridRAGPipeline, SYSTEM_PROMPT_V1

    out_path = out_path or (PROJECT_ROOT / "results" / "milestone2_rag_eval_runs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = discover_bundle()
    pipe = HybridRAGPipeline(
        corpus_path=str(bundle.corpus_path),
        bm25_index_path=str(bundle.bm25_index),
        bm25_tokens_path=str(bundle.bm25_tokens),
        faiss_index_path=str(bundle.faiss_index),
        metadata_path=str(bundle.semantic_meta),
        top_k=top_k,
        system_prompt=SYSTEM_PROMPT_V1,
    )

    results: list[dict] = []
    for i, q in enumerate(MILESTONE2_RAG_QUERIES, start=1):
        out = pipe.answer(q, system_prompt=SYSTEM_PROMPT_V1)
        docs = out["docs"]
        titles: list[str] = []
        doc_ids: list[str] = []
        if docs is not None and not docs.empty:
            take = docs.head(top_k)
            titles = take["product_title"].fillna("").astype(str).tolist()
            doc_ids = take["doc_id"].astype(str).tolist()

        results.append(
            {
                "id": i,
                "query": q,
                "answer": out["answer"],
                "retrieved_titles_top5": titles,
                "retrieved_doc_ids_top5": doc_ids,
                "unique_doc_ids_in_top5": len(set(doc_ids)),
            }
        )
        print(f"  milestone2_rag eval: query {i}/{len(MILESTONE2_RAG_QUERIES)}", file=sys.stderr)

    payload = {
        "bundle_label": bundle.label,
        "top_k": top_k,
        "system_prompt": "SYSTEM_PROMPT_V1",
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(results)} queries, top_k={top_k}).")
    return out_path


def main() -> None:
    try:
        run_milestone2_rag_eval()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
