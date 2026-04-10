# Qualitative retrieval discussion

## Evaluation setup

- **Corpus scope:** BM25 and semantic search were run on the **same 1,000-document subset** as the saved sample FAISS index (`faiss_sample.index` + `semantic_sample_metadata.pkl` from the exploration notebook). BM25 is rebuilt on `semantic.corpus_df` so rankings are comparable.
- **Query set:** `data/processed/ground_truth.csv` — **24 queries** with `difficulty` in {easy, medium, complex}. Column `relevant_doc_ids` is left empty for now (optional labels for quantitative metrics later).
- **Top-k:** Retrieval uses **top 10** hits per method; summaries in `qualitative_eval_runs.csv` show **top 5** titles/snippets per query.
- **Regenerate results:** From the repository root, with conda env `dsci575-ml` active: `make eval` (requires the sample semantic artifacts under `data/processed/`).

## Summary table (high level)

| Query ID | Difficulty | Preferred method (this sample) | Short rationale |
|----------|------------|----------------------------------|-----------------|
| q01 | easy | BM25 | Query is keyword-heavy (“ps5”, “wireless”, “controller”, “rechargeable”). BM25 surfaces chargers and battery accessories; semantic ranks DualShock/PS4 controllers highly — plausible embedding match to “controller” but weaker on the exact PS5 accessory intent. |
| q02 | easy | Tie / semantic slight | Both return fitness and Wii-era sports titles; neither strongly targets “Nintendo Switch Sports” as a product name — keyword overlap on “sports” and “fitness”. |
| q03 | easy | Tie | Both retrieve racing titles (Forza, Cars, Burnout, CTR); strong agreement. |
| q04 | easy | Tie | Both rank *Final Fantasy VII Remake* at the top; exact product-title match. |
| q05 | easy | Tie | Battery packs and charging docks dominate both lists; same product class. |
| q06 | easy | BM25 | BM25 ties “Minecraft” to Creeper controllers and *Minecraft: Story Mode*; semantic drifts to unrelated kid-friendly games without “Minecraft” in the title. |
| q07 | easy | Tie | *Call of Duty* titles appear in both; high lexical overlap. |
| q08 | easy | Tie | “Steam Deck” case ranks first in both; remaining hits mix Switch accessories (expected vocabulary overlap in reviews). |
| q09 | medium | BM25 | “Relaxing cozy calm story” — BM25 picks *Story of Seasons* and related life-sim vibes; semantic returns horror/puzzle/word games with weak intent match (embedding noise on a small corpus). |
| q10 | medium | Tie | Horror-adjacent titles (Resident Evil, Scratches, Metro, Among the Sleep, Dead by Daylight) appear in both; semantic adds *Silent Hill: Downpour*. |
| q11 | medium | BM25 | “Party / family multiplayer” — BM25 surfaces *Just Dance*, *Splatoon 2*, *Mario Party*-adjacent; semantic leans generic “family” games. |
| q12 | medium | BM25 | “Open world crafting exploration” — BM25 still pulls survival/open-world-ish titles; semantic returns obscure puzzle/sim titles with weak thematic fit. |
| q13 | medium | Neither strong | “Soulslike” is rare in this slice; both return generic combat/difficulty mentions. |
| q14 | medium | Tie | Sports titles (*Big League Sports*, *Kinect Sports*, *Rocket League*, *Wii Sports*) align on “sports”. |
| q15-q24 | mixed | (see CSV) | Inspect `qualitative_eval_runs.csv` for full top-10 `doc_id` lists and side-by-side summaries. |

*Preferred method* is a **qualitative judgment** for the team report, not a formal metric.

## Overall strengths and weaknesses

**BM25**

- **Strengths:** Strong when queries contain **specific product names, franchises, or rare tokens** (e.g., *Final Fantasy VII Remake*, *Steam Deck*, *Call of Duty*). Interpretable failure modes (empty hits when vocabulary does not overlap).
- **Weaknesses:** Misses **paraphrases and intent** (e.g., “headset for footsteps” without game-specific terms). Sensitive to tokenization; no notion of semantic similarity across synonyms.

**Semantic (sentence-transformers + FAISS)**

- **Strengths:** Can rank **conceptually related** items when embeddings align (e.g., horror/sports clusters in the sample). Helps when users describe **intent** with few exact keywords.
- **Weaknesses:** On a **small 1k-doc** slice, neighbors can be **off-topic** (e.g., wrong franchise or accessory class). Sensitive to **review noise** mixed into `retrieval_text`; can match “controller” broadly without “PS5” specificity.

## When BM25 fails but semantic helps (and the reverse)

- **Semantic better (potential):** Broad, intent-heavy queries where **wording differs** from product titles (e.g., paraphrased “relaxing story game” if the corpus contained more life-sim text). In our **q09** run, semantic did not outperform BM25 — illustrating that **semantic is not guaranteed to win** on small or biased samples.
- **BM25 better:** **Named entities** and **exact game or accessory names** (q04, q08, parts of q01/q05).
- **Both struggle:** **Complex constraints** (price ceiling + genre + platform) and **niche genres** (“soulslike”) if the slice lacks relevant documents.

## Are top results useful?

- For **easy** queries aligned with frequent tokens in Video Games reviews, top-5 often includes **relevant product families**.
- For **medium/complex** queries, **usefulness drops** when the 1k sample does not cover enough relevant SKUs — a limitation of **sample size**, not only the algorithm.

## Where hybrid / reranking / RAG might help

- **Hybrid:** Combine BM25 (precision on keywords) with semantic (recall on paraphrases), e.g., RRF or weighted fusion — matches a Streamlit app “Hybrid” mode if you add one later.
- **Reranking:** Cross-encoder or LLM rerank on top-20 candidates could fix **accessory vs. game** confusion (e.g., controllers vs. headsets).
- **RAG:** Useful when answers need **synthesis** beyond listing products; optional for retrieval-only comparisons.

## Artifacts

| File | Purpose |
|------|---------|
| `data/processed/ground_truth.csv` | Query IDs, text, difficulty; optional `relevant_doc_ids` for labeled evaluation |
| `data/processed/qualitative_eval_runs.csv` | Automated BM25 vs semantic summaries and top-10 `doc_id` lists |
| `src/run_qualitative_eval.py` | Regenerates `qualitative_eval_runs.csv` (invoked via `make eval`) |
| `src/utils.py` | `load_ground_truth`, `format_topk_for_eval`, etc. |

## Limitations (labels and corpus)

- **Labels:** `relevant_doc_ids` is empty; **Precision@k / Recall@k / MRR** need team agreement on **binary relevance** per query–doc pair or partial labels before automated metrics are meaningful.
- **Corpus mismatch:** Full **50k** `video_games_corpus_sample` BM25 index in the notebook **does not** match the **1k** semantic index; this evaluation script **forces alignment** on the 1k subset. Full-corpus comparisons need **one shared index size** for both methods.
