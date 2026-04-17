# Milestone 2 — RAG qualitative discussion

## How to run the Milestone 2 RAG eval

1. **Environment:** `conda activate dsci575-ml` (or any env with `requirements.txt` installed).
2. **Secrets:** set **`GROQ_API_KEY`** in `.env` (see `.env.example`).
3. **Data:** build the notebook sample bundle under `data/processed/` (same as the Streamlit app).
4. **Generate JSON** (pick one):

   ```bash
   # Only Milestone 2 hybrid RAG JSON
   PYTHONPATH=. python -m src.evaluation milestone2_rag

   # Milestone 1 qualitative CSV + Milestone 2 JSON (recommended)
   make eval
   ```

   Outputs:

   - `data/processed/qualitative_eval_runs.csv` — BM25 vs semantic (Milestone 1).
   - `results/milestone2_rag_eval_runs.json` — hybrid RAG answers + retrieval rows (Milestone 2).

The JSON schema is a single object with keys `bundle_label`, `top_k`, `system_prompt`, and **`results`**: an array of `{ id, query, answer, retrieved_titles_top5, retrieved_doc_ids_top5, unique_doc_ids_in_top5 }`. Implementation: **`src/milestone2_rag_eval.py`** (invoked from **`src/evaluation.py`** via `milestone2_rag` or `eval`).

We did **not** compute quantitative RAG metrics (per course handout). The table below uses **manual** yes/no on three dimensions after reading each answer **and** the same JSON file.

---

## Query design: principles and characteristics

The ten strings live in **`MILESTONE2_RAG_QUERIES`** inside `src/milestone2_rag_eval.py`. They were chosen to:

- **Stay in-domain** for **Amazon Video Games** reviews (games, peripherals, accessories).
- **Mix query styles:** keyword-heavy (e.g. PS5 controller), short genre/intent (racing, horror, cozy), yes/no (Minecraft), and budget/constraints (headset under $50).
- **Stress-test retrieval + LLM:** some queries expect **sparse** evidence in a **1k review** sample (e.g. soulslike, Steam Deck “travel”, strict price caps).
- **Stay stable for reproducibility:** fixed list so `make eval` always regenerates comparable JSON for this report.

---

## Repeated product titles in `retrieved_titles_top5` (e.g. id 1 and id 3)

Our corpus is **review-level**: each row is one review with metadata fields such as **`product_title`** and **`doc_id`**. The **hybrid** retriever merges BM25 and semantic hits with **RRF over `doc_id`**, so the top-5 list contains **five distinct reviews** (`unique_doc_ids_in_top5 == 5` in our runs).

**Why the same title can appear more than once**

- **Different reviews for the same product (same ASIN / product)** can each rank in the top-5. For example, **id 1** lists *LittleBigPlanet Karting - Playstation 3* three times with **different** `doc_id`s (`doc_17602`, `doc_37695`, `doc_22239`) — three separate reviews that all strongly match the query, so RRF keeps all three.
- **id 3** repeats *HyperX Cloud II Wireless…* at ranks 1, 3, and 5 with **different** `doc_id`s — again, multiple reviews for one headset that mention battery / PS5 compatibility in ways that match lexical and semantic signals, even though the user asked for a **controller**; the duplicate titles make it obvious that retrieval is **review-centric**, not **product-deduplicated**.

**Implications**

- The model sees **multiple snippets** from the same product; answers may over-weight that product unless the prompt pushes diversity or we **dedupe by `parent_asin`** before building context.
- For UI or reports, we could **collapse by product** and keep the highest-scoring review per SKU; the current pipeline does **not** dedupe, which is why duplicates appear in the JSON.

---

## How this report was produced (reproducible)

- **Pipeline:** `HybridRAGPipeline` in `src/rag_pipeline.py` (BM25 + dense + RRF, **top_k = 5**).
- **System prompt:** `SYSTEM_PROMPT_V1` (same as default in code).
- **LLM:** Groq **`llama-3.1-8b-instant`** via `GROQ_API_KEY` in `.env`.
- **Corpus:** Notebook sample bundle under `data/processed/` (same 1k-doc setting as Milestone 1).
- **Raw runs:** **`results/milestone2_rag_eval_runs.json`** produced by **`make eval`** or **`python -m src.evaluation milestone2_rag`**.

---

## Evaluation dimensions (yes / no each)

| Dimension | What we check |
|-----------|----------------|
| **Accuracy** | Claims are supported by the **retrieved** review text/metadata in the run (no invented prices, player counts, or products not grounded in context). |
| **Completeness** | The answer addresses the question’s intent **as far as the retrieved context allows** (we mark **No** when evidence is thin or the model hedges correctly but leaves the user without a full answer). |
| **Fluency** | Clear, readable English and sensible structure. |

---

## Ten-query summary (all rated)

Same ten queries as in **`src/milestone2_rag_eval.py`** → `MILESTONE2_RAG_QUERIES`. Ratings refer to the saved answers under **`results`** in **`milestone2_rag_eval_runs.json`**.

| # | Query | Acc | Cmp | Flu |
|---|-------|-----|-----|-----|
| 1 | best racing game with fun tracks | Y | N | Y |
| 2 | story-rich scary game with dark atmosphere | Y | Y | Y |
| 3 | good wireless PS5 controller with long battery life | Y | N | Y |
| 4 | relaxing cozy game for stress relief | Y | Y | Y |
| 5 | competitive online FPS with a large player base | N | N | Y |
| 6 | Steam Deck games that work well for travel | Y | N | Y |
| 7 | family-friendly party multiplayer on Nintendo Switch | Y | Y | Y |
| 8 | Is Minecraft good for creative kids? | Y | Y | Y |
| 9 | soulslike or very hard action RPG recommendations | N | N | Y |
| 10 | best gaming headset under $50 with clear mic for Discord | N | N | Y |

**Notes on ratings (tied to actual outputs):**

- **Q1:** Recommends *LittleBigPlanet Karting* (also repeated three times in `retrieved_titles_top5` as **different** `doc_id`s); “enjoyable tracks” is inferred more from genre than from quoted track detail → **Cmp = N**.
- **Q5:** Names Overwatch from context and cites matchmaking / online play, but **“large player base”** is still not directly evidenced → **Acc = N**, **Cmp = N**.
- **Q6:** Honestly states **no** Steam Deck **games** in context (mostly accessories) → **Acc = Y**, **Cmp = N**.
- **Q9:** Bloodborne is grounded in context; the answer also brings in **Dark Souls** as an implicit comparison / sequel discussion, which is **not** clearly supported by the top-5 rows alone → **Acc = N**.
- **Q10:** Discusses headsets and prices; **$50** and “clear mic for Discord” are **not** reliably grounded in the short review excerpts in the JSON → **Acc = N**, **Cmp = N**.

---

## Five queries in detail (verbatim excerpts from saved JSON)

Excerpts below are **copied from** `results/milestone2_rag_eval_runs.json` → `results[].answer`, trimmed only with `[...]` where noted.

### Q1 — best racing game with fun tracks

**Retrieved top-5 titles (from JSON):** *LittleBigPlanet Karting - Playstation 3* appears **three times** (positions 1, 3, 5) with **different** `doc_id`s; also Phar Lap Horse Racing Challenge; Burnout Revenge.

**Answer excerpt:**

> Based on the provided context, I would recommend the "LittleBigPlanet Karting - Playstation 3" (ASIN: B0050SX00Y). This game has a rating of 5.0 and multiple reviews mentioning that it's a fun game with a variety of difficulties suitable for players of all ages, which suggests that it has enjoyable tracks.

**Ratings:** Acc **Y**; Cmp **N**; Flu **Y**.

---

### Q2 — story-rich scary game with dark atmosphere

**Retrieved top-5 titles:** Fatal Frame 2; Among the Sleep (Switch); Little Nightmares; Scratches; Among the Sleep (PS4).

**Answer excerpt:**

> 1. Fatal Frame 2: Crimson Butterfly (ASIN: B0000AI1KK) - described as "one of the best survival horror games out there" with a "great story" and "atmosphere the developers created" [...]  
> 2. Scratches: Directors Cut (ASIN: B000KMCF0G) - described as "dark" with "dark ambiance, dark motivations, dark secrets" [...]  
> However, the review for Scratches: Directors Cut provides the most detailed description of a story-rich scary game with a dark atmosphere.

**Ratings:** Acc **Y**; Cmp **Y**; Flu **Y**.

---

### Q3 — good wireless PS5 controller with long battery life

**Retrieved top-5 titles:** HyperX Cloud II Wireless (headset) appears **three times** (positions 1, 3, 5) with **different** `doc_id`s; DualSense PS5; PowerA Switch controller.

**Answer excerpt:**

> Based on the provided reviews, it seems that the Sony PlayStation Dualsense Wireless Controller (ASIN: B08GSL374K) has received positive reviews, but there is no mention of its battery life. However, the PowerA Enhanced Wireless Controller for Nintendo Switch (ASIN: B0BCHWZX95) has been praised for its battery life, lasting a long time. Unfortunately, there is no direct review for a "good wireless PS5 controller with long battery life" in the provided context.

**Ratings:** Acc **Y** (honest limitation); Cmp **N**; Flu **Y**.

---

### Q5 — competitive online FPS with a large player base

**Retrieved top-5 titles:** Razer mouse pad; Rocket League; Ratchet & Clank; MySims Party; Overwatch GOTY.

**Answer excerpt:**

> Based on the reviews provided, it seems that the competitive online FPS with a large player base is Overwatch - Game of the Year Edition.  
> [...]  
> This is mentioned in the review for ASIN: B017QU5KLK, where the reviewer states that "there's always online action available if you want" and that the matchmaking is "very good and doesn't put you in with people who are out of your league skill-wise."

**Ratings:** Acc **N** (FPS label OK; **player base** not shown in JSON excerpts); Cmp **N**; Flu **Y**.

---

### Q10 — best gaming headset under $50 with clear mic for Discord

**Retrieved top-5 titles:** BEAVIIOO headset; Amazon Basics; CORSAIR Void Pro; Dragon War; HyperX Cloud Revolver S.

**Answer excerpt:**

> Based on the reviews provided, I would recommend the BEAVIIOO Wireless Gaming Headset with Microphone (ASIN: B0BTTHYVBJ) for its clear mic performance on Discord [...] It's also mentioned as a budget-friendly option that performs well for a low price.  
> [...]  
> Please note that the HyperX Cloud Revolver S (ASIN: B01N7WNV5M) is also a high-rated option, but its price is not explicitly mentioned in the reviews provided, and it is likely above $50.

**Ratings:** Acc **N** (price / Discord claims not tightly checkable from titles alone); Cmp **N**; Flu **Y**.

---

## Overall observations

- **Retrieval noise dominates bad answers:** Q5’s top-5 includes non-FPS items first; Q6’s top-5 is mostly **hardware**, not games; the model can still write fluent paragraphs.
- **Honest “insufficient context”** (Q3, Q6) scores well on **Accuracy** but poorly on **Completeness**—which is appropriate for a small sample.
- **Franchise / genre queries** (Q2, Q7, Q8) align better with retrieved horror/party/Minecraft rows.

---

## Limitations (2–4)

1. **Single automated run** — no repeated sampling at different temperatures; ratings are for one execution of `make eval` / `python -m src.evaluation milestone2_rag`.
2. **1k sample corpus** — retrieval and answers change if the corpus or indices change.
3. **Manual yes/no** — coarse; borderline cases (e.g. Q1) could be debated.
4. **JSON is the source of truth** — if you edit prompts or `top_k`, regenerate JSON and update this file.

---

## Improvement directions (brief)

- **Rerank** top-20 → top-5 (cross-encoder or lightweight scorer) to drop accessories when the query asks for **games** (Q6) or to improve FPS relevance (Q5).
- **Stricter prompt** when retrieval is mixed (require citing review quotes for price or player-count claims).
- **Larger index** or **query classification** (game vs accessory) before retrieval.

---

## Artifacts

| File | Purpose |
|------|---------|
| `results/milestone2_rag_eval_runs.json` | Raw hybrid RAG answers + top-5 titles per query |
| `src/milestone2_rag_eval.py` | Fixed queries + JSON writer; run via `src.evaluation` or `make eval` |
| `src/rag_pipeline.py` | Pipeline definition |
| `notebooks/milestone2_rag.ipynb` | Exploratory comparisons (prompts, semantic vs hybrid) |
