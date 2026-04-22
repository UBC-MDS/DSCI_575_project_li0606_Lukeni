# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used: 10,000 unique products
- Changes to sampling strategy:
  - In the earlier workflow, we worked with a smaller representative sample.
  - For the final workflow, we changed the selection logic to scale by **unique products** instead of taking an arbitrary number of rows.
  - We selected 10,000 unique products and capped the number of reviews per product at 3 to keep the build practical on local hardware.
  - After filtering, the corpus used for retrieval contained 29,589 rows and still preserved 10,000 unique products.

- Notes on the final scaled workflow:
  - BM25 and semantic retrieval were both rebuilt on the same scaled corpus.
  - This ensured the final retrieval pipeline satisfied the minimum requirement of processing at least 10,000 products.
  - The final artifacts were saved using the final naming convention rather than the old sample naming convention.

### LLM Experiment

- Models compared:
  - Primary / original model: `llama-3.1-8b-instant`
  - Comparison model: `llama-3.3-70b-versatile`

- Prompt used:

```text
You are a careful product recommendation assistant.
Use only the retrieved Amazon review context below.
If the context is insufficient, say so clearly.
Keep the answer concise, helpful, and grounded in the retrieved evidence.
```

- Results and discussion:

We compared the two models on five product-search queries while keeping the retrieved context and prompt fixed. This allowed us to compare model behavior fairly without changing the retrieval side of the pipeline.

For the query **“best racing game with fun tracks”**, both models recommended *LittleBigPlanet Karting - Playstation 3*. The 70B model gave a more cautious answer and explicitly stated that the reviews did not clearly mention the track quality. The 8B model was shorter and more direct.

For the query **“story-rich scary game with dark atmosphere”**, the 8B model recommended *The Dark Pictures: House of Ashes* directly. The 70B model suggested *Among the Sleep* and also mentioned *House of Ashes* as an alternative. The 70B answer was more detailed, but also less decisive.

For the query **“good football game with realistic teams”**, the 8B model took a conservative approach and said that the retrieved context did not provide enough information to confirm realistic teams. The 70B model tried to infer realism from indirect clues such as comparisons to Madden. That made it somewhat more helpful, but also slightly less strict in staying tied to the evidence.

For the query **“family-friendly game for casual players”**, both models produced useful results. The 8B model returned multiple suggestions, while the 70B model focused on one strong recommendation. The 70B response was cleaner, but the 8B response gave the user more options.

For the query **“game with strong story and puzzle elements”**, both models recommended *The Book of Unwritten Tales*. The 70B model added an alternative title and more explanation, while the 8B model stayed shorter and more focused.

* Key observations:

  * `llama-3.3-70b-versatile` generally produced more detailed and nuanced responses.
  * `llama-3.1-8b-instant` was more concise and often more conservative when the evidence was weak.
  * The larger model was not clearly better for every query in this product-review setting.
  * In some cases, the extra detail from the 70B model came with mild over-interpretation beyond the retrieved evidence.
  * For this project, groundedness, response speed, and practical API usage are all important, not just answer length.

We decided to keep `llama-3.1-8b-instant` as the default model in the pipeline. Although `llama-3.3-70b-versatile` often produced richer answers, the 8B model remained more concise, more conservative when the retrieved evidence was limited, and more practical for a review-grounded shopping assistant. For this application, those qualities were more valuable than longer or more elaborate responses.

### Offline evaluation after scaling (analysis of new runs)

We rebuilt indices on the **final** scaled bundle (`discover_bundle` → `label=final`, **29,589** review rows, **10,000** products) and re-ran **`make eval`** and **`make metrics`**, refreshing:

| Output | Role |
|--------|------|
| `data/processed/qualitative_eval_runs.csv` | For each of **24** `ground_truth` queries: BM25 vs semantic top-5 text summaries and top-10 `doc_id` strings on the **large** index. |
| `data/processed/retrieval_metrics_*.csv` | Binary P@5, P@10, R@5, R@10, MRR using **unchanged** `relevant_doc_ids` cells in `ground_truth.csv`. |
| `results/milestone2_rag_eval_runs.json` | **10** fixed hybrid RAG queries; `"bundle_label": "final"`, `top_k=5`, `SYSTEM_PROMPT_V1`, Groq `llama-3.1-8b-instant`. |

**What is comparable, and what is not**

- **RAG eval (`milestone2_rag_eval_runs.json`):** Retrieved **`doc_id` strings and rank positions** are **not** meant to match the older JSON from the 1k index—the corpus row order and pool size changed. It **is** meaningful to compare **qualitatively**: whether answers stay grounded, whether top-5 **product titles** remain sensible for the same query (e.g. racing and narrative queries still surface recognizable games), and whether duplicate product titles in the top-5 list still come from **different** `doc_id`s (review-level grain). The new run shows that pattern unchanged; hits draw from a much larger candidate pool, so **IDs and ordering differ** while **intent coverage** of the 10 queries remains good for a demo.
- **Retrieval metrics (same label file, new index):**
  - **BM25 vs semantic on this single re-run** are **mutually comparable**: same queries, same (legacy) `doc_id` strings, same metric code—so saying “BM25’s aggregate scores are **higher** than semantic’s on this run” is **internally valid**.
  - The **magnitude of the scores** (and any **trend vs Milestone 1’s table in `milestone1_discussion.md`**) is **not** a fair measure of “retrieval got better or worse after scaling” because the labels were authored when each `doc_k` denoted a **row in the old ~1k table**. After rescaling, `doc_k` denotes the *k-th row of the new table*—usually a **different review and product** than the one the human labeler had in mind. The metric is then “how often is an arbitrary old index token hit,” not “how often is the intended relevant review hit.”
  - Therefore: use **Milestone 1’s P@5 ~0.19 vs ~0.08** as the authoritative **1k-corpus, label-aligned** snapshot; use the **new** numbers below as a **post-scaling diagnostic** on a **legacy label file**, not as a direct continuation of the same ground-truth experiment.

**New retrieval metrics (aggregate, `retrieval_metrics_summary.csv`)**

| Method | P@5 | P@10 | R@5 | R@10 | MRR |
|--------|-----|------|-----|------|-----|
| BM25 | 0.0083 | 0.0042 | 0.0417 | 0.0417 | 0.0417 |
| Semantic | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

`queries_with_labels` = 24 for both. **Interpretation of the gap:** on this (misaligned) label set, **lexical** retrieval still places the *labeled id string* in the top-10 for **one** query on average in the BM25 case (driving non-zero MRR and R@k); **dense** search hits **none** of the legacy id strings in top-5 or top-10 for any of the 24 rows—consistent with the fact that **semantic ordering** is even less likely to line up with an arbitrary old row index by chance.

**Per-query pattern (`retrieval_metrics_per_query.csv`)**

- For **BM25**, only **q04** (*final fantasy vii remake playstation*), with label `doc_1`, has non-zero scores: P@5 = 0.2, R@5 = R@10 = 1, reciprocal rank 1.0. All other **q01–q03, q05–q24** register **0** for both methods.
- For **semantic**, **all 24** queries are **0** across P@5, P@10, R@5, R@10, and RR.
- This pattern supports the view that the **aggregate** BM25 P@5 ≈ 1/24 ≈ 0.008 and MRR ≈ 1/24 ≈ 0.042 are largely **stochastic** with respect to “true” relevance on the 10k corpus, not a stable estimate of user-facing precision. **Qualitative** inspection of `qualitative_eval_runs.csv` (which titles appear in top-5 for each query) remains the right way to see how BM25 and semantic **behave** on the scaled data.

**How this relates to `milestone1_discussion.md` and `milestone2_discussion.md`**

- **Milestone 1:** The discussion table and per-query **preferences** (BM25 vs semantic) refer to a **1k** slice where labels and `doc_id`s were **aligned** by construction. The **refreshed** CSV on the 10k index **updates the side-by-side rankings** for the *same* natural-language query strings; readers should treat the **narrative** in Milestone 1 as methodology and the **new CSV** as the **current** BM25/semantic output on the production corpus.
- **Milestone 2:** Explains RAG **protocol** and duplicate titles. The new JSON extends that protocol to the **final** bundle; compare **answer quality and titles**, not `doc_id` identity across files.

**Practical takeaway**

- For **project reporting after scaling:** cite **this subsection and the new artifact paths** for “offline runs on 10k products,” and use **Milestone 1/2** for **methodology and historical 1k results**.
- For **defensible** precision/recall on the 10k stack, add a **relabeling** pass (relevant `doc_id`s chosen from a snapshot of `video_games_corpus_final` or from stable keys such as `parent_asin` + review id if exported).

## Step 2: Additional Feature (state which option you chose)

### What You Implemented

- Description of the feature
- Key results or examples
  
## Step 3: Improve Documentation and Code Quality

- created src/config.py and updated build_retriver.py to remove hardcoded path
- remove api key from .env.example
- ensure all functions have docstring


### Documentation Update
- Summary of `README` improvements

### Code Quality Changes
- Summary of cleanups

## Step 4: Cloud Deployment Plan
(See Step 4 above for required subsections)