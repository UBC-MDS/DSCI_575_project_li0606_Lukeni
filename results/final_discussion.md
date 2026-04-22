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

**Option chosen:** **Option 4 — Deploy the application** (public, persistent URL).

### What we implemented

- **Public app:** The Streamlit interface is hosted on **Streamlit Community Cloud** at [**https://dsci575li0606lukeni.streamlit.app/**](https://dsci575li0606lukeni.streamlit.app/) (BM25, semantic, hybrid **Search**; **RAG** with Groq). The link is also listed under **Outcomes** in the root `README.md`.
- **Why the main repo does not store indices:** The scaled `*_final` artifacts (corpus CSV, BM25 and FAISS-related files) total on the order of **hundreds of MB**; they are listed in `.gitignore` so clones stay small and Git history does not bloat.
- **Where the large files live:** We version them in a **separate public repository** as **GitHub Release assets** (tag `0.0.1`): [JayLBean/data_model_storage — Release 0.0.1](https://github.com/JayLBean/data_model_storage/releases/tag/0.0.1). The five objects match what `src.retrieval.discover_bundle()` expects for the `final` bundle.
- **How the hosted app gets data:** The application repository is **connected to Streamlit Cloud** in the usual way (GitHub integration: **pull** the app code on deploy). In **Secrets**, we set **`GROQ_API_KEY`**, **`FETCH_REMOTE_ARTIFACTS=1`**, and **`REMOTE_ARTIFACTS_BASE_URL`** to the release download prefix. At startup, `src.artifact_fetch` (called from `app/app.py`) **streams** any missing `*_final` files into the configured `PROCESSED_DATA_DIR` before indices load. Local development is unchanged: build `data/processed/` with `milestone3_scaling.ipynb` or `python -m src.build_retrievers` and no remote fetch is required.
- **Design choice:** A **dedicated storage repo** plus **per-release** URLs is simple for this course, avoids LFS on the app repo, and matches “code in one place, blobs versioned under another.”

## Step 3: Improve Documentation and Code Quality

### Documentation Update

- **`README.md`:** Describes **corpus scope** (10k products, review-level ~29.6k rows), **reproducibility** (environment, `make raw`, scaling notebook or `build_retrievers`, `.env`, `make dev`, optional `make eval` / `make metrics` and other notebooks), and **outcomes** (public Streamlit URL, this file, and the **asset** release for large files—not a deploy tutorial for third parties).
- **`results/final_discussion.md`:** Single written report for the final milestone (this document).
- **`.env.example`:** Local Groq and optional `FETCH` / `REMOTE_ARTIFACTS_BASE_URL` for the hosted case; no secrets in committed files.

### Code Quality Changes

- **`src/config.py`:** Central paths and defaults; builds use the same `*_final` names as the app.
- **`src/retrieval.py`:** `discover_bundle()` loads only the **scaled `final`** artifact set; removed reliance on the old sample filenames for production.
- **`src/artifact_fetch.py`:** Minimal helper to download the five `*_final` files when `FETCH_REMOTE_ARTIFACTS` and `REMOTE_ARTIFACTS_BASE_URL` are set (hosted Streamlit).
- **Secrets:** API keys are read from **environment** / Streamlit **Secrets**, not from source; `.env` is gitignored.
- **Docstrings:** New and updated modules follow the one-line (or more) docstring pattern where appropriate.
- **`.gitignore`:** Continues to exclude `data/raw/*`, `data/processed/*` (with small tracked CSV exceptions as before), and `.env`.

## Step 4: Cloud Deployment Plan

This section meets the handout’s requirement to assume a **hypothetical** production stack on a major cloud (we use **AWS** below) and to cover **data placement**, **compute / concurrency / LLM**, and **updates**. The **Streamlit + sub-repo release** stack above is our **current** class demo; the following is a **forward-looking** design.

### 1. Data storage

| Asset | **Current (course)** | **Planned on AWS (example)** |
|--------|----------------------|------------------------------|
| **Raw** reviews + meta JSONL | Local / optional `make raw`; not on Streamlit by default | **S3** bucket prefix `s3://…/raw/` (or AWS Open Data / curated snapshot); access via IAM roles for batch jobs, not the browser. |
| **Processed** review-level corpus (Parquet/CSV) | `video_games_corpus_final.*` in GitHub Release or local `data/processed/` | **S3** `…/processed/corpus/corpus_v{n}.parquet` with version prefix or object tags. |
| **Vector (FAISS) index** + **semantic metadata** | `faiss_final.index`, `semantic_final_metadata.pkl` in the same release or local disk | **S3** `…/indices/semantic/…`; containers or Lambdas read via **pre-signed URL** or **instance profile**; optional **EFS** mount if a service needs POSIX paths and shared read across tasks. |
| **BM25** (pickles, token lists) | `bm25_final_*.pkl` — same as above | **S3** `…/indices/bm25/…` next to versioned semantic artifacts so each **environment** (dev/stage/prod) points at one **immutable** prefix. |

**Rationale:** S3 is durable, pay-per-GB, and supports **versioning** and **lifecycle** rules for old index generations.

### 2. Compute

- **Where the app runs (AWS pattern):** **Amazon ECS on Fargate** or **App Runner** (or **EKS** if we later split services): one **container image** per release that contains the app code and Python stack but **not** the multi-hundred-MB indices (or only a small cache). **CloudFront** in front of **Application Load Balancer** can serve static assets and TLS if we add a custom domain.
- **Concurrent users:** The Streamlit *process* model is not ideal for high concurrency; in AWS we would either run **one task per user session** (expensive) or replace the UI with a **stateless API** (FastAPI) + separate front end, and scale out **Fargate tasks** or use **API Gateway** + **Lambda** for throttled, short-lived work. For **RAG** specifically, the heavy work is **embedding/vector search** (CPU/GPU) and **LLM** calls.
- **LLM inference:** We continue to use an **API** (today **Groq**; in AWS one could add **Amazon Bedrock** for managed models or **SageMaker** for a dedicated endpoint). A production setup would **route** traffic: default answers through Groq/Bedrock, with **SageMaker** for optional fine-tuned or larger models, and **IAM + usage quotas** to control cost. **“AI + collaboration”** in operations means: use **CloudWatch** alarms, **SageMaker Model Monitor** or **Bedrock** evaluation jobs for **quality regression** checks, and **human-in-the-loop** relabeling (e.g. new `ground_truth` rows in S3) to refresh offline metrics after corpus updates.

### 3. Streaming / updates

- **New products / reviews:** A **scheduled** job (e.g. **EventBridge** → **Fargate task** or **AWS Glue**) runs the same **notebook/build pipeline** on fresh raw data, writes a **new** `corpus_v{n}` and indices to **S3**, and registers a **version id** in **SSM Parameter Store** or a small **DynamoDB** row so the app reads **only the active version**.
- **Zero-downtime cutover:** The web tier reads **version** from config; after validation, flip the pointer to the new S3 prefix; **stale** objects can be expired under **lifecycle** policy.
- **How the pipeline “stays up to date”:** Re-run ingestion when **Amazon Reviews** snapshots or internal feeds change; **monitor** drift (e.g. recall on a fixed query set stored in S3) before promoting a new index to production.

This AWS sketch aligns with what we already practice at small scale—**versioned artifacts**, **separation of code and data**, and **API-based LLM**—while describing a path to **managed storage (S3)**, **elastic compute**, and **governed model usage** in production.