# DSCI 575 Project

This project implements retrieval over the **Video Games** category of [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023): **BM25** lexical search, **dense** retrieval (sentence embeddings + FAISS), and **hybrid** ranking via reciprocal rank fusion. It also adds **RAG** (**semantic** or **hybrid** retrieval + **Groq** LLM) over the same review-level corpus. A **Streamlit** application exposes search-only and RAG modes; **offline evaluation** outputs qualitative comparisons, hybrid RAG JSON runs, and precision/recall/MRR-style metrics against labeled queries.

**Release:** [v0.2.0](https://github.com/UBC-MDS/DSCI_575_project_li0606_Lukeni/releases/tag/v0.2.0)

**Corpus scope:** The pipeline uses a **review-level** corpus from the Amazon Reviews 2023 **Video_Games** category: **10,000 unique products** and at most **three reviews per product** in the build (typically **~29k–30k** review rows; the exact row count is in `data/processed/video_games_corpus_final.*`). Each document combines product title, categories, features, description, review title, and review text, with the preprocessing described in the scaling notebook. Only this filtered subset is indexed (not the full category JSONL).

## Outcomes

- **Public Streamlit app:** [Link](https://dsci575li0606lukeni.streamlit.app/) — search (BM25 / dense / hybrid) and RAG over the scaled index.
- **Report:** [`results/final_discussion.md`](results/final_discussion.md) — dataset scaling, LLM comparison, offline evaluation, deployment plan, and code-quality notes.
- **Large `*_final` files** (not stored in this repo): [release `0.0.1` — *data_model_storage*](https://github.com/JayLBean/data_model_storage/releases/tag/0.0.1) (the hosted app can pull from this URL via `app` settings; not required to reproduce locally if you build indices yourself).

## Reproducibility (local)

1. **Environment:** `make install` → `conda activate dsci575-ml` (or `python -m venv .venv` + `pip install -r requirements.txt`; see *Setup* below).
2. **Raw inputs:** `make raw` → `data/raw/Video_Games.jsonl` and `meta_Video_Games.jsonl`.
3. **Build indices and corpus:** run `notebooks/milestone3_scaling.ipynb` to completion, **or** `python -m src.build_retrievers` — both produce the `*_final` files under `data/processed/`.
4. **Environment variables:** copy `.env.example` to `.env` and set at least **`GROQ_API_KEY`** for the RAG tab.
5. **Run the app:** from the repo root, `make dev` or `streamlit run app/app.py` and open the local URL (default `http://127.0.0.1:8501`).
6. **Optional — notebooks / Make:** `notebooks/milestone2_rag.ipynb` for additional RAG experiments; `make eval` and `make metrics` for the offline tables (see *Qualitative evaluation*). Earlier milestone write-ups: `results/milestone1_discussion.md`, `results/milestone2_discussion.md`.

## Badges

![Python](https://img.shields.io/badge/python-3.x-informational)
![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B)
![dotenv](https://img.shields.io/badge/config-.env%20%2B%20python--dotenv-yellow)

## Repository structure

```
.
├── README.md
├── requirements.txt         # Python dependencies (single source of truth for pip)
├── environment.yml          # Conda: Python + PyTorch base, then `pip install -r requirements.txt`
├── Makefile                 # shortcuts: install, raw, eval, metrics, dev, clean (see `make help`)
├── .env.example             # example environment variables (optional; copy to .env)
├── .gitignore               # ignores secrets, raw data, and local processed artifacts (small eval CSVs may be tracked)
├── data/
│   ├── raw/                 # downloaded *.jsonl (ignored)
│   └── processed/           # generated indices and eval outputs (ignored except whitelisted CSVs)
├── notebooks/
│   ├── milestone1_exploration.ipynb  # EDA + early sample indices (optional; dev history)
│   ├── milestone2_rag.ipynb         # RAG exploration: Groq, semantic vs hybrid, prompts
│   └── milestone3_scaling.ipynb     # Scaled 10k-product corpus + final BM25 / FAISS artifacts (app input)
├── docs/
│   └── RELEASE_NOTES_v0.2.0.md       # copy-paste body for GitHub Release v0.2.0
├── src/
│   ├── __init__.py          # marks `src` as a Python package
│   ├── bm25.py              # BM25 retriever
│   ├── semantic.py          # embedding + vector search
│   ├── retrieval_metrics.py # Precision@k, Recall@k, MRR
│   ├── retrieval.py         # index bundle discovery, load, RRF hybrid
│   ├── rag_pipeline.py      # semantic + hybrid RAG (Groq LLM)
│   ├── hybrid.py            # BM25 + dense hybrid retriever (RRF) for RAG
│   ├── evaluation.py        # offline eval: ``python -m src.evaluation {qualitative|metrics|milestone2_rag|eval|all}``
│   ├── milestone2_rag_eval.py   # hybrid RAG JSON → ``results/milestone2_rag_eval_runs.json``
│   ├── artifact_fetch.py   # optional: fetch `*_final` if FETCH + base URL in env
│   └── utils.py             # corpus construction + tokenization utilities
├── results/
│   ├── milestone1_discussion.md   # qualitative retrieval evaluation notes
│   ├── milestone2_discussion.md   # RAG qualitative discussion / evaluation
│   └── milestone2_rag_eval_runs.json  # generated by ``make eval`` (requires ``GROQ_API_KEY``)
└── app/
    └── app.py               # Streamlit app (local)
```

## Setup

### 1) Create and activate a Python environment

The **conda** environment name is **`dsci575-ml`** (hyphen between `575` and `ml`, not an underscore).

#### Conda (recommended): `make install`

You only need a working **conda** (Miniconda or Anaconda) on your PATH. You **do not** need to create `dsci575-ml` beforehand, and you **do not** need to `conda activate` anything before running this.

From the repository root:

```bash
make install
```

This runs:

```bash
conda env update -f environment.yml --prune
```

That command **creates** the environment if it is missing, or **updates** it if it already exists; `--prune` removes packages that were dropped from `environment.yml`. Then install the pip stack from `requirements.txt` as specified in the YAML.

After it finishes, activate:

```bash
conda activate dsci575-ml
```

Equivalent without Make: run `conda env update -f environment.yml --prune` yourself, then `conda activate dsci575-ml`.

#### `venv` (no conda)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Dependencies inside the conda env

`requirements.txt` is the canonical list of pip packages (used for `venv`, CI, Streamlit Cloud, and as the `-r` file for conda). The conda path uses `environment.yml` for a small conda base (Python 3.11, NumPy, PyTorch) and then installs pip dependencies via `make install` / `conda env update` as above. If you already use `venv` only, install with:

```bash
pip install -r requirements.txt
```

### 3) Environment variables

Copy `.env.example` to `.env` when you need overrides. Do not commit `.env`.

**Paths (retrieval app):** **`PROCESSED_DATA_DIR`** (default `data/processed/`) and **`FEEDBACK_LOG_PATH`** (default `data/processed/app_feedback.csv`).

**Groq API (RAG):** `src/rag_pipeline.py` reads **`GROQ_API_KEY`** and optional **`LLM_MODEL`** (see `.env.example`). Without `GROQ_API_KEY`, the **Search** tab still works; the **RAG** tab shows an error until the key is set. Create your own key at [Groq Console](https://console.groq.com/keys) and put it in `.env` only.

### 4) Dependency changes after `git pull`

If `requirements.txt` or `environment.yml` changed, refresh the conda env from the repo root (no need to activate first):

```bash
make install
```

Or manually: `conda env update -f environment.yml --prune`. If you use **venv** only: `pip install -r requirements.txt`.

---

## RAG pipeline

Natural-language answers use **only** retrieved Amazon review rows as context, generated by a hosted LLM. Each request picks **one** RAG mode; the diagram below is the single path the code implements (`src/rag_pipeline.py`, Streamlit **RAG** tab).

```mermaid
flowchart TD
    Q[User query]
    Q --> MODE{RAG mode}
    MODE -->|Semantic RAG| SEM["SemanticRetriever (FAISS + embeddings)"]
    SEM --> DOCS[Top-k review rows]
    MODE -->|Hybrid RAG| BM[BM25Retriever]
    MODE -->|Hybrid RAG| SEM2[SemanticRetriever]
    BM --> RRF[Reciprocal rank fusion]
    SEM2 --> RRF
    RRF --> DOCS
    DOCS --> CTX["build_context (ASIN, title, rating, review text)"]
    CTX --> PROMPT[System prompt + context + question]
    PROMPT --> GROQ[ChatGroq on Groq API]
    GROQ --> ANS[Generated answer]
```

**Following the diagram:** **Semantic RAG** sends the query only through **SemanticRetriever** (sentence embeddings + FAISS) to obtain top-`k` rows. **Hybrid RAG** runs **BM25** and **SemanticRetriever** in parallel, merges ranked lists with **reciprocal rank fusion (RRF)**, then keeps top-`k`. Both modes produce the same kind of **review rows** (`DOCS`), which **build_context** turns into a text block (ASIN, title, rating, review text). That block is combined with a **system prompt** and the user question into a single prompt string, then **ChatGroq** calls the Groq API to produce the final answer. The **Search** tab uses the same retrievers but skips `build_context` and Groq: it shows ranked hits only (typically top 10).

| Step | Where in code | Role |
|------|---------------|------|
| Semantic retrieval only | `SemanticRAGPipeline` in `src/rag_pipeline.py` | Dense search → top-`k` → shared prompt path. |
| Hybrid retrieval | `HybridRAGPipeline` + `HybridRetriever` in `src/hybrid.py` | BM25 + dense → RRF → top-`k` → same prompt path as semantic-only. |
| Offline hybrid runs | `src/milestone2_rag_eval.py` | Fixed query set → `results/milestone2_rag_eval_runs.json` (see `results/milestone2_discussion.md`). |
| Notebook exploration | `notebooks/milestone2_rag.ipynb` | API checks, retrieval vs full RAG, prompt variants V1–V3, optional exports under `data/processed/`. |

**System prompts** `SYSTEM_PROMPT_V1`–`V3` are defined in `src/rag_pipeline.py`; the Streamlit **RAG** tab can use presets or a custom system string (same context + question wrapping as in code). **Optional `src/tools.py`:** not used; no agent tools are wired into RAG.

**Model and credentials:** RAG uses **Groq** via `langchain-groq` (`ChatGroq`). Set **`GROQ_API_KEY`** and optionally **`LLM_MODEL`** in `.env` (default model `llama-3.1-8b-instant` if unset). See *Environment variables* above.

## Download the raw dataset

This project uses the **Video_Games** category from the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset. Two files are required: review records and product metadata.

From the repository root:

```bash
make raw
```

| File | Role |
|------|------|
| `data/raw/Video_Games.jsonl` | Reviews |
| `data/raw/meta_Video_Games.jsonl` | Product metadata |

Requires `curl`. Downloads can take several minutes.

## Run the Streamlit app locally

You need a **working app on your machine**; retrieval indices are **saved locally** (not required to be in Git).

1. **Install the environment** (sections *Setup* → 1–2 above).
2. **Download raw data** with `make raw` (required to build the processed corpus and indices from source).

3. **Build the retrieval artifacts**

The app and `src.retrieval.discover_bundle()` require **one** complete **final** bundle under `data/processed/` (Parquet or CSV corpus plus the four index files). Two equivalent paths:

### Option A: Python build entrypoint

```bash
python -m src.build_retrievers
```

Writes the scaled artifacts, including:

* `video_games_corpus_final.parquet` (or `video_games_corpus_final.csv`)
* `bm25_final_index.pkl`, `bm25_final_tokens.pkl`
* `faiss_final.index`, `semantic_final_metadata.pkl`

### Option B: Scaling notebook

Run all cells in `notebooks/milestone3_scaling.ipynb` to reproduce EDA, filtering, and the same `*_final` outputs.

4. **Configure environment variables**

Create a local `.env` file and set:

```bash
GROQ_API_KEY=[REPLACE_WITH_YOUR_KEY]
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
```
Do not commit `.env` to GitHub.

5. **Start the app** from the repo root:

```bash
conda activate dsci575-ml
make dev
```

The Makefile target `dev` checks that conda env **`dsci575-ml`** is active. If you use **venv** instead, run Streamlit directly (no `check-env`):

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

6. Open the URL shown in the terminal (default `http://127.0.0.1:8501`). If indices are missing, the app will error until step 3 completes successfully.

### App modes

| Tab | What it does | Requirements |
|-----|----------------|--------------|
| **Search** | BM25, Semantic, or **Hybrid** (RRF) over the scaled **final** index — **no LLM**. Top **10** hits. | `*_final` bundle in `data/processed/` (see step 3) |
| **RAG** | **Semantic RAG** (dense only) or **Hybrid RAG** (BM25 + dense → RRF). **Top 5** reviews feed the prompt. Preset **V1–V3** or custom system prompt. | Same bundle + **`GROQ_API_KEY`** |

How **Semantic RAG** vs **Hybrid RAG** connect to **build_context** and Groq is shown in the **RAG pipeline** section (workflow diagram above).

Optional: `make install` updates the conda environment **`dsci575-ml`** from `environment.yml` after dependency changes.

### RAG exploration notebook

Run `notebooks/milestone2_rag.ipynb` after a retrieval bundle exists and `.env` is configured. It is **not** required to launch the app; the production path uses the **`*_final`** artifacts from `milestone3_scaling.ipynb` (or `build_retrievers`).

## Qualitative evaluation

With the **final** retrieval bundle in `data/processed/`:

```bash
conda activate dsci575-ml
make eval
```

This runs: (1) BM25 vs semantic comparison for every query in `data/processed/ground_truth.csv` → `data/processed/qualitative_eval_runs.csv`; (2) **hybrid RAG** on a fixed 10-query set → `results/milestone2_rag_eval_runs.json` (requires **`GROQ_API_KEY`** in `.env`). The write-ups in `results/milestone1_discussion.md` and `results/milestone2_discussion.md` reference runs on a **1,000-row development index**; after scaling, re-executing these commands regenerates the CSV/JSON against the **10,000-product** corpus. Interpretation of the refresh is summarized in `results/final_discussion.md`.

To run only the qualitative CSV or only the RAG JSON:

```bash
PYTHONPATH=. python -m src.evaluation qualitative
PYTHONPATH=. python -m src.evaluation milestone2_rag
```

### Retrieval metrics

With `relevant_doc_ids` filled in `data/processed/ground_truth.csv` and the **final** artifacts above:

```bash
conda activate dsci575-ml
make metrics
```

This writes `data/processed/retrieval_metrics_summary.csv` and `data/processed/retrieval_metrics_per_query.csv`. Ground-truth labels were defined when the index used a **1k-row** sample; `doc_id` is assigned by **row order** in each build, so **aggregate P@k / R@k / MRR on the scaled corpus are not directly comparable** to the Milestone 1 table in `milestone1_discussion.md` as “the same” relevance experiment—see `results/final_discussion.md` for a concise discussion. Use the updated CSVs to compare methods **on the current index** or relabel for strict relevance on the 10k-product corpus.

`src/retrieval.discover_bundle()` loads the **final** bundle only. To run qualitative export plus metrics (without the RAG JSON step):

```bash
PYTHONPATH=. python -m src.evaluation all
```

## Makefile shortcuts

```bash
make help      # list targets
make install   # update conda env dsci575-ml from environment.yml
make raw       # download Video_Games JSONL files into data/raw/ (Hugging Face)
make eval      # qualitative CSV + hybrid RAG JSON (needs GROQ_API_KEY for JSON)
make metrics   # P@k, R@k, MRR from labeled ground_truth.csv
make dev       # local Streamlit dev server
make clean     # remove __pycache__, *.pyc, data/raw downloads, and data/processed/* (except .gitkeep)
```

`make clean` deletes local downloads and most processed outputs. Rebuild the **`*_final`** bundle with `milestone3_scaling.ipynb` or `python -m src.build_retrievers` (and `make raw` if source JSONL is missing). For a full local reproduction sequence, use **Reproducibility** at the top of this README.
