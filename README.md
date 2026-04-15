# DSCI 575 Project

This project implements retrieval over the **Video Games** category of [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023): **BM25** lexical search, **dense** retrieval (sentence embeddings + FAISS), and **hybrid** ranking via reciprocal rank fusion. A **Streamlit** application provides interactive search and optional relevance feedback; **offline evaluation** scripts output qualitative comparisons and precision/recall/MRR-style metrics against labeled queries.

Data processing:
We use the Amazon Reviews 2023 Video_Games category. Each retrieval document is built at the review level and enriched with product metadata. The final retrieval text combines product title, categories, features, description, review title, and review text. Preprocessing includes lowercasing, removing most punctuation, normalizing whitespace, and using whitespace tokenization for BM25. For efficiency, this project uses a representative sample rather than the full category.

## Badges

![Python](https://img.shields.io/badge/python-3.x-informational)
![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B)
![dotenv](https://img.shields.io/badge/config-.env%20%2B%20python--dotenv-yellow)

## Repository structure

```
.
├── README.md
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment specification (optional)
├── Makefile                 # shortcuts: install, raw, eval, metrics, dev, clean (see `make help`)
├── .env.example             # example environment variables (optional; copy to .env)
├── .gitignore               # ignores secrets, raw data, and local processed artifacts (small eval CSVs may be tracked)
├── data/
│   ├── raw/                 # downloaded *.jsonl (ignored)
│   └── processed/           # generated indices and eval outputs (ignored except whitelisted CSVs)
├── notebooks/
│   └── milestone1_exploration.ipynb  # EDA + preprocessing + sample indices
├── src/
│   ├── __init__.py          # marks `src` as a Python package
│   ├── bm25.py              # BM25 retriever
│   ├── semantic.py          # embedding + vector search
│   ├── retrieval_metrics.py # Precision@k, Recall@k, MRR
│   ├── retrieval.py         # index bundle discovery, load, RRF hybrid
│   ├── evaluation.py        # offline eval: ``python -m src.evaluation {qualitative|metrics|all}``
│   └── utils.py             # corpus construction + tokenization utilities
├── results/
│   └── milestone1_discussion.md  # qualitative evaluation notes
└── app/
    └── app.py               # Streamlit app (local)
```

## Setup

### 1) Create and activate a Python environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate dsci575-ml
```

### 2) Install dependencies

If you used `conda`, dependencies are installed as part of the environment creation step above.
If you used `venv`, install with pip:

```bash
pip install -r requirements.txt
```

### 3) Environment variables (optional)

Copy `.env.example` to `.env` only if you need non-default paths. The app uses **`PROCESSED_DATA_DIR`** (default `data/processed/`) and **`FEEDBACK_LOG_PATH`** (default `data/processed/app_feedback.csv`). Do not commit `.env`.

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

The milestone expects a **working app on your machine**; retrieval indices are **saved locally** (not required to be in Git).

1. **Install the environment** (sections *Setup* → 1–2 above).
2. **Download raw data** with `make raw` (needed for the exploration notebook).
3. **Open** `notebooks/milestone1_exploration.ipynb` and run through at least:
   - representative **sample corpus** build and save,
   - **BM25** build and save,
   - **semantic (embeddings + FAISS)** build and save.  
   This writes the notebook sample bundle under `data/processed/`, including:
   - `video_games_corpus_sample.parquet` or `video_games_corpus_sample.csv`
   - `bm25_sample_index.pkl`, `bm25_sample_tokens.pkl`
   - `faiss_sample.index`, `semantic_sample_metadata.pkl`
4. **Start the app** from the repo root (with `dsci575-ml` activated if using conda):

```bash
make dev
```

or:

```bash
streamlit run app/app.py
```

5. Open the URL shown in the terminal (default `http://127.0.0.1:8501`). If indices are missing, the app will error until step 3 completes successfully.

Optional: `make install` updates the conda env from `environment.yml` after dependency changes.

## Qualitative evaluation

With the sample retrieval bundle in `data/processed/`, regenerate BM25 vs semantic comparison rows for all queries in `data/processed/ground_truth.csv`:

```bash
conda activate dsci575-ml
make eval
```

This writes `data/processed/qualitative_eval_runs.csv`. Discussion notes belong in `results/milestone1_discussion.md`.

### Retrieval metrics

With `relevant_doc_ids` filled in `data/processed/ground_truth.csv` and the same sample artifacts as above:

```bash
conda activate dsci575-ml
make metrics
```

This writes `data/processed/retrieval_metrics_summary.csv` and `retrieval_metrics_per_query.csv`. See `results/milestone1_discussion.md` for interpretation.

Both steps use the same index bundle as the Streamlit app (`src/retrieval.discover_bundle()` loads the **notebook sample** bundle only). To run qualitative export and metrics in one command:

```bash
PYTHONPATH=. python -m src.evaluation all
```

## Makefile shortcuts

```bash
make help      # list targets
make install   # conda env update from environment.yml
make raw       # download Video_Games JSONL files into data/raw/ (Hugging Face)
make eval      # BM25 vs semantic comparison (ground_truth.csv → qualitative_eval_runs.csv)
make metrics   # P@k, R@k, MRR from labeled ground_truth.csv
make dev       # local Streamlit dev server
make clean     # remove __pycache__, *.pyc, data/raw downloads, and data/processed/* (except .gitkeep)
```

`make clean` deletes local downloads and processed outputs. Regenerate indices by re-running the notebook (and `make raw` if needed).

## Reproducibility checklist

- Create the environment from `environment.yml` or `requirements.txt`.
- Run `make raw`, then execute the sample corpus, BM25, and semantic sections of `notebooks/milestone1_exploration.ipynb` so `data/processed/` contains the app bundle.
- Run `make dev` to launch the Streamlit app locally.

## Milestone 2 LLM setup



## Workflow diagram

```mermaid
flowchart TD
    A[User Query] --> B[Retriever]
    B --> C[Top-k Documents]
    C --> D[Context Builder]
    D --> E[Prompt Template]
    E --> F[Hosted LLM API]
    F --> G[Generated Answer]

    subgraph Semantic RAG
        B1[Semantic Retriever / FAISS]
    end

    subgraph Hybrid RAG
        B2[BM25 Retriever]
        B3[Semantic Retriever]
        B4[Hybrid Merger / RRF]
    end

    A --> B1
    B1 --> C

    A --> B2
    A --> B3
    B2 --> B4
    B3 --> B4
    B4 --> C
```

