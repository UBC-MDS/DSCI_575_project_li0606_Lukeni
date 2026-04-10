# DSCI 575 Project

Information retrieval with BM25 and embeddings on the Amazon Reviews 2023 dataset.

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
├── .env.example             # example environment variables (copy to .env)
├── .gitignore               # ignores secrets, raw data, and local artifacts
├── data/
│   ├── raw/                 # downloaded *.jsonl (NOT committed; folder kept via .gitkeep)
│   └── processed/           # cleaned / chunked / indices (folder kept via .gitkeep)
├── notebooks/
│   └── milestone1_exploration.ipynb  # EDA + preprocessing notebook
├── src/
│   ├── __init__.py          # marks `src` as a Python package
│   ├── bm25.py              # BM25 retriever
│   ├── semantic.py          # embedding + vector search
│   ├── retrieval_metrics.py # Precision@k, Recall@k, MRR
│   ├── run_qualitative_eval.py
│   ├── run_retrieval_metrics.py
│   └── utils.py             # corpus construction + tokenization utilities
├── results/
│   └── milestone1_discussion.md  # qualitative evaluation notes
└── app/
    └── app.py               # Streamlit retrieval demo
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

### 3) Configure environment variables

Copy the example environment file and edit values as needed:

```bash
cp .env.example .env
```

Notes:
- Do **not** commit `.env`.
- Large/raw dataset files should live in `data/raw/` and are ignored by git.

## Download the raw dataset

This project uses the **Video_Games** category from the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset on Hugging Face. Two files are required: review records and product metadata.

From the repository root, download both into `data/raw/`:

```bash
make raw
```

This fetches:

| File | Role |
|------|------|
| `data/raw/Video_Games.jsonl` | Reviews |
| `data/raw/meta_Video_Games.jsonl` | Product metadata |

Requires `curl` (available by default on macOS/Linux). The files are large; the command may take several minutes depending on your network.

## Qualitative evaluation

After building the **sample** semantic artifacts in `data/processed/` (`faiss_sample.index`, `semantic_sample_metadata.pkl` — e.g. from `notebooks/milestone1_exploration.ipynb`), regenerate BM25 vs semantic comparison rows for all queries in `data/processed/ground_truth.csv`:

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

## Running the app

With conda env `dsci575-ml` activated:

```bash
make dev
```

Or directly:

```bash
streamlit run app/app.py
```

### Makefile shortcuts

```bash
make help      # list targets
make install   # conda env update from environment.yml
make raw       # download Video_Games JSONL files into data/raw/ (Hugging Face)
make eval      # BM25 vs semantic comparison (ground_truth.csv → qualitative_eval_runs.csv)
make metrics   # P@k, R@k, MRR from labeled ground_truth.csv
make dev       # local Streamlit dev server
make clean     # remove __pycache__ / *.pyc
```

## Reproducibility checklist

- A TA should be able to clone the repo, create an environment, set `.env` values, and run the app.
- Run `make raw` once to pull Amazon Reviews 2023 (Video_Games) into `data/raw/` before notebooks or `src/build_retrievers.py`.
