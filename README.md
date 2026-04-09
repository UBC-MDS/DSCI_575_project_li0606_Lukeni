# DSCI 575 Project (Milestone 1)

Information Retrieval with BM25 and Embeddings on the Amazon Reviews 2023 dataset.

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
├── Makefile                 # shortcuts: install, dev, clean (see `make help`)
├── .env.example             # example environment variables (copy to .env)
├── .gitignore               # ignores secrets, raw data, and local artifacts
├── data/
│   ├── raw/                 # downloaded *.jsonl.gz (NOT committed; folder kept via .gitkeep)
│   └── processed/           # cleaned / chunked / indices (folder kept via .gitkeep)
├── notebooks/
│   └── milestone1_exploration.ipynb  # EDA + preprocessing notebook
├── src/
│   ├── __init__.py          # marks `src` as a Python package
│   ├── bm25.py              # BM25 retriever implementation (Milestone 1)
│   ├── semantic.py          # embedding + vector search implementation (Milestone 1)
│   ├── retrieval_metrics.py # optional: Precision@k / Recall@k / MRR, etc.
│   └── utils.py             # corpus construction + tokenization utilities
├── results/
│   └── milestone1_discussion.md  # qualitative evaluation notes for Milestone 1
└── app/
    └── app.py               # single app, updated each milestone
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

For Milestone 1, this project uses the `Video_Games` category from the Amazon Reviews 2023 dataset.

Download the datasets into exiting data folders:

```bash
curl -L "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Video_Games.jsonl?download=true" -o data/raw/Video_Games.jsonl

curl -L "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl?download=true" -o data/raw/meta_Video_Games.jsonl
```



## Running the app (Milestone 1)

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
make dev       # local Streamlit dev server
make clean     # remove __pycache__ / *.pyc
```

## Reproducibility checklist

- A TA should be able to clone the repo, create an environment, set `.env` values, and run the app.
- Dataset files (Amazon Reviews 2023) must be downloaded separately and placed under `data/raw/`.
