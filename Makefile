# ==========================================
# Variables
# ==========================================

SHELL := /bin/bash
# Prefer active conda env Python when CONDA_PREFIX is set; otherwise python3
PYTHON := $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,python3)
PIP := $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/pip,pip3)

# Must match `name:` in environment.yml (dsci575-ml).
ENV_NAME := dsci575-ml

APP := app/app.py
RAW_DIR := data/raw

# Hugging Face dataset hub: McAuley-Lab/Amazon-Reviews-2023 (Video_Games category)
HF_BASE := https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main
URL_REVIEWS := $(HF_BASE)/raw/review_categories/Video_Games.jsonl?download=true
URL_META := $(HF_BASE)/raw/meta_categories/meta_Video_Games.jsonl?download=true

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
RESET := \033[0m

.PHONY: help install dev clean check-env raw eval metrics

help:
	@echo -e "$(YELLOW)DSCI 575 ML — project tasks$(RESET)"
	@echo "========================================================"
	@echo -e "  $(GREEN)make install$(RESET)  : Create/update Conda env $(ENV_NAME) from environment.yml"
	@echo -e "  $(GREEN)make raw$(RESET)      : Download Video_Games review + meta JSONL into data/raw/"
	@echo -e "  $(GREEN)make eval$(RESET)     : BM25 vs semantic comparison from ground_truth.csv → qualitative_eval_runs.csv"
	@echo -e "  $(GREEN)make metrics$(RESET)  : Precision@k, Recall@k, MRR from labeled ground_truth.csv"
	@echo -e "  $(GREEN)make dev$(RESET)      : Run Streamlit app (local dev server)"
	@echo -e "  $(GREEN)make clean$(RESET)    : Remove __pycache__, *.pyc, data/raw downloads, data/processed/*"
	@echo "========================================================"

# --- Environment: sync from environment.yml ---
install:
	@echo -e "$(YELLOW)Updating Conda environment ($(ENV_NAME))...$(RESET)"
	@conda env update -f environment.yml --prune
	@echo -e "$(GREEN)Environment ready.$(RESET)"
	@echo -e "$(YELLOW)Activate with:$(RESET)"
	@echo -e "    $(GREEN)conda activate $(ENV_NAME)$(RESET)"

check-env:
ifneq ($(CONDA_DEFAULT_ENV),$(ENV_NAME))
	@echo -e "$(RED)Error: conda env '$(ENV_NAME)' is not active.$(RESET)"
	@echo -e "$(YELLOW)Run: conda activate $(ENV_NAME)$(RESET)"
	@exit 1
endif

# --- Local dev: Streamlit ---
dev: check-env
	@echo -e "$(GREEN)Starting Streamlit (local dev)...$(RESET)"
	@$(PYTHON) -m streamlit run $(APP)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo -e "$(YELLOW)Removing data/raw downloads and data/processed contents (except .gitkeep)...$(RESET)"
	@[ -d "$(RAW_DIR)" ] && find "$(RAW_DIR)" -maxdepth 1 -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	@[ -d data/processed ] && find data/processed -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
	@echo -e "$(GREEN)Clean complete$(RESET)"

# --- Raw data: Amazon Reviews 2023 (Video_Games) from Hugging Face ---
raw:
	@echo -e "$(YELLOW)Downloading Amazon Reviews 2023 — Video_Games (reviews + metadata)...$(RESET)"
	@mkdir -p $(RAW_DIR)
	curl -fL "$(URL_REVIEWS)" -o "$(RAW_DIR)/Video_Games.jsonl"
	curl -fL "$(URL_META)" -o "$(RAW_DIR)/meta_Video_Games.jsonl"
	@echo -e "$(GREEN)Saved:$(RESET)"
	@echo "  $(RAW_DIR)/Video_Games.jsonl"
	@echo "  $(RAW_DIR)/meta_Video_Games.jsonl"

# --- Qualitative eval (requires sample FAISS + metadata under data/processed/) ---
eval: check-env
	@echo -e "$(GREEN)Running qualitative retrieval comparison...$(RESET)"
	@PYTHONPATH=. $(PYTHON) -m src.evaluation qualitative

# --- Retrieval metrics (requires relevant_doc_ids in ground_truth.csv) ---
metrics: check-env
	@echo -e "$(GREEN)Computing retrieval metrics...$(RESET)"
	@PYTHONPATH=. $(PYTHON) -m src.evaluation metrics
