# ==========================================
# Variables
# ==========================================

SHELL := /bin/bash
# Prefer active conda env Python when CONDA_PREFIX is set; otherwise python3
PYTHON := $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,python3)
PIP := $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/pip,pip3)

ENV_NAME := dsci575-ml

APP := app/app.py

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
RESET := \033[0m

.PHONY: help install dev clean check-env

help:
	@echo -e "$(YELLOW)DSCI 575 ML — project tasks$(RESET)"
	@echo "========================================================"
	@echo -e "  $(GREEN)make install$(RESET)  : Create/update Conda env from environment.yml"
	@echo -e "  $(GREEN)make dev$(RESET)      : Run Streamlit app (local dev server)"
	@echo -e "  $(GREEN)make clean$(RESET)    : Remove __pycache__ and *.pyc"
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
	@echo -e "$(GREEN)Clean complete$(RESET)"
