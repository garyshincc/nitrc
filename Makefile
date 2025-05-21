# NITRC CMI Resaerch

PROJ_DIR = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3

.PHONY: setup
setup:
	@echo "Setting up environment..."
	$(PYTHON) -m venv venv
	pip install -U pip setuptools wheel toml
	pip install -r requirements.txt

.PHONY: install
install:
	@echo "Installing Python packages..."
	pip install -r requirements.txt

.PHONY: fix
fix: ## Lint code, auto-fix and generate pipelines' configurations
	@echo "Formatting $(PROJ_DIR) ..."
	isort $(PROJ_DIR)
	black $(PROJ_DIR)
	ruff check --fix $(PROJ_DIR)

	@echo "\nNo auto-fix for mypy, manual fixes may be required"
	mypy $(PROJ_DIR)

	@echo "Done, please 'git add' changes if any."

requirements.txt: pyproject.toml $(PROJ_DIR)/ci/generate_requirements.py ## Generate requirements.txt from pyproject.toml
	$(PYTHON) $(PROJ_DIR)/ci/generate_requirements.py -i $< -o $@

.PHONY: clean_cache
clean_cache:
	@echo "Clearing Data Cache..."
	rm -rf data_cache/*
	rm -rf data_cache_cleaned/*
