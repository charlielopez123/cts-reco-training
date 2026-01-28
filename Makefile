# ---- Config ---------------------------------------------------------------
RDATA ?= data/raw/original_R_dataset.RData
WHATSON_CSV ?= data/raw/original_raw_whatson.csv

OUT_PROCESSED ?= data/processed/programming/processed_programming.parquet
OUT_ENRICHED ?= data/processed/programming/programming_enriched.parquet
ML_FEATURES_PATH ?= data/processed/programming/ML_features.parquet

MODEL_OUTPUT ?= data/models/audience_ratings_model.joblib

OUT_WHATSON_INIT_EXTRACTION ?= data/processed/whatson/whatson_extracted_movies.parquet
OUT_WHATSON_ENRICHED ?= data/processed/whatson/whatson_catalogue_enriched_tmdb.parquet
OUT_WHATSON_CATALOG ?= data/processed/whatson/whatson_catalog.parquet

OUT_HISTORICAL_PROGRAMMING ?= data/processed/programming/historical_programming.parquet

OUT_IL_TRAINING_DATA ?= data/processed/IL/training_data.joblib
OUT_CURATOR_MODEL ?= data/models/curator_logistic_model.joblib
OUT_CTS_MODEL ?= data/models/cts_model.npz

# Curtain mode outputs (21-dim context)
OUT_IL_TRAINING_DATA_CURTAIN ?= data/processed/IL/curtain_mode/training_data_curtain.joblib
OUT_CURATOR_MODEL_CURTAIN ?= data/models/curtain_mode/curator_logistic_model_curtain.joblib
OUT_CTS_MODEL_CURTAIN ?= data/models/curtain_mode/cts_model_curtain.npz

LOG_LEVEL ?= INFO

##--------- Hyperparameters Config -------------------------------------------
CURATOR_GAMMA ?= 0.4  # Weighting for curator signal in IL reward function
CTS_LR_WARMSTART ?= 0.005  # Learning rate for warm-start (conservative to avoid overfitting to binary data)


# SSL Configuration - Use certifi's CA bundle for HTTPS requests (macOS fix)
export REQUESTS_CA_BUNDLE := $(shell .venv/bin/python -c "import certifi; print(certifi.where())" 2>/dev/null)

# ---- Defaults -------------------------------------------------------------
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.PHONY: help setup bootstrap lint fmt fix test prepare-programming train-audience-ratings extract-whatson historical-programming extract-IL-training-data extract-IL-training-data-curtain train-curator-model train-curator-model-curtain train-cts-model train-cts-model-curtain interactive-test run-all clean env

help: ## List available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## / - /'

run-all: ## Run the complete end-to-end pipeline
	@echo "=== Starting complete pipeline ==="
	@echo "Step 1/7: Processing programming data (RData -> processed -> enriched -> ML features)"
	$(MAKE) prepare-programming
	@echo ""
	@echo "Step 2/7: Extracting WhatsOn catalog"
	$(MAKE) extract-whatson
	@echo ""
	@echo "Step 3/7: Building historical programming (matching broadcasts to catalog)"
	$(MAKE) historical-programming
	@echo ""
	@echo "Step 4/7: Training audience ratings model"
	$(MAKE) train-audience-ratings
	@echo ""
	@echo "Step 5/7: Extracting IL training data"
	$(MAKE) extract-IL-training-data
	@echo ""
	@echo "Step 6/7: Training curator model"
	$(MAKE) train-curator-model
	@echo ""
	@echo "Step 7/7: Training CTS model with warm-start"
	$(MAKE) train-cts-model
	@echo ""
	@echo "=== Pipeline complete! ==="
	@echo "Outputs:"
	@echo "  - Programming: $(OUT_PROCESSED)"
	@echo "  - Enriched: $(OUT_ENRICHED)"
	@echo "  - ML Features: $(ML_FEATURES_PATH)"
	@echo "  - WhatsOn Catalog: $(OUT_WHATSON_CATALOG)"
	@echo "  - Historical Programming: $(OUT_HISTORICAL_PROGRAMMING)"
	@echo "  - Audience Model: $(MODEL_OUTPUT)"
	@echo "  - IL Training Data: $(OUT_IL_TRAINING_DATA)"
	@echo "  - Curator Model: $(OUT_CURATOR_MODEL)"
	@echo "  - CTS Model: $(OUT_CTS_MODEL)"

setup: ## Install / sync deps
	uv sync

bootstrap: ## One-time dev setup
	uv sync
	uv run pre-commit install

lint: ## Lint
	LOG_LEVEL=$(LOG_LEVEL)uv run ruff check .

fmt: ## Format
	LOG_LEVEL=$(LOG_LEVEL) uv run ruff format .

fix: ## Lint with fixes
	LOG_LEVEL=$(LOG_LEVEL) uv run ruff check --fix .

test: ## Run tests
	LOG_LEVEL=$(LOG_LEVEL) uv run pytest -q

prepare-programming: ## Build programming parquet from RData
	mkdir -p $(dir $(OUT_PROCESSED))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-prepare-programming --rdata $(RDATA) --out_processed $(OUT_PROCESSED) --out_enriched $(OUT_ENRICHED) --out_ml $(ML_FEATURES_PATH)
# If no console script, use:
#	uv run python -m cts_recommender.cli.preprocessing_programming --rdata $(RDATA) --out $(OUT)

train-audience-ratings: ## Train audience ratings model
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-audience-ratings --ml_features $(ML_FEATURES_PATH) --model_output $(MODEL_OUTPUT)
# If no console script, use:
#	uv run python -m cts_recommender.cli.train_audience_ratings --ml_features $(ML_FEATURES_PATH) --model_output $(MODEL_OUTPUT)

extract-whatson: ## Process WhatsOn catalog (extract movies, enrich with TMDB, build additional features)
	mkdir -p $(dir $(OUT_WHATSON_CATALOG))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-extract-whatson \
		--csv $(WHATSON_CSV) \
		--out_init_extraction $(OUT_WHATSON_INIT_EXTRACTION) \
		--out_enriched $(OUT_WHATSON_ENRICHED) \
		--out_catalog $(OUT_WHATSON_CATALOG)

historical-programming: ## Build historical programming by matching broadcasts to catalog
	mkdir -p $(dir $(OUT_HISTORICAL_PROGRAMMING))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-historical-programming \
		--programming $(OUT_ENRICHED) \
		--catalog $(OUT_WHATSON_CATALOG) \
		--out $(OUT_HISTORICAL_PROGRAMMING)

extract-IL-training-data: ## Extract imitation learning training data
	mkdir -p $(dir $(OUT_IL_TRAINING_DATA))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-extract-IL-training-data \
		--historical $(OUT_HISTORICAL_PROGRAMMING) \
		--catalog $(OUT_WHATSON_CATALOG) \
		--audience-model $(MODEL_OUTPUT) \
		--out $(OUT_IL_TRAINING_DATA) \
		--gamma $(CURATOR_GAMMA)

train-curator-model: ## Train curator logistic regression model
	mkdir -p $(dir $(OUT_CURATOR_MODEL))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-curator-model \
		--training-data $(OUT_IL_TRAINING_DATA) \
		--out $(OUT_CURATOR_MODEL) \

train-cts-model: ## Train Contextual Thompson Sampler with warm-start
	mkdir -p $(dir $(OUT_CTS_MODEL))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-cts-model \
		--training-data $(OUT_IL_TRAINING_DATA) \
		--curator-model $(OUT_CURATOR_MODEL) \
		--out $(OUT_CTS_MODEL) \
		--lr-warmstart $(CTS_LR_WARMSTART) \
		--lr-bias-ratio 1.0

# ---- Curtain Mode Targets (21-dim context) --------------------------------
extract-IL-training-data-curtain: ## Extract curtain mode IL training data (21-dim context)
	mkdir -p $(dir $(OUT_IL_TRAINING_DATA_CURTAIN))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-extract-IL-training-data \
		--historical $(OUT_HISTORICAL_PROGRAMMING) \
		--catalog $(OUT_WHATSON_CATALOG) \
		--audience-model $(MODEL_OUTPUT) \
		--out $(OUT_IL_TRAINING_DATA_CURTAIN) \
		--gamma $(CURATOR_GAMMA) \
		--context-mode rts_curtain

train-curator-model-curtain: ## Train curtain mode curator logistic regression model
	mkdir -p $(dir $(OUT_CURATOR_MODEL_CURTAIN))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-curator-model \
		--training-data $(OUT_IL_TRAINING_DATA_CURTAIN) \
		--out $(OUT_CURATOR_MODEL_CURTAIN)

train-cts-model-curtain: ## Train curtain mode CTS with warm-start (21-dim context)
	mkdir -p $(dir $(OUT_CTS_MODEL_CURTAIN))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-cts-model \
		--training-data $(OUT_IL_TRAINING_DATA_CURTAIN) \
		--curator-model $(OUT_CURATOR_MODEL_CURTAIN) \
		--out $(OUT_CTS_MODEL_CURTAIN) \
		--lr-warmstart $(CTS_LR_WARMSTART) \
		--lr-bias-ratio 1.0

interactive-test: ## Run interactive CTS testing in terminal
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-interactive-test \
		--cts-model $(OUT_CTS_MODEL) \
		--curator-model $(OUT_CURATOR_MODEL) \
		--audience-model $(MODEL_OUTPUT) \
		--catalog $(OUT_WHATSON_CATALOG) \
		--historical-programming $(OUT_HISTORICAL_PROGRAMMING)

clean: ## Clean caches
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info

env: ## Show env info
	uv --version
	uv run python -V
	uv run python -c "import sys,site;print('site-packages:', site.getsitepackages()); print('python:', sys.executable)"
