# ─────────────────────────────────────────────────────────────────────────────
# CKD Predictor — Makefile
# Author: Abhishek Ashok Deshmukh · CSS 581 ML · UW Bothell
# Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON     := .venv/bin/python3.13
PYTHONPATH := src
STREAMLIT  := $(PYTHON) -m streamlit
UVICORN    := $(PYTHON) -m uvicorn
PYTEST     := $(PYTHON) -m pytest
PIP        := $(PYTHON) -m pip

.DEFAULT_GOAL := help

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: install
install:  ## Install all dependencies into .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

.PHONY: install-dev
install-dev: install  ## Install dev extras (ruff, pre-commit)
	$(PIP) install ruff pre-commit
	pre-commit install
	@echo "✅ Dev tools installed"

# ── Training ──────────────────────────────────────────────────────────────────
.PHONY: train
train:  ## Train all models and save artifacts
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train.py --verbose

.PHONY: train-quick
train-quick:  ## Train without generating evaluation plots
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train.py --no-plots --verbose

# ── Inference ─────────────────────────────────────────────────────────────────
.PHONY: predict
predict:  ## Batch predict from CSV (set INPUT= and OUTPUT=)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/predict.py \
		--input  $(or $(INPUT),  data/sample_patients.csv) \
		--output $(or $(OUTPUT), data/predictions.csv)

# ── Apps ──────────────────────────────────────────────────────────────────────
.PHONY: dashboard
dashboard:  ## Launch Streamlit dashboard (http://localhost:8501)
	PYTHONPATH=$(PYTHONPATH) $(STREAMLIT) run src/ckd/app/dashboard.py \
		--server.port 8501

.PHONY: api
api:  ## Launch FastAPI server with hot-reload (http://localhost:8000/docs)
	PYTHONPATH=$(PYTHONPATH) $(UVICORN) ckd.api.server:app \
		--reload --host 0.0.0.0 --port 8000

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test:  ## Run full pytest suite
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v

.PHONY: test-unit
test-unit:  ## Run only fast unit tests (no artifact required)
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_data.py tests/test_features.py \
		tests/test_models.py tests/test_db.py tests/test_monitoring.py -v

.PHONY: test-cov
test-cov:  ## Run tests with HTML coverage report
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		--cov=src/ckd --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────────
.PHONY: lint
lint:  ## Run ruff linter
	$(PYTHON) -m ruff check src/ scripts/ tests/

.PHONY: format
format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format src/ scripts/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:  ## Build Docker image
	docker build -t ckd-predictor:latest .

.PHONY: docker-api
docker-api:  ## Run API in Docker
	docker run -p 8000:8000 --env-file .env ckd-predictor:latest

.PHONY: docker-dashboard
docker-dashboard:  ## Run dashboard in Docker
	docker run -p 8501:8501 --env-file .env ckd-predictor:latest \
		streamlit run src/ckd/app/dashboard.py --server.address=0.0.0.0

# ── Database ──────────────────────────────────────────────────────────────────
.PHONY: db-init
db-init:  ## Initialise the predictions SQLite database
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from ckd.db.database import init_db; init_db(); print('DB ready')"

.PHONY: db-stats
db-stats:  ## Print prediction database statistics
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c \
		"from ckd.db.database import get_stats; import json; print(json.dumps(get_stats(), indent=2))"

# ── Drift monitoring ──────────────────────────────────────────────────────────
.PHONY: drift-report
drift-report:  ## Generate data drift report against training baseline
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_drift.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove generated files (keeps artifacts and .venv)
	rm -rf reports/__pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaned"

.PHONY: clean-artifacts
clean-artifacts:  ## Remove model artifacts (forces retrain)
	rm -rf artifacts/full_pipeline.joblib artifacts/metrics.json artifacts/xgb_pipeline.joblib
	@echo "🗑  Artifacts removed — run 'make train' to rebuild"

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
