.PHONY: install install-dev install-advanced lint lint-fix type-check \
        test test-unit test-integration test-ci test-coverage \
        train train-fast train-advanced run run-prod bootstrap \
        gen-data docker-dev docker-prod clean help db-init monitor

PYTHON      = python
PYTHONPATH  = PYTHONPATH=.
DATA_PATH   = Data/maintenance.csv
CI_DATA     = Data/maintenance_ci.csv

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install: ## Install all runtime dependencies (via pyproject.toml)
	pip install --upgrade pip
	pip install -e .

install-advanced: ## Install advanced training extras (XGBoost, Optuna, MLflow)
	pip install -e .[advanced]

install-dev: ## Install dev + runtime deps + pre-commit hooks
	pip install --upgrade pip
	pip install -e .[dev]
	pre-commit install

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint: ## Run ruff linter
	ruff check src/ tests/ scripts/

lint-fix: ## Auto-fix lint issues where possible
	ruff check --fix src/ tests/ scripts/

type-check: ## Run mypy strict type checking
	mypy src/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test-unit: ## Run unit tests (no model or server required)
	$(PYTHONPATH) pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests (requires running server)
	$(PYTHONPATH) pytest tests/integration/ -v --tb=short

test-coverage: ## Run unit tests with full coverage report
	$(PYTHONPATH) pytest tests/unit/ -v \
		--cov=src --cov-report=term-missing --cov-report=html \
		--cov-fail-under=20

test: test-unit ## Run all tests that don't need external services
	@echo "Run 'make test-integration' with a live server for full coverage."

test-ci: ## Generate CI data, train, run full test suite
	$(PYTHONPATH) $(PYTHON) scripts/generate_ci_data.py --rows 600
	$(PYTHONPATH) $(PYTHON) src/models/train.py --data $(CI_DATA) --no-search
	$(PYTHONPATH) pytest tests/unit/ -v

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
train: ## Train model on the real dataset (requires Data/maintenance.csv)
	$(PYTHONPATH) $(PYTHON) src/models/train.py --data $(DATA_PATH)

train-fast: ## Train without grid search — useful for iteration
	$(PYTHONPATH) $(PYTHON) src/models/train.py --data $(DATA_PATH) --no-search

train-advanced: ## XGBoost + Optuna (requires install-advanced)
	$(PYTHONPATH) $(PYTHON) src/models/train_advanced.py --data $(DATA_PATH)

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
run: ## Start dev server with hot-reload
	$(PYTHON) run.py --reload

run-prod: ## Start production server (2 workers)
	$(PYTHON) run.py --workers 2

bootstrap: ## One-command environment setup (generate data + fast-train)
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -e .[dev]
	@echo "Generating synthetic dataset..."
	$(PYTHONPATH) $(PYTHON) scripts/generate_ci_data.py --rows 1000
	@echo "Baseline training..."
	$(PYTHONPATH) $(PYTHON) src/models/train.py --data $(CI_DATA) --no-search
	@echo "Sentinel is ready. Run 'make run' to start."

db-init: ## Initialize the prediction audit database
	$(PYTHON) -c "import asyncio; from src.core.database import init_db; asyncio.run(init_db())"

monitor: ## Spin up the full monitoring stack (Prometheus & Grafana)
	cd monitoring && docker compose up -d
	@echo "Monitoring stack is warming up..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

gen-data: ## Generate synthetic CI dataset
	$(PYTHONPATH) $(PYTHON) scripts/generate_ci_data.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-dev: ## Docker Compose — dev profile (hot-reload, source mounted)
	docker compose --profile dev up

docker-prod: ## Docker Compose — prod profile (immutable image, 2 workers)
	docker compose --profile prod up

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artefacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -f model.pkl
	rm -f artifacts/meta.json
