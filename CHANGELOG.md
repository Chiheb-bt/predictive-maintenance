# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.2.0] — 2026-04-01

### Added
- **FastAPI lifespan**: Model loading now happens via an explicit `lifespan` context
  manager rather than as a module-level import side effect. This makes unit tests
  cleaner, eliminates hidden startup failures, and aligns with FastAPI ≥ 0.93 best practice.
- **Structured logging via structlog**: All server logs are now emitted as structured
  key-value pairs. In a local terminal (TTY), output is human-readable and coloured;
  in production / Docker, output is newline-delimited JSON, ready for ingestion into
  ELK, Loki, Datadog, or CloudWatch.
- **prometheus-fastapi-instrumentator**: HTTP-level RED metrics (Rate, Errors, Duration)
  are now provided automatically for every endpoint by the instrumentator middleware,
  replacing hand-rolled latency timers in every route handler.
- **`pm_predictions_total` counter**: Business metric tracking prediction count by
  risk level, separate from the HTTP-level metrics.
- **mypy strict type checking**: `[tool.mypy]` block added to `pyproject.toml`.
  `make type-check` runs `mypy src/` and is gated in CI after the lint stage.
- **SECURITY.md**: Vulnerability reporting policy and known security considerations.
- **PEP 621 packaging**: `pyproject.toml` now contains a complete `[project]` table
  with optional dependency groups (`advanced`, `dev`). The project installs with
  `pip install -e .` — no separate `requirements.txt` required.

### Changed
- **Security**: API key comparison now uses `secrets.compare_digest` (constant-time),
  preventing timing attacks where an attacker could guess the key character-by-character
  by measuring server response time differences.
- **Version consistency**: Health endpoint (`GET /`) now correctly returns `"2.2.0"`
  (was `"2.1.0"` — a copy-paste oversight).
- **CI cache key**: GitHub Actions pip cache now keys on `pyproject.toml` instead of
  `requirements.txt`.
- **CI install**: `pip install -e .[dev]` replaces the previous `pip install -r requirements.txt`.
- **Smoke tests**: `load_model()` is now called explicitly in CI smoke tests, reflecting
  the new architecture (no auto-load at import time).
- **Dockerfile**: Dependencies installed via `pip install .` (from `pyproject.toml`).
- **pre-commit**: Added `mypy`, `check-toml`, and `check-merge-conflict` hooks.
- **Makefile**: New `type-check` and `test-coverage` targets; `install` targets updated
  to use `pip install -e .[...]`.

### Removed
- **`requirements.txt` / `requirements-advanced.txt`**: Superseded by `pyproject.toml`
  optional dependency groups.
- **Bespoke `_init_prometheus()` block**: ~40 lines of manual Prometheus boilerplate
  replaced by `prometheus-fastapi-instrumentator`.
- **`_METRICS_ENABLED` flag**: HTTP metrics are now always available; `GET /metrics`
  no longer returns 503 if `prometheus_client` is absent.

---

## [2.1.0] — 2026-03-15

### Added
- Drift detection endpoint (`POST /drift`) with per-feature Z-score analysis.
- HuggingFace Spaces deployment support (`huggingface_spaces/`).
- `scripts/download_model.py` for fetching trained models from GitHub Releases or HF Hub.
- `train_advanced.py` with XGBoost + Optuna hyperparameter search and MLflow logging.

### Changed
- Batch endpoint now returns a `summary` dict with counts by risk level.
- Coverage threshold enforced at 70% in CI.

---

## [2.0.0] — 2026-02-20

### Added
- Batch prediction endpoint (`POST /predict/batch`), supporting 1–100 machines per request.
- Prometheus metrics (`GET /metrics`) with request latency histogram.
- Gradio interactive UI embedded at `/ui`.
- Multi-stage Dockerfile with non-root user.
- GitHub Actions CI pipeline (lint → train → Docker).

### Changed
- Decision threshold moved from hardcoded 0.5 to F1-optimal value stored in `artifacts/meta.json`.
- Model pipeline serialises the full `sklearn.Pipeline` (preprocessor + calibrated classifier)
  into a single `model.pkl` — eliminating transform mismatch between training and serving.

---

## [1.0.0] — 2026-01-15

### Added
- Initial release: single `/predict` endpoint, RandomForest classifier, basic FastAPI app.
