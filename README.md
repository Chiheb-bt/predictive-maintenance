# Sentinel: Industrial AI Predictive Maintenance

[![CI](https://github.com/Chiheb-bt/predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/Chiheb-bt/predictive-maintenance/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/demo-HuggingFace%20Spaces-orange.svg)](https://huggingface.co/spaces/Chiheb-bt/predictive-maintenance)

A production ML system that predicts industrial machine failures from real-time sensor readings. Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (~10,000 machines, 3.4% failure rate).

---

## Why I built this

I built this project to move beyond simple Jupyter notebooks and actually solve the hard parts of deploying an ML model into production. I spent a lot of time engineering a solid foundation—like decoupling model loading from import times using FastAPI's `lifespan` manager, using `structlog` so the app's telemetry is fully observable, and wiring up Prometheus explicitly for endpoint metrics. Beyond just throwing together an API, I wanted to handle edge cases properly: the backend protects itself against timing attacks on the API key using constant-time string comparisons, and gracefully falls back to a 503 degraded state if the model bytes are missing instead of just crashing the container at boot. The architecture guarantees there is zero chance for training-serving skew, since the preprocessing transforms are baked directly into the exact same `.pkl` pipeline that's scored by a fully calibrated classifier. It's the kind of reliable, predictable backend that I'd want to inherit if I were joining an ML team tomorrow!

---

## Model performance

Results on the held-out test set (20% of the dataset, never seen during training or threshold selection):

| Metric | This model | Naive baseline¹ |
|---|---|---|
| F1 (failure class) | **0.82** | 0.00 |
| Precision | **0.79** | — |
| Recall | **0.87** | — |
| ROC-AUC | **0.96** | 0.50 |

¹ Naive baseline always predicts "no failure" — it achieves 96.6% accuracy but catches zero actual failures. These numbers show what the model actually adds.

Decision threshold is selected on a separate **validation set** by maximising F1 on the precision-recall curve, then written to `artifacts/meta.json`. The threshold used in production always matches the one that was optimised — not a generic 0.5.

---

## What it does

Send sensor readings, get a structured risk assessment back:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-key" \
  -d '{
    "Type": "M",
    "Air_temperature": 298.1,
    "Process_temperature": 308.6,
    "Rotational_speed": 1551,
    "Torque": 42.8,
    "Tool_wear": 0
  }'
```

```json
{
  "request_id": "a3f7c291-4e2b-4d3a-9f1e-bc82d5e0f3a1",
  "prediction": 0,
  "probability": 0.0312,
  "confidence": 0.9376,
  "risk_level": "LOW",
  "status": "Operating Normally",
  "recommendations": [
    "No action required — continue normal operations.",
    "Log readings and re-evaluate at the next scheduled inspection.",
    "Confirm tool wear tracking is current."
  ],
  "top_factors": [
    { "rank": 1, "feature": "Tool Wear",       "importance": 0.38 },
    { "rank": 2, "feature": "Torque",           "importance": 0.27 },
    { "rank": 3, "feature": "Rotational Speed", "importance": 0.19 }
  ]
}
```

---

## Architecture

```
POST /predict
     │
     ▼
FastAPI (main.py)          — Pydantic validation + optional API key auth
     │
     ▼
inference.py               — alias mapping, degraded-state handling
     │
     ▼
sklearn Pipeline           — StandardScaler + OHE + CalibratedRandomForest
     │
     ▼
risk_engine.py             — risk tier, sensor-aware recommendations, explainability
```

The preprocessor is serialised inside `model.pkl` alongside the model — a single
`pipeline.predict_proba()` call handles scaling, encoding, and inference with no
opportunity for a transform mismatch.

---

## Project structure

```
.
├── src/
│   ├── core/
│   │   ├── preprocessing.py       Feature schema, ColumnTransformer, field aliases
│   │   └── risk_engine.py         Risk classification, recommendations, explainability
│   ├── models/
│   │   ├── train.py               RandomForest pipeline (calibration, threshold, drift stats)
│   │   └── train_advanced.py      XGBoost + Optuna + MLflow (optional)
│   ├── serving/
│   │   └── inference.py           Single and batch prediction, degraded-state handling
│   └── app/
│       └── main.py                FastAPI + Gradio UI + Prometheus + drift endpoint
├── tests/
│   ├── conftest.py                Real tiny pipeline fixture (session-scoped, no mocking)
│   ├── unit/                      Pure function tests — no model or server required
│   └── integration/               End-to-end API tests (requires running server)
├── scripts/
│   ├── generate_ci_data.py        Synthetic dataset generator for CI
│   └── download_model.py          Download model.pkl from GitHub Releases or HF Hub
├── huggingface_spaces/
│   ├── app.py                     Entry point for HF Spaces deployment
│   └── README.md                  Spaces configuration card
├── artifacts/                     Auto-generated: meta.json after training
├── Data/                          Place maintenance.csv here (not committed)
├── .github/workflows/ci.yml       Three-stage CI: lint → pipeline → Docker
├── Dockerfile                     Multi-stage, non-root user
├── docker-compose.yml             dev and prod profiles
├── Makefile                       make install / train / test / run
└── requirements.txt
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Chiheb-bt/predictive-maintenance
cd predictive-maintenance
make install

# 2. Get the dataset
# Download AI4I 2020 CSV → Data/maintenance.csv
# https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

# 3. Train
make train            # full pipeline with grid search
make train-fast       # skip grid search, use defaults

# OR download a pre-trained model:
python scripts/download_model.py

# 4. Start the server
make run              # development (hot-reload)
make run-prod         # production (2 workers)
```

Open:
- **Interactive UI** — http://localhost:8000/ui
- **API docs** — http://localhost:8000/docs
- **Health check** — http://localhost:8000/
- **Prometheus metrics** — http://localhost:8000/metrics

---

## Configuration

All runtime settings are environment variables. Copy `.env.example` to `.env`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Path to the trained pipeline |
| `MODEL_THRESHOLD` | *(from meta.json)* | Decision threshold — auto-loaded from training artefact |
| `API_KEY` | *(empty)* | Set to enable `X-Api-Key` auth on `/predict` and `/predict/batch` |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `PORT` | `8000` | Server port |

When `API_KEY` is unset, `/predict` is publicly accessible and a startup warning is logged.

---

## Authentication

The `/predict` and `/predict/batch` endpoints support optional API key authentication:

```bash
# With auth enabled (API_KEY env var is set):
curl -H "X-Api-Key: your-secret-key" -X POST http://localhost:8000/predict ...

# Without auth (local dev — API_KEY not set):
curl -X POST http://localhost:8000/predict ...
```

Health (`GET /`), metrics (`GET /metrics`), and drift (`POST /drift`) endpoints are
always public — load balancers, Prometheus scrapers, and monitoring systems need
unauthenticated access.

---

## API reference

### `GET /`

Health check. Returns service status, model path, uptime, and whether auth/metrics are enabled. Suitable as a Kubernetes readiness probe.

### `POST /predict`

Score a single machine. See the request/response example at the top of this README.

| Field | Type | Range | Description |
|---|---|---|---|
| `Type` | string | L, M, H | Machine quality grade |
| `Air_temperature` | float | 200–400 | Air temperature (Kelvin) |
| `Process_temperature` | float | 200–400 | Process temperature (Kelvin) |
| `Rotational_speed` | int | 0–10000 | Rotational speed (RPM) |
| `Torque` | float | 0–500 | Torque (Nm) |
| `Tool_wear` | int | 0–500 | Accumulated tool wear (minutes) |

### `POST /predict/batch`

Score 1–100 machines in a single request. Returns per-row results plus a risk level summary — designed for fleet dashboards where you need to update many machines at once without paying per-call overhead.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "X-Api-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '[
    {"Type": "M", "Air_temperature": 298.1, "Process_temperature": 308.6, "Rotational_speed": 1551, "Torque": 42.8, "Tool_wear": 0},
    {"Type": "H", "Air_temperature": 310.0, "Process_temperature": 320.0, "Rotational_speed": 1800, "Torque": 68.0, "Tool_wear": 220}
  ]'
```

```json
{
  "n_requested": 2,
  "n_succeeded": 2,
  "n_failed": 0,
  "summary": {"LOW": 1, "HIGH": 1},
  "predictions": [...]
}
```

### `POST /drift`

Check whether recent sensor readings have drifted away from the training distribution. Useful for detecting sensor calibration issues or operating condition changes *before* they degrade model accuracy.

For each numeric feature, computes `z = (incoming_mean − training_mean) / training_std` and classifies it as:
- **NORMAL** — `|z| < 2`
- **WARNING** — `2 ≤ |z| < 3`
- **DRIFT** — `|z| ≥ 3` (investigate sensors or consider retraining)

```bash
curl -X POST http://localhost:8000/drift \
  -H "Content-Type: application/json" \
  -d '{"readings": [<10 or more sensor reading objects>]}'
```

```json
{
  "n_samples": 50,
  "overall_status": "WARNING",
  "features": [
    {"feature": "Rotational speed [rpm]", "training_mean": 1538.8, "incoming_mean": 1612.4, "z_score": 2.1, "status": "WARNING"},
    {"feature": "Air temperature [K]",    "training_mean": 300.0,  "incoming_mean": 299.9,  "z_score": -0.1, "status": "NORMAL"}
  ],
  "assessed_at": "2026-03-31T14:22:01+00:00",
  "note": "Minor drift detected in some features. Monitor closely over the next collection window."
}
```

No auth required — monitoring systems need unauthenticated access.

### `GET /metrics`

Prometheus metrics: prediction counts by risk level, request latency histogram, failure probability distribution, and error counts.

---

## Testing

```bash
# Unit tests — no model or server required
make test-unit

# Full pipeline (trains on synthetic data, then tests)
make test-ci

# Integration tests (start the server first)
make run &
make test-integration
```

Unit tests use a real tiny pipeline trained on synthetic data rather than mocking the
model — this catches actual inference bugs that mocking would hide. Coverage threshold
enforced at 70%: `--cov-fail-under=70`.

---

## Docker

```bash
make train           # model.pkl must exist before building

# Development (source mounted, hot-reload)
make docker-dev

# Production (immutable image, 2 workers)
make docker-prod
```

---

## Model and artefacts versioning

`model.pkl` is excluded from the repository. Two options for sharing a trained model:

**GitHub Releases (recommended):** Tag a release, attach `model.pkl`, then download it:

```bash
python scripts/download_model.py --github-repo Chiheb-bt/predictive-maintenance --version v2.2.0
```

**Hugging Face Hub:**

```bash
python scripts/download_model.py --hf-repo Chiheb-bt/predictive-maintenance
```

Both commands write `model.pkl` to the current directory and optionally verify it with `--verify`.

---

## Deploying to Hugging Face Spaces

The `huggingface_spaces/` directory contains a self-contained Gradio app that runs the interactive demo on Spaces — the fastest way to get a public live URL.

1. Create a new Space (SDK: Gradio) at huggingface.co/new-space
2. Push this repo to it
3. Upload `model.pkl` (from a GitHub Release or your local training run) to the Space files
4. The demo starts automatically

The Space runs the Gradio UI only. For the full REST API, deploy via Docker to Fly.io, Render, or Railway.

---

## Advanced training

`train_advanced.py` runs an Optuna hyperparameter study on XGBoost, calibrates the best trial, and optionally logs everything to MLflow.

```bash
make install-advanced     # installs xgboost, optuna, mlflow

make train-advanced

mlflow ui                 # http://localhost:5000
```

The output `model.pkl` is a drop-in replacement — the inference layer accepts both pipelines without modification.

---

## Deployment

```bash
docker build -t pm-api:latest .
docker push your-registry/pm-api:latest
```

**Railway / Render / Fly.io**: connect the repo, set `MODEL_PATH`, `API_KEY`, and `PORT`. The `GET /` endpoint works as both a readiness and liveness probe.

**Kubernetes**: the `GET /` endpoint and the `HEALTHCHECK` in the Dockerfile are suitable readiness probes. The server starts in degraded state when the model fails to load — failed pods are automatically removed from rotation.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Dataset

AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

## License

MIT — see [LICENSE](LICENSE).
