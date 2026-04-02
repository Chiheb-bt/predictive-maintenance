"""
FastAPI application with embedded Gradio UI.

Endpoints:
    GET  /               — health check + model metadata
    POST /predict        — single structured inference result
    POST /predict/batch  — bulk inference (1–100 readings, returns per-row results + summary)
    POST /drift          — distribution drift check vs training data (10–500 readings)
    GET  /ui             — Gradio interactive demo
    GET  /docs           — Swagger / OpenAPI documentation (auto-generated)
    GET  /metrics        — Prometheus metrics (via prometheus-fastapi-instrumentator)
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections import Counter as _Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import structlog
from fastapi import BackgroundTasks, Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter as PrometheusCounter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

from src.core.database import log_prediction
from src.core.preprocessing import NUMERIC_FEATURES
from src.serving.inference import (
    get_load_error,
    get_model_path,
    load_model,
    model_is_ready,
    predict,
    predict_batch,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
if "*" in _ALLOWED_ORIGINS:
    log.warning(
        "CORS is open to ALL origins (ALLOWED_ORIGINS contains '*') — "
        "this is unsafe in production."
    )

# ---------------------------------------------------------------------------
# API key auth — secrets.compare_digest prevents timing attacks.
# A plain string comparison (==) leaks timing information that lets an
# attacker guess the key one character at a time by measuring response times.
# compare_digest takes constant time regardless of where strings first differ.
# ---------------------------------------------------------------------------
_API_KEY = os.getenv("API_KEY", "").strip()
if not _API_KEY:
    log.warning(
        "API_KEY is not set — the /predict endpoint is publicly accessible. "
        "Set the API_KEY environment variable to enable key-based authentication."
    )


async def _require_api_key(x_api_key: str = Header(default="")) -> None:
    """
    FastAPI dependency that enforces constant-time API key authentication.

    Disabled when API_KEY env var is unset — suitable for local dev and demos.
    Clients send: X-Api-Key: <your-key>
    """
    if _API_KEY and not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _get_audit_meta(request: Request) -> tuple[str | None, str | None]:
    """Extract client IP and User-Agent for the audit trail."""
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    return ip, ua


# ---------------------------------------------------------------------------
# Business metrics — risk-level prediction counts.
# HTTP-level metrics (latency, error rate, request rate) are provided
# automatically by prometheus-fastapi-instrumentator below.
# ---------------------------------------------------------------------------
_PREDICTION_COUNTER = PrometheusCounter(
    "pm_predictions_total",
    "Total predictions labelled by risk level",
    ["risk_level"],
)

# Instrumentator handles rate / errors / duration for every endpoint.
# Exclude /metrics itself and the Gradio UI from being tracked.
_instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/ui"],
    inprogress_name="pm_inprogress_requests",
    inprogress_labels=True,
)


# ---------------------------------------------------------------------------
# Training stats — loaded at startup for the /drift endpoint
# ---------------------------------------------------------------------------
def _load_training_stats() -> dict:  # type: ignore[type-arg]
    """
    Read per-feature distribution statistics from artifacts/meta.json.

    Written by train.py as ``training_stats``. Used by /drift to detect
    when incoming sensor distributions shift away from what the model saw
    at training time. Returns an empty dict gracefully if the file is
    missing or the key is absent (model not yet trained).
    """
    candidates = [
        Path("artifacts/meta.json"),
        Path(__file__).resolve().parents[2] / "artifacts" / "meta.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text()).get("training_stats", {})  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read training_stats from %s: %s", path, exc)
    log.info(
        "artifacts/meta.json not found — /drift endpoint will return 503 until model is trained"
    )
    return {}


# ---------------------------------------------------------------------------
# Lifespan — explicit startup / shutdown lifecycle (FastAPI ≥ 0.93).
#
# Why lifespan instead of module-level side effects?
#   • Unit tests that import this module don't trigger model I/O.
#   • Startup failures are reported via the health check before traffic arrives.
#   • Any teardown (DB pool, thread executor, etc.) lives here, not in atexit.
# ---------------------------------------------------------------------------
_TRAINING_STATS: dict = {}  # type: ignore[type-arg]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    global _TRAINING_STATS
    load_model()
    _TRAINING_STATS = _load_training_stats()
    _instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)
    log.info(
        "app_startup_complete",
        model_path=str(get_model_path()),
        model_ready=model_is_ready(),
        metrics_enabled=True,
    )
    yield
    # Teardown: close any resources here (thread pools, DB connections, etc.)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "ML API that predicts industrial machine failure from real-time sensor readings.\n\n"
        "**Authentication**: set `X-Api-Key: <your-key>` on all `/predict` and "
        "`/predict/batch` requests when the server is started with an `API_KEY` env var.\n\n"
        "**Core endpoints**\n"
        "- `POST /predict`       — full structured risk assessment for one machine\n"
        "- `POST /predict/batch` — score up to 100 machines in a single request\n"
        "- `POST /drift`         — check whether recent sensor readings have drifted "
        "away from the training distribution (useful for catching sensor calibration issues)\n"
        "- `GET  /ui`            — browser-based interactive demo\n"
        "- `GET  /metrics`       — Prometheus metrics\n\n"
        "**Outputs per prediction**\n"
        "- `request_id` — UUID v4 for tracing and audit\n"
        "- Failure probability and model certainty score\n"
        "- Risk level: LOW / MEDIUM / HIGH / CRITICAL\n"
        "- Sensor-aware maintenance recommendations\n"
        "- Top 3 contributing sensor factors\n"
    ),
    version="10.0.0",
    contact={"name": "Sentinel Engineering"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Api-Key"],
)

_instrumentator.instrument(app)

_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SensorInput(BaseModel):
    """Real-time sensor readings from a single machine."""

    Type:                str   = Field(..., description="Machine quality grade — L, M, or H")
    Air_temperature:     float = Field(..., ge=200, le=400,   description="Air temperature (Kelvin)")
    Process_temperature: float = Field(..., ge=200, le=400,   description="Process temperature (Kelvin)")
    Rotational_speed:    int   = Field(..., ge=0,   le=10000, description="Rotational speed (RPM)")
    Torque:              float = Field(..., ge=0,   le=500,   description="Torque (Nm)")
    Tool_wear:           int   = Field(..., ge=0,   le=500,   description="Accumulated tool wear (minutes)")

    @field_validator("Type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"L", "M", "H"}:
            raise ValueError("Type must be 'L', 'M', or 'H'.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Type":                "M",
                "Air_temperature":     298.1,
                "Process_temperature": 308.6,
                "Rotational_speed":    1551,
                "Torque":              42.8,
                "Tool_wear":           0,
            }
        }
    }


class FeatureFactor(BaseModel):
    rank:       int   = Field(..., description="1 = most influential")
    feature:    str   = Field(..., description="Human-readable sensor name")
    importance: float = Field(..., description="Relative importance (0–1)")


class PredictionResponse(BaseModel):
    request_id:      str                 = Field(..., description="UUID v4 for tracing and audit")
    prediction:      int                 = Field(..., description="0 = no failure  |  1 = failure predicted")
    probability:     float               = Field(..., description="Failure probability (0.00–1.00)")
    confidence:      float               = Field(..., description="Model certainty, distance from decision boundary (0–1)")
    risk_level:      str                 = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    status:          str                 = Field(..., description="Plain-English verdict")
    recommendations: list[str]           = Field(..., description="Ordered maintenance actions")
    top_factors:     list[FeatureFactor] = Field(..., description="Top 3 contributing sensors")


class BatchItemResult(BaseModel):
    """Result for a single row in a batch request. success=False rows include only error."""
    success:         bool            = Field(..., description="Whether inference succeeded for this row")
    error:           str | None   = Field(None, description="Validation error message, if success=False")
    request_id:      str | None   = Field(None, description="UUID v4 for tracing")
    prediction:      int | None   = Field(None, description="0 = no failure | 1 = failure predicted")
    probability:     float | None = Field(None, description="Failure probability (0.00–1.00)")
    risk_level:      str | None   = Field(None, description="LOW | MEDIUM | HIGH | CRITICAL")
    status:          str | None   = Field(None, description="Plain-English verdict")


class BatchPredictionResponse(BaseModel):
    n_requested: int                  = Field(..., description="Number of readings submitted")
    n_succeeded: int                  = Field(..., description="Rows that returned a prediction")
    n_failed:    int                  = Field(..., description="Rows that failed validation")
    summary:     dict                 = Field(..., description="Count of predictions by risk level")  # type: ignore[type-arg]
    predictions: list[BatchItemResult] = Field(..., description="Per-row results, same order as input")


class DriftRequest(BaseModel):
    """Batch of recent sensor readings to check against the training distribution."""
    readings: list[SensorInput] = Field(
        ...,
        min_length=10,
        max_length=500,
        description="10–500 recent sensor readings from the fleet",
    )


class FeatureDrift(BaseModel):
    feature:        str   = Field(..., description="Sensor / feature name")
    training_mean:  float = Field(..., description="Mean from training data")
    incoming_mean:  float = Field(..., description="Mean from the submitted readings")
    z_score:        float = Field(..., description="(incoming_mean − training_mean) / training_std")
    status:         str   = Field(..., description="NORMAL | WARNING | DRIFT")


class DriftResponse(BaseModel):
    n_samples:      int                = Field(..., description="Number of readings analysed")
    overall_status: str                = Field(..., description="NORMAL | WARNING | DRIFT")
    features:       list[FeatureDrift] = Field(..., description="Per-feature drift analysis")
    assessed_at:    str                = Field(..., description="ISO-8601 UTC timestamp")
    note:           str                = Field(..., description="Plain-English summary")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Health check")
def health() -> dict:  # type: ignore[type-arg]
    """
    Returns service status, version, and uptime.
    Reports 'degraded' if the model failed to load at startup.
    Suitable as a Kubernetes readiness probe — no auth required.
    """
    ready = model_is_ready()
    model_path = get_model_path()
    return {
        "status":       "ok" if ready else "degraded",
        "service":      "Predictive Maintenance API",
        "version":      "2.2.0",
        "model":        str(model_path.name) if model_path else None,
        "model_loaded": ready,
        "model_error":  get_load_error(),
        "uptime_s":     round(time.time() - _START_TIME, 1),
        "auth_enabled": bool(_API_KEY),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict machine failure",
    dependencies=[Depends(_require_api_key)],
)
def predict_failure(
    data: SensorInput,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:  # type: ignore[type-arg]
    """
    Accept sensor readings and return a full structured risk assessment.

    Requires `X-Api-Key` header when the server is started with `API_KEY` set.

    Returns HTTP 503 if the model failed to load at startup.
    Returns HTTP 401 if the API key is wrong or missing (when auth is enabled).
    """
    if not model_is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded — server is degraded. Reason: {get_load_error()}",
        )

    try:
        result = predict(data.model_dump())
    except ValueError as exc:
        log.warning("inference_validation_failed", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.error("inference_server_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    _PREDICTION_COUNTER.labels(risk_level=result["risk_level"]).inc()

    # Log to Audit Trail (Async)
    client_ip, user_agent = _get_audit_meta(request)
    background_tasks.add_task(
        log_prediction,
        request_id=result["request_id"],
        machine_type=data.Type,
        input_data=data.model_dump(),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        client_ip=client_ip,
        user_agent=user_agent,
    )

    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch machine failure prediction",
    dependencies=[Depends(_require_api_key)],
)
def predict_batch_endpoint(
    readings: list[SensorInput] = Body(..., min_length=1, max_length=100),
) -> BatchPredictionResponse:
    """
    Score 1–100 machines in a single request and return a summary alongside per-row results.

    Ideal for dashboards that need to update risk scores across an entire fleet
    without making individual HTTP calls per machine. Each row is processed
    independently — a validation failure in one row doesn't affect the others.

    Returns HTTP 503 if the model failed to load at startup.
    """
    if not model_is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded — server is degraded. Reason: {get_load_error()}",
        )

    try:
        raw_results = predict_batch([r.model_dump() for r in readings])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    items     = [BatchItemResult(**r) for r in raw_results]
    succeeded = [i for i in items if i.success]
    failed    = [i for i in items if not i.success]

    risk_counts: dict = dict(_Counter(i.risk_level for i in succeeded if i.risk_level))  # type: ignore[type-arg]

    return BatchPredictionResponse(
        n_requested=len(readings),
        n_succeeded=len(succeeded),
        n_failed=len(failed),
        summary=risk_counts,
        predictions=items,
    )


@app.post(
    "/drift",
    response_model=DriftResponse,
    tags=["Observability"],
    summary="Distribution drift detection",
)
def check_drift(payload: DriftRequest) -> DriftResponse:
    """
    Detect whether recent sensor readings have drifted from the training distribution.

    For each numeric feature, computes a Z-score:

        z = (incoming_mean − training_mean) / training_std

    and classifies the feature as:
    - **NORMAL**  — |z| < 2.0  (within two standard deviations)
    - **WARNING** — 2.0 ≤ |z| < 3.0  (worth monitoring)
    - **DRIFT**   — |z| ≥ 3.0  (significant shift — retrain or investigate sensors)

    Common causes of drift: sensor recalibration, new machine models added to the fleet,
    or operating condition changes (seasonal temperature, new product type).

    No auth required — monitoring systems need unauthenticated access.
    Submit at least 10 readings for statistically meaningful results.
    """
    if not _TRAINING_STATS:
        raise HTTPException(
            status_code=503,
            detail=(
                "Training statistics not available. "
                "Run 'python src/models/train.py' to generate artifacts/meta.json."
            ),
        )

    import pandas as pd  # noqa: PLC0415

    from src.core.preprocessing import INPUT_ALIASES  # noqa: PLC0415

    readings_dicts = [r.model_dump() for r in payload.readings]
    rows = []
    for d in readings_dicts:
        norm = {INPUT_ALIASES.get(k, k): v for k, v in d.items()}
        rows.append({col: float(norm[col]) for col in NUMERIC_FEATURES if col in norm})

    df = pd.DataFrame(rows)
    feature_results: list[FeatureDrift] = []
    overall_statuses: list[str] = []

    for col in NUMERIC_FEATURES:
        if col not in _TRAINING_STATS or col not in df.columns:
            continue
        ts            = _TRAINING_STATS[col]
        training_mean = ts["mean"]
        training_std  = ts["std"] or 1.0
        incoming_mean = float(df[col].mean())
        z_score       = round((incoming_mean - training_mean) / training_std, 3)
        abs_z         = abs(z_score)

        if abs_z < 2.0:
            status = "NORMAL"
        elif abs_z < 3.0:
            status = "WARNING"
        else:
            status = "DRIFT"

        overall_statuses.append(status)
        feature_results.append(FeatureDrift(
            feature=col,
            training_mean=round(training_mean, 4),
            incoming_mean=round(incoming_mean, 4),
            z_score=z_score,
            status=status,
        ))

    if any(s == "DRIFT" for s in overall_statuses):
        overall = "DRIFT"
        note    = "One or more features show significant drift (|z| ≥ 3). Investigate sensors or consider retraining."
    elif any(s == "WARNING" for s in overall_statuses):
        overall = "WARNING"
        note    = "Minor drift detected in some features. Monitor closely over the next collection window."
    else:
        overall = "NORMAL"
        note    = "All features are within expected distribution. Model predictions should be reliable."

    return DriftResponse(
        n_samples=len(payload.readings),
        overall_status=overall,
        features=feature_results,
        assessed_at=datetime.now(UTC).isoformat(),
        note=note,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_CSS = """
:root {
    --risk-low-bg:       #f0fdf4;  --risk-low-border:   #86efac;  --risk-low-text:     #166534;
    --risk-med-bg:       #fffbeb;  --risk-med-border:   #fcd34d;  --risk-med-text:     #92400e;
    --risk-high-bg:      #fff7ed;  --risk-high-border:  #fb923c;  --risk-high-text:    #9a3412;
    --risk-crit-bg:      #fff1f2;  --risk-crit-border:  #fda4af;  --risk-crit-text:    #9f1239;
    --risk-idle-bg:      #f8fafc;  --risk-idle-border:  #e2e8f0;  --risk-idle-text:    #94a3b8;
    --bar-track:         #e2e8f0;
    --accent:            #6366f1;
}
.gradio-container { font-family: 'Inter', 'Segoe UI', sans-serif !important; max-width: 1100px !important; margin: 0 auto !important; }
.risk-card { border-radius: 12px; padding: 1.2rem 1.5rem; font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; }
.risk-low      { background: var(--risk-low-bg);  border: 2px solid var(--risk-low-border);  color: var(--risk-low-text);  }
.risk-medium   { background: var(--risk-med-bg);  border: 2px solid var(--risk-med-border);  color: var(--risk-med-text);  }
.risk-high     { background: var(--risk-high-bg); border: 2px solid var(--risk-high-border); color: var(--risk-high-text); }
.risk-critical { background: var(--risk-crit-bg); border: 2px solid var(--risk-crit-border); color: var(--risk-crit-text); }
.risk-idle     { background: var(--risk-idle-bg); border: 2px solid var(--risk-idle-border); color: var(--risk-idle-text); }
.risk-meta     { margin-top: 8px; font-size: .85rem; font-weight: 400; opacity: .85; }
.req-id        { margin-top: 6px; font-size: .75rem; font-weight: 400; opacity: .6; font-family: monospace; }
.bar-track { background: var(--bar-track); border-radius: 99px; height: 10px; overflow: hidden; margin: 6px 0 14px; }
.bar-fill  { height: 100%; border-radius: 99px; transition: width .4s ease; }
.factors-table    { width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: .5rem; }
.factors-table th { text-align: left; color: #64748b; font-weight: 600; padding: 4px 8px; border-bottom: 1px solid #e2e8f0; }
.factors-table td { padding: 6px 8px; border-bottom: 1px solid #f1f5f9; }
.imp-bar-wrap { background: #f1f5f9; border-radius: 99px; height: 8px; width: 120px; overflow: hidden; display: inline-block; vertical-align: middle; }
.imp-bar-fill { height: 100%; background: var(--accent); border-radius: 99px; }
.rec-list          { list-style: none; padding: 0; margin: .5rem 0 0; }
.rec-list li       { padding: 5px 0 5px 1.4rem; position: relative; font-size: .9rem; line-height: 1.5; border-bottom: 1px solid #f1f5f9; }
.rec-list li::before { content: "->"; position: absolute; left: 0; color: var(--accent); font-weight: 700; }
"""

_RISK_META: dict[str, tuple[str, str, str]] = {
    "LOW":      ("risk-low",      "Low Risk",      "#22c55e"),
    "MEDIUM":   ("risk-medium",   "Medium Risk",   "#f59e0b"),
    "HIGH":     ("risk-high",     "High Risk",     "#f97316"),
    "CRITICAL": ("risk-critical", "Critical Risk", "#ef4444"),
}


def _result_html(result: dict) -> str:  # type: ignore[type-arg]
    prob       = result["probability"]
    conf       = result["confidence"]
    risk       = result["risk_level"]
    status     = result["status"]
    recs       = result["recommendations"]
    factors    = result["top_factors"]
    request_id = result["request_id"]

    cls, label, colour = _RISK_META.get(risk, ("risk-idle", risk, "#94a3b8"))
    bar_pct  = int(prob * 100)
    conf_pct = int(conf * 100)

    factors_rows = "".join(
        f"""<tr>
              <td>#{f["rank"]}</td>
              <td>{f["feature"]}</td>
              <td>
                <span class="imp-bar-wrap">
                  <span class="imp-bar-fill" style="width:{int(f['importance']*100)}%"></span>
                </span>
                &nbsp;{f['importance']:.2%}
              </td>
            </tr>"""
        for f in factors
    )
    rec_items = "".join(f"<li>{r}</li>" for r in recs)

    return f"""
<div class="risk-card {cls}">
  {label} — {status}
  <div class="risk-meta">
    Failure probability: <strong>{prob:.1%}</strong> &nbsp;|&nbsp;
    Model certainty: <strong>{conf_pct}%</strong>
  </div>
  <div class="req-id">Request ID: {request_id}</div>
</div>

<p><strong>Failure probability</strong></p>
<div class="bar-track">
  <div class="bar-fill" style="width:{bar_pct}%;background:{colour}"></div>
</div>

<p><strong>Contributing factors</strong></p>
<table class="factors-table">
  <thead><tr><th>Rank</th><th>Sensor</th><th>Influence</th></tr></thead>
  <tbody>{factors_rows}</tbody>
</table>

<p style="margin-top:1rem"><strong>Recommended actions</strong></p>
<ul class="rec-list">{rec_items}</ul>
"""


def _gradio_predict(
    machine_type: str,
    air_temp: float,
    process_temp: float,
    rpm: int,
    torque: float,
    tool_wear: int,
) -> str:
    if not model_is_ready():
        return f"<p style='color:red'>Model not loaded: {get_load_error()}</p>"
    try:
        result = predict({
            "Type":                machine_type,
            "Air_temperature":     air_temp,
            "Process_temperature": process_temp,
            "Rotational_speed":    rpm,
            "Torque":              torque,
            "Tool_wear":           tool_wear,
        })
        return _result_html(result)
    except Exception as exc:
        return f"<p style='color:red'>Error: {exc}</p>"


def _build_gradio_app() -> gr.Blocks:
    with gr.Blocks(css=_CSS, title="Predictive Maintenance") as demo:
        gr.Markdown("## Predictive Maintenance — Interactive Demo")
        gr.Markdown(
            "Enter sensor readings and click **Predict** to get a risk assessment. "
            "Default values are typical operating conditions for a medium-grade machine."
        )
        with gr.Row():
            with gr.Column(scale=1):
                machine_type = gr.Dropdown(["L", "M", "H"], value="M", label="Machine Type")
                air_temp     = gr.Slider(200, 400, value=298.1, step=0.1, label="Air Temperature (K)")
                process_temp = gr.Slider(200, 400, value=308.6, step=0.1, label="Process Temperature (K)")
                rpm          = gr.Slider(0, 10000, value=1551,  step=1,   label="Rotational Speed (RPM)")
                torque       = gr.Slider(0, 500,   value=42.8,  step=0.1, label="Torque (Nm)")
                tool_wear    = gr.Slider(0, 500,   value=0,     step=1,   label="Tool Wear (min)")
                btn          = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                output = gr.HTML(
                    value="<div class='risk-card risk-idle'>Submit a reading to see the risk assessment.</div>"
                )

        btn.click(
            fn=_gradio_predict,
            inputs=[machine_type, air_temp, process_temp, rpm, torque, tool_wear],
            outputs=output,
        )

    return demo


_gradio_app = _build_gradio_app()
app = gr.mount_gradio_app(app, _gradio_app, path="/ui")
