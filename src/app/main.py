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
from typing import Any

import gradio as gr
import plotly.graph_objects as go
import structlog
from fastapi import BackgroundTasks, Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter as PrometheusCounter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

from src.core.database import get_recent_audit_logs, init_db, log_prediction
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
        "version":      "10.0.0",
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
# Gradio "Command Center" — The Ultimate Expert Dashboard
# ---------------------------------------------------------------------------

def _create_risk_gauge(probability: float, risk_level: str) -> go.Figure:
    """Create a high-fidelity industrial risk gauge."""
    colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#7f1d1d"}
    color = colors.get(risk_level, "#3b82f6")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "FAILURE PROBABILITY", "font": {"size": 20, "color": "#1f2937"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4b5563"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.1)"},
                {"range": [30, 70], "color": "rgba(245, 158, 11, 0.1)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.1)"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _create_factor_chart(factors: list[dict[str, Any]]) -> go.Figure:
    """Create a horizontal bar chart showing sensor influence."""
    names = [f["feature"] for f in reversed(factors)]
    values = [f["importance"] * 100 for f in reversed(factors)]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color="#3b82f6", line=dict(color="#2563eb", width=1))
    ))
    fig.update_layout(
        title={"text": "KEY SENSOR INFLUENCE", "font": {"size": 16}},
        xaxis_title="Influence Score (%)",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


async def _get_history_table() -> list[list[Any]]:
    """Fetch recent predictions from the audit trail."""
    try:
        logs = await get_recent_audit_logs(limit=8)
        return [
            [l["timestamp"][:19], l["machine_type"], l["risk_level"], l["probability"], l["client_ip"] or "-"]
            for l in logs
        ]
    except Exception:
        return []


async def _ui_predict_handler(
    m_type: str,
    air_t: float,
    proc_t: float,
    rpm: int,
    torque: float,
    tool_wear: int,
) -> tuple[go.Figure | None, go.Figure | None, str, list[list[Any]]]:
    """Bridge the UI to the inference engine with live audit logging."""
    payload = {
        "Type": m_type,
        "Air_temperature": air_t,
        "Process_temperature": proc_t,
        "Rotational_speed": rpm,
        "Torque": torque,
        "Tool_wear": tool_wear,
    }

    try:
        # We don't have BackgroundTasks/Request here in Gradio, 
        # so we call predict directly.
        res = predict(payload)
        
        # Log to DB manually for UI predictions
        try:
            await log_prediction(
                request_id=res["request_id"],
                machine_type=m_type,
                input_data=payload,
                prediction=res["prediction"],
                probability=res["probability"],
                risk_level=res["risk_level"],
                client_ip="Grad-UI-User",
                user_agent="Gradio Browser"
            )
        except Exception as e:
             log.error("ui_db_log_failed", error=str(e))

        gauge = _create_risk_gauge(res["probability"], res["risk_level"])
        factors = _create_factor_chart(res["top_factors"])
        history = await _get_history_table()
        
        recs = "\n".join([f"• {r}" for r in res["recommendations"]])
        
        return gauge, factors, recs, history

    except Exception as e:
        return None, None, f"Error: {e}", []


def _build_gradio_ui() -> gr.Blocks:
    """The Ultimate Sentinel Command Center."""
    custom_css = """
    .gradio-container { background-color: #f9fafb; font-family: 'Inter', sans-serif; }
    .risk-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; background: white; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .header-center { text-align: center; margin-bottom: 2rem; }
    .logo-font { font-size: 2.5rem; font-weight: 800; color: #111827; letter-spacing: -0.025em; }
    """

    with gr.Blocks(title="Sentinel Command Center") as demo:
        with gr.Column(elem_classes="header-center"):
            gr.Markdown(
                "<div class='logo-font'>SENTINEL <span style='color:#3b82f6'>v10.0</span></div>"
                "<p style='color:#6b7280; font-size:1.1rem;'>Industrial Intelligence Command Center • Real-time Predictive Maintenance</p>"
            )

        with gr.Row():
            # LEFT: TELEMETRY CONTROL
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ TELEMETRY CONTROL")
                with gr.Column(elem_classes="risk-card"):
                    m_type    = gr.Dropdown(choices=["L", "M", "H"], value="M", label="Machine Type")
                    air_t     = gr.Slider(290, 310, 298, label="Air Temp [K]")
                    proc_t    = gr.Slider(300, 320, 308, label="Process Temp [K]")
                    rpm       = gr.Slider(1000, 3000, 1550, label="Rotational Speed [rpm]")
                    torque    = gr.Slider(10, 80, 45, label="Torque [Nm]")
                    wear      = gr.Slider(0, 250, 0, label="Tool Wear [min]")
                    btn       = gr.Button("RUN INFERENCE", variant="primary")

            # RIGHT: EXPERT INSIGHTS
            with gr.Column(scale=2):
                gr.Markdown("### 🧠 EXPERT INSIGHTS")
                with gr.Row():
                    gauge_out = gr.Plot(label="Risk Assessment")
                    factors_out = gr.Plot(label="Signal Analysis")
                
                with gr.Column(elem_classes="risk-card"):
                    recs_out = gr.Markdown("### 🛠️ RECOMMENDATIONS\n*Awaiting telemetry...*")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📜 LIVE AUDIT FEED")
                history_out = gr.Dataframe(
                    headers=["TIMESTAMP", "MACHINE", "RISK", "PROB", "CLIENT"],
                    datatype=["str", "str", "str", "number", "str"],
                    value=[],
                    interactive=False
                )

        # Wire it up
        btn.click(
            fn=_ui_predict_handler,
            inputs=[m_type, air_t, proc_t, rpm, torque, wear],
            outputs=[gauge_out, factors_out, recs_out, history_out]
        )
        
        # Load initial history
        demo.load(fn=_get_history_table, outputs=history_out)

    demo._css = custom_css
    demo._theme = gr.themes.Soft()
    return demo


_gradio_app = _build_gradio_ui()
app = gr.mount_gradio_app(app, _gradio_app, path="/ui")
