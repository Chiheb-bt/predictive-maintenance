"""
Production inference layer.

Responsibilities:
  1. Locate and load the trained sklearn Pipeline on explicit call to load_model().
     The module no longer triggers I/O at import time — callers (the FastAPI lifespan
     or the CLI) invoke load_model() when they are ready.
     If the model file is missing or corrupt, the server starts anyway —
     GET / reports a "degraded" status and POST /predict returns HTTP 503.
  2. Validate and normalise incoming sensor data.
  3. Build a DataFrame aligned to the training schema and run it through
     the pipeline (preprocessing + model in one call — no drift possible).
  4. Return a fully structured result with a UUID request_id for tracing.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.core.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    FIELD_RANGES,
    INPUT_ALIASES,
    NUMERIC_FEATURES,
    VALID_TYPES,
)
from src.core.risk_engine import (
    FeatureFactor,
    classify_risk,
    get_recommendations,
    top_factors,
)

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_pipeline: object | None = None
_model_path: Path | None = None
_load_error: str | None  = None


def _find_model() -> Path:
    """
    Resolve the model file.

    Priority: MODEL_PATH env var → project root → src/serving/ → cwd.
    """
    if env := os.getenv("MODEL_PATH"):
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f"MODEL_PATH='{env}' does not exist.")

    candidates = [
        Path(__file__).resolve().parents[2] / "model.pkl",
        Path(__file__).resolve().parent / "model.pkl",
        Path("model.pkl"),
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "model.pkl not found. Run: python src/models/train.py\n"
        "Or set MODEL_PATH=/path/to/model.pkl"
    )


def load_model() -> None:
    """
    Locate and load the trained sklearn Pipeline into module-level state.

    Called explicitly by the FastAPI lifespan context manager at server startup,
    or by the CI smoke test before calling predict(). Never called at import time —
    this ensures that importing this module has no file-system side effects,
    which keeps unit tests clean and fast.
    """
    global _pipeline, _model_path, _load_error
    try:
        _model_path = _find_model()
        _pipeline   = joblib.load(_model_path)
        log.info("Pipeline loaded from %s", _model_path)
    except Exception as exc:
        _load_error = str(exc)
        log.error("Failed to load model: %s", exc)


def model_is_ready() -> bool:
    return _pipeline is not None


def get_load_error() -> str | None:
    return _load_error


def get_model_path() -> Path | None:
    return _model_path


def _validate(data: dict) -> None:  # type: ignore[type-arg]
    """Raise ValueError with a clear message on invalid input."""
    norm = {INPUT_ALIASES.get(k, k): v for k, v in data.items()}

    machine_type = str(norm.get("Type", "")).strip().upper()
    if machine_type not in VALID_TYPES:
        raise ValueError(f"'Type' must be one of {sorted(VALID_TYPES)}, got '{machine_type}'.")

    for field, (lo, hi) in FIELD_RANGES.items():
        raw = norm.get(field)
        if raw is None:
            raise ValueError(f"Missing required field: '{field}'.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{field}' must be numeric, got '{raw}'.") from None
        if not (lo <= value <= hi):
            raise ValueError(f"'{field}' = {value} is out of range [{lo}, {hi}].")


def _build_dataframe(data: dict) -> pd.DataFrame:  # type: ignore[type-arg]
    """
    Return a single-row DataFrame with canonical column names matching training.

    API aliases (Air_temperature) are mapped to dataset column names
    (Air temperature [K]) so the pipeline's ColumnTransformer can find them.
    """
    norm  = {INPUT_ALIASES.get(k, k): v for k, v in data.items()}
    mtype = str(norm.get("Type", "")).strip().upper()
    row   = {col: float(norm[col]) for col in NUMERIC_FEATURES}
    row["Type"] = mtype
    return pd.DataFrame([row])[RAW_FEATURES]


def _extract_importances(pipeline: object) -> tuple[list[float], list[str]]:
    """
    Average feature importances across the calibrated classifier's folds.

    CalibratedClassifierCV with cv=5 wraps five base estimators. Averaging
    gives a more stable importance estimate than any single fold would.
    """
    preprocessor  = pipeline.named_steps["preprocessor"]  # type: ignore[union-attr]
    classifier    = pipeline.named_steps["classifier"]     # type: ignore[union-attr]
    feature_names = list(preprocessor.get_feature_names_out())
    importances   = np.mean([
        est.estimator.feature_importances_
        for est in classifier.calibrated_classifiers_
    ], axis=0).tolist()
    return importances, feature_names


def predict_batch(
    data_list: list[dict],  # type: ignore[type-arg]
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:  # type: ignore[type-arg]
    """
    Run inference on a list of sensor reading dicts.

    Each item in the returned list is either a full prediction result dict
    (on success) or a dict with keys ``success=False`` and ``error`` (on
    validation failure). RuntimeErrors (model not loaded) are re-raised so
    the caller can surface an appropriate HTTP 503.

    Args:
        data_list: List of sensor reading dicts (same format as predict()).
        threshold: Probability cut-off shared across all readings.

    Returns:
        List of result dicts, one per input, in the same order.
    """
    if _pipeline is None:
        raise RuntimeError(
            f"Model not loaded — server is in degraded state. Reason: {_load_error}"
        )

    results = []
    for data in data_list:
        try:
            results.append({"success": True, "error": None, **predict(data, threshold)})
        except ValueError as exc:
            results.append({"success": False, "error": str(exc), "request_id": str(uuid.uuid4())})
    return results


def predict(data: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:  # type: ignore[type-arg]
    """
    Run inference and return a structured result dict.

    Both API alias keys (Air_temperature) and canonical dataset keys
    (Air temperature [K]) are accepted — the function normalises them.

    Args:
        data:      Sensor readings as a dict.
        threshold: Probability cut-off for the positive (failure) class.

    Returns:
        Dict with keys: request_id, prediction, probability, confidence,
        risk_level, status, recommendations, top_factors.

    Raises:
        RuntimeError: Model not loaded (server in degraded state).
        ValueError:   Input data is invalid.
    """
    if _pipeline is None:
        raise RuntimeError(
            f"Model not loaded — server is in degraded state. Reason: {_load_error}"
        )

    _validate(data)

    df          = _build_dataframe(data)
    probability = float(np.round(_pipeline.predict_proba(df)[0, 1], 4))  # type: ignore[union-attr]
    prediction  = int(probability >= threshold)

    # Confidence: how far the probability sits from the decision boundary,
    # mapped to [0, 1]. A score of 1.0 means the model is certain;
    # 0.0 means the reading lands exactly on the threshold.
    confidence = round(abs(probability - 0.5) * 2, 4)

    risk_level = classify_risk(probability)

    factors:   list[FeatureFactor] = []
    top_label: str | None          = None
    try:
        importances, feature_names = _extract_importances(_pipeline)
        factors   = top_factors(importances, feature_names, FEATURE_LABELS, top_n=3)
        top_label = factors[0].label if factors else None
    except (AttributeError, KeyError, IndexError):
        log.warning("Feature importance extraction failed; top_factors will be empty.")

    recommendations = get_recommendations(risk_level, top_factor=top_label)
    status          = "Failure Likely" if prediction == 1 else "Operating Normally"
    request_id      = str(uuid.uuid4())

    log.info(
        "request_id=%s prediction=%s probability=%.4f risk=%s",
        request_id, status, probability, risk_level,
    )

    return {
        "request_id":      request_id,
        "prediction":      prediction,
        "probability":     probability,
        "confidence":      confidence,
        "risk_level":      risk_level,
        "status":          status,
        "recommendations": recommendations,
        "top_factors": [
            {"rank": f.rank, "feature": f.label, "importance": f.importance}
            for f in factors
        ],
    }
