"""
Training pipeline for the Predictive Maintenance model.

Trains a calibrated Random Forest on the AI4I 2020 Predictive Maintenance dataset
and writes two artefacts to disk:

    model.pkl            — full sklearn Pipeline (preprocessor + calibrated RF)
    artifacts/meta.json  — threshold, metrics, feature importances, best params

Data split strategy:
    train (60%) / validation (20%) / test (20%)

    Grid search CV runs entirely on the training set.
    Threshold selection uses the validation set — not the test set — to avoid
    data leakage. Final metrics are reported on the held-out test set only.

Usage:
    python src/models/train.py
    python src/models/train.py --data Data/maintenance.csv --output model.pkl
    python src/models/train.py --no-search   # skip grid search, use defaults
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.core.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
    build_preprocessor,
)

log = structlog.get_logger(__name__)

TARGET_COLUMN = "Machine failure"

# UDI and Product ID are identifiers, not features.
# TWF, HDF, PWF, OSF, RNF are sub-failure type labels — they're outcomes,
# not sensor readings, so including them would leak the target.
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_PARAM_GRID = {
    "classifier__estimator__n_estimators":      [100, 200, 300],
    "classifier__estimator__max_depth":         [6, 10, 15],
    "classifier__estimator__min_samples_split": [2, 5, 10],
}

_DEFAULT_PARAMS = {
    "n_estimators":      200,
    "max_depth":         10,
    "min_samples_split": 5,
}


def load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_csv(p)
    log.info("dataset_loaded", rows=len(df), cols=len(df.columns))
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop non-feature columns, impute missing values, return (X, y)."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    missing = df[RAW_FEATURES].isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        log.warning("missing_values_detected", columns=missing.to_dict())

    # Impute numeric features with the training-set median.
    # Filling with 0 is physically wrong (0 Kelvin, 0 RPM are not plausible
    # sensor readings) and would silently corrupt model inputs.
    numeric_medians = df[NUMERIC_FEATURES].median()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(numeric_medians)

    # Categorical fallback: most frequent category.
    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any():
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    y = df[TARGET_COLUMN].astype(int)

    missing_cols = [c for c in RAW_FEATURES if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns: {missing_cols}")

    return df[RAW_FEATURES], y


def best_threshold(y_true: np.ndarray[Any, Any], proba: np.ndarray[Any, Any]) -> float:
    """Return the cut-off that maximises F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def extract_feature_importance(pipeline: Pipeline) -> list[dict[str, Any]]:
    """Average importances across calibrated estimator folds."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())

    importances = np.mean([
        est.estimator.feature_importances_
        for est in classifier.calibrated_classifiers_
    ], axis=0)

    total = importances.sum() or 1.0
    ranked = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {
            "feature":    FEATURE_LABELS.get(name, name),
            "importance": round(float(imp / total), 4),
        }
        for name, imp in ranked
    ]


def evaluate(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Select threshold on validation set, report final metrics on test set.

    Separating threshold selection (val) from final evaluation (test)
    gives honest estimates — using the test set for both would overfit
    the threshold to noise in that split.
    """
    val_proba  = pipeline.predict_proba(X_val)[:, 1]
    threshold  = best_threshold(y_val.to_numpy(), val_proba)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred     = (test_proba >= threshold).astype(int)

    f1        = round(f1_score(y_test, y_pred, zero_division=0), 4)
    precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, zero_division=0), 4)
    roc_auc   = round(roc_auc_score(y_test, test_proba), 4)

    # Naive baseline: always predict the majority class.
    # Shows how much the model actually adds beyond a zero-effort guess.
    naive_f1 = round(f1_score(y_test, [0] * len(y_test), zero_division=0), 4)

    log.info(
        "test_metrics",
        f1=f1,
        precision=precision,
        recall=recall,
        roc_auc=roc_auc,
        threshold=threshold,
    )
    log.info("baseline_comparison", naive_f1=naive_f1)
    log.info("classification_report", report=classification_report(y_test, y_pred, zero_division=0))

    return {
        "f1":                f1,
        "precision":         precision,
        "recall":            recall,
        "roc_auc":           roc_auc,
        "threshold":         round(threshold, 4),
        "naive_baseline_f1": naive_f1,
    }


def _base_pipeline(n_estimators: int, max_depth: int, min_samples_split: int) -> Pipeline:
    base       = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    classifier = CalibratedClassifierCV(base, cv=5, method="isotonic")
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   classifier),
    ])


def search_best_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict[str, Any]]:
    """
    Stratified 3-fold grid search over key RandomForest hyperparameters.
    Returns the best fitted pipeline and the winning parameter set.
    F1 is the scoring metric — appropriate for this imbalanced failure dataset.
    """
    base       = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
    classifier = CalibratedClassifierCV(base, cv=5, method="isotonic")
    pipeline   = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   classifier),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=_PARAM_GRID,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_params = {
        k.replace("classifier__estimator__", ""): v
        for k, v in search.best_params_.items()
    }
    log.info(
        "grid_search_complete",
        best_cv_f1=round(float(search.best_score_), 4),
        best_params=best_params,
    )
    return search.best_estimator_, best_params


def _compute_training_stats(X_train: pd.DataFrame) -> dict[str, Any]:
    """
    Compute per-feature distribution statistics from the training set.

    Written to meta.json and consumed by the ``/drift`` API endpoint at
    runtime, where it detects incoming sensor distributions that have shifted
    away from what the model was trained on.  Common causes: sensor calibration
    drift, operating-condition change, or new machine type being added to the fleet.
    """
    stats = {}
    for col in NUMERIC_FEATURES:
        stats[col] = {
            "mean": round(float(X_train[col].mean()),            4),
            "std":  round(float(X_train[col].std()),             4),
            "min":  round(float(X_train[col].min()),             4),
            "max":  round(float(X_train[col].max()),             4),
            "p25":  round(float(X_train[col].quantile(0.25)),    4),
            "p75":  round(float(X_train[col].quantile(0.75)),    4),
        }
    return stats


def run(data_path: str, output_path: str, run_search: bool = True) -> None:
    # Initialize MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Sentinel-Industrial-AI")

    df   = load(data_path)
    X, y = prepare(df)

    failure_rate = float(y.mean())
    log.info("target_distribution", failure_rate=round(failure_rate, 4))

    # 60 / 20 / 20 split — stratified to preserve class balance in every split.
    X_train, X_tmp,  y_train, y_tmp  = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42,
    )
    log.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    if run_search:
        log.info("Starting grid search (this takes a few minutes)…")
        pipeline, best_params = search_best_pipeline(X_train, y_train)
    else:
        log.info("Skipping grid search — using default params")
        best_params = _DEFAULT_PARAMS
        pipeline    = _base_pipeline(**best_params)
        pipeline.fit(X_train, y_train)

    metrics        = evaluate(pipeline, X_val, y_val, X_test, y_test)
    factors        = extract_feature_importance(pipeline)
    training_stats = _compute_training_stats(X_train)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest-Calibrated")
        mlflow.log_param("run_search", run_search)

        # Log metrics (mlflow.log_metrics expects float values)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        # Save artifacts locally
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out)

        Path("artifacts").mkdir(exist_ok=True)
        meta = {
            "model_type":         "RandomForest (calibrated, isotonic)",
            "feature_columns":    NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "best_params":        best_params,
            "threshold":          metrics["threshold"],
            "metrics":            metrics,
            "feature_importance": factors,
            "training_stats":     training_stats,
        }
        meta_path = Path("artifacts/meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

        # Log to MLflow
        mlflow.log_artifact(str(meta_path))
        mlflow.sklearn.log_model(pipeline, "model")

        log.info("pipeline_saved", path=str(out.resolve()))
        log.info("metadata_saved", path=str(meta_path))
        log.info("mlflow_run_complete", experiment="Sentinel-Industrial-AI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Predictive Maintenance pipeline")
    parser.add_argument("--data",      default="Data/maintenance.csv")
    parser.add_argument("--output",    default="model.pkl")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip grid search and use default hyperparameters")
    args = parser.parse_args()
    run(args.data, args.output, run_search=not args.no_search)
