"""
Advanced training pipeline: XGBoost + Optuna + MLflow.

Runs an Optuna study to find optimal XGBoost hyperparameters via stratified
5-fold CV, retrains the winner on the full training set, calibrates
probabilities with isotonic regression, and logs everything to MLflow.
The output model.pkl is a drop-in replacement — inference.py accepts both
the standard RandomForest pipeline and this XGBoost pipeline without changes.

Install advanced deps first:
    pip install -r requirements-advanced.txt

Usage:
    python src/models/train_advanced.py
    python src/models/train_advanced.py --data Data/maintenance.csv --trials 100
    python src/models/train_advanced.py --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    raise ImportError(
        "optuna is required for advanced training.\n"
        "Install with: pip install -r requirements-advanced.txt"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required for advanced training.\n"
        "Install with: pip install -r requirements-advanced.txt"
    ) from exc

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

from src.core.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)
from src.models.train import (
    best_threshold,
    evaluate,
    extract_feature_importance,
    load,
    prepare,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _build_xgb_pipeline(params: dict) -> Pipeline:
    clf = XGBClassifier(**params, eval_metric="logloss", random_state=42, n_jobs=-1)
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   CalibratedClassifierCV(clf, cv=5, method="isotonic")),
    ])


def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective: stratified 5-fold mean F1.

    F1 is preferred over AUC here because it directly penalises both missed
    failures (false negatives) and false alarms (false positives), which maps
    to the real cost of industrial maintenance decisions.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
    }

    pipeline = _build_xgb_pipeline(params)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores   = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        proba  = pipeline.predict_proba(X_val)[:, 1]
        thr    = best_threshold(y_val.values, proba)
        y_pred = (proba >= thr).astype(int)
        scores.append(f1_score(y_val, y_pred, zero_division=0))

    return float(np.mean(scores))


def run(
    data_path: str,
    output_path: str,
    n_trials: int = 50,
    experiment_name: str = "predictive-maintenance",
    use_mlflow: bool = True,
) -> None:
    df   = load(data_path)
    X, y = prepare(df)

    X_train, X_tmp,  y_train, y_tmp  = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42,
    )

    mlflow_run = None
    if use_mlflow and _MLFLOW:
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run(run_name=f"xgboost-optuna-{n_trials}trials")
    elif use_mlflow and not _MLFLOW:
        log.warning("mlflow not installed — tracking disabled.")

    try:
        log.info("Starting Optuna search: %d trials…", n_trials)
        study = optuna.create_study(direction="maximize", study_name="pm-xgb")
        study.optimize(
            lambda trial: _objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        log.info("Best CV F1 : %.4f", study.best_value)
        log.info("Best params: %s", best_params)

        pipeline = _build_xgb_pipeline(best_params)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_val, y_val, X_test, y_test)
        factors = extract_feature_importance(pipeline)

        if mlflow_run:
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_metrics({
                "f1":             metrics["f1"],
                "roc_auc":        metrics["roc_auc"],
                "threshold":      metrics["threshold"],
                "optuna_best_cv": round(study.best_value, 4),
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out)
        log.info("Pipeline saved → %s", out.resolve())

        Path("artifacts").mkdir(exist_ok=True)
        meta = {
            "model_type":         "XGBoost (calibrated, Optuna-tuned)",
            "feature_columns":    NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "best_params":        best_params,
            "threshold":          metrics["threshold"],
            "metrics":            {"f1": metrics["f1"], "roc_auc": metrics["roc_auc"]},
            "optuna_best_cv_f1":  round(study.best_value, 4),
            "n_trials":           n_trials,
            "feature_importance": factors,
        }
        Path("artifacts/meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Metadata saved → artifacts/meta.json")

    finally:
        if mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced training with XGBoost + Optuna")
    parser.add_argument("--data",       default="Data/maintenance.csv")
    parser.add_argument("--output",     default="model.pkl")
    parser.add_argument("--trials",     type=int, default=50)
    parser.add_argument("--experiment", default="predictive-maintenance")
    parser.add_argument("--no-mlflow",  action="store_true")
    args = parser.parse_args()

    run(
        data_path=args.data,
        output_path=args.output,
        n_trials=args.trials,
        experiment_name=args.experiment,
        use_mlflow=not args.no_mlflow,
    )
