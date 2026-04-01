"""
Feature schema and preprocessing pipeline for the Predictive Maintenance system.

Keeping the ColumnTransformer here (rather than inline in train.py) means training
and serving are always using the same transformations — no chance of skew creeping in
between a notebook experiment and the production API.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES: list[str] = ["Type"]

# H is the reference class dropped by OneHotEncoder, leaving Type_L and Type_M
# as the two encoded columns. Drop="first" avoids multicollinearity.
TYPE_CATEGORIES: list[list[str]] = [["H", "L", "M"]]

# Pipeline-prefixed names → human-readable labels shown in API responses.
FEATURE_LABELS: dict[str, str] = {
    "num__Air temperature [K]":     "Air Temperature",
    "num__Process temperature [K]": "Process Temperature",
    "num__Rotational speed [rpm]":  "Rotational Speed",
    "num__Torque [Nm]":             "Torque",
    "num__Tool wear [min]":         "Tool Wear",
    "cat__Type_L":                  "Machine Type (Low)",
    "cat__Type_M":                  "Machine Type (Medium)",
}

# API field aliases → canonical column names matching the training dataset.
INPUT_ALIASES: dict[str, str] = {
    "Air_temperature":     "Air temperature [K]",
    "Process_temperature": "Process temperature [K]",
    "Rotational_speed":    "Rotational speed [rpm]",
    "Torque":              "Torque [Nm]",
    "Tool_wear":           "Tool wear [min]",
}

# Valid ranges keyed by canonical column name. Used for input validation
# before hitting the model — lets us return a 422 with a clear message
# instead of a cryptic scaler output for wild input values.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "Air temperature [K]":     (200.0, 400.0),
    "Process temperature [K]": (200.0, 400.0),
    "Rotational speed [rpm]":  (0.0,   10_000.0),
    "Torque [Nm]":             (0.0,   500.0),
    "Tool wear [min]":         (0.0,   500.0),
}

VALID_TYPES: frozenset[str] = frozenset({"L", "M", "H"})


def build_preprocessor() -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer ready to be composed into a Pipeline.

    Numeric features get StandardScaler (zero mean, unit variance).
    Type is one-hot encoded with H as the dropped reference category.
    The transformer is intentionally stateless here — fit() is always
    called inside a Pipeline so joblib serialises everything together.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    categories=TYPE_CATEGORIES,
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
