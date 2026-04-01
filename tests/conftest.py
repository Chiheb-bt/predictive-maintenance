"""
Shared pytest fixtures.

The `tiny_pipeline` fixture trains a minimal but real sklearn Pipeline on
200 synthetic rows — enough to exercise the full inference stack without
mocking the entire model. Training takes ~0.5s on a laptop.

This replaces patching the module-level _pipeline for predict() tests,
which was unreliable and masked real issues in the predict() code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.core.preprocessing import build_preprocessor


@pytest.fixture(scope="session")
def tiny_pipeline() -> Pipeline:
    """
    A real (tiny) sklearn Pipeline trained on synthetic data.

    Uses cv=3 and n_estimators=10 to keep runtime under half a second.
    The pipeline structure is identical to the production model — same
    ColumnTransformer, same CalibratedClassifierCV wrapper — so tests
    exercise the real code paths.
    """
    rng = np.random.default_rng(0)
    n   = 200

    X = pd.DataFrame({
        "Air temperature [K]":     rng.uniform(295, 305, n),
        "Process temperature [K]": rng.uniform(305, 315, n),
        "Rotational speed [rpm]":  rng.integers(1200, 2800, n).astype(float),
        "Torque [Nm]":             rng.uniform(10, 70, n),
        "Tool wear [min]":         rng.integers(0, 250, n).astype(float),
        "Type":                    rng.choice(["L", "M", "H"], n),
    })
    # Use 15% failure rate so every CV fold contains both classes.
    y = pd.Series((rng.random(n) < 0.15).astype(int))

    base       = RandomForestClassifier(n_estimators=10, random_state=42)
    classifier = CalibratedClassifierCV(base, cv=3, method="isotonic")
    pipeline   = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   classifier),
    ])
    pipeline.fit(X, y)
    return pipeline
