"""
Unit tests for src/serving/inference.py.

Validation and DataFrame-building tests exercise pure functions in isolation.
predict() tests use the `tiny_pipeline` fixture from conftest.py — a real
(but tiny) sklearn Pipeline trained on synthetic data — so the inference
code path runs end-to-end without mocking the model away entirely.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MODEL_PATH", "/tmp/mock_model_does_not_exist_skip.pkl")


VALID_PAYLOAD = {
    "Type":                "M",
    "Air_temperature":     298.1,
    "Process_temperature": 308.6,
    "Rotational_speed":    1551,
    "Torque":              42.8,
    "Tool_wear":           0,
}


class TestValidate:
    def setup_method(self):
        from src.serving import inference as inf
        self._validate = inf._validate

    def test_valid_payload_does_not_raise(self):
        self._validate(VALID_PAYLOAD)

    def test_canonical_keys_accepted(self):
        canonical = {
            "Type":                    "H",
            "Air temperature [K]":     298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]":  1500,
            "Torque [Nm]":             40.0,
            "Tool wear [min]":         50,
        }
        self._validate(canonical)

    @pytest.mark.parametrize("mtype", ["L", "M", "H"])
    def test_all_valid_machine_types(self, mtype):
        self._validate({**VALID_PAYLOAD, "Type": mtype})

    @pytest.mark.parametrize("mtype", ["X", "l", "m", "h", "", "LL", "123"])
    def test_invalid_machine_type_raises(self, mtype):
        with pytest.raises(ValueError, match="Type"):
            self._validate({**VALID_PAYLOAD, "Type": mtype})

    def test_missing_field_raises(self):
        for field in ["Air_temperature", "Process_temperature",
                      "Rotational_speed", "Torque", "Tool_wear"]:
            payload = {k: v for k, v in VALID_PAYLOAD.items() if k != field}
            with pytest.raises(ValueError, match="Missing"):
                self._validate(payload)

    @pytest.mark.parametrize("field,bad_value", [
        ("Air_temperature",     9999),
        ("Air_temperature",     100),
        ("Process_temperature", 9999),
        ("Rotational_speed",    -1),
        ("Rotational_speed",    99999),
        ("Torque",              -1),
        ("Tool_wear",           -5),
        ("Tool_wear",           9999),
    ])
    def test_out_of_range_raises(self, field, bad_value):
        with pytest.raises(ValueError, match="out of range"):
            self._validate({**VALID_PAYLOAD, field: bad_value})

    @pytest.mark.parametrize("field", ["Air_temperature", "Process_temperature", "Torque"])
    def test_non_numeric_raises(self, field):
        with pytest.raises(ValueError, match="must be numeric"):
            self._validate({**VALID_PAYLOAD, field: "not_a_number"})

    def test_boundary_values_accepted(self):
        self._validate({**VALID_PAYLOAD, "Air_temperature": 200.0})
        self._validate({**VALID_PAYLOAD, "Air_temperature": 400.0})
        self._validate({**VALID_PAYLOAD, "Tool_wear": 0})
        self._validate({**VALID_PAYLOAD, "Tool_wear": 500})


class TestBuildDataframe:
    def setup_method(self):
        from src.serving import inference as inf
        self._build = inf._build_dataframe

    def test_returns_dataframe(self):
        assert isinstance(self._build(VALID_PAYLOAD), pd.DataFrame)

    def test_single_row(self):
        assert len(self._build(VALID_PAYLOAD)) == 1

    def test_canonical_column_names(self):
        from src.core.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
        df = self._build(VALID_PAYLOAD)
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            assert col in df.columns, f"Missing column: {col}"

    def test_alias_keys_mapped_correctly(self):
        df = self._build(VALID_PAYLOAD)
        assert df["Air temperature [K]"].iloc[0]     == pytest.approx(298.1)
        assert df["Torque [Nm]"].iloc[0]             == pytest.approx(42.8)
        assert df["Tool wear [min]"].iloc[0]         == pytest.approx(0.0)

    def test_machine_type_uppercased(self):
        df = self._build({**VALID_PAYLOAD, "Type": "m"})
        assert df["Type"].iloc[0] == "M"


class TestPredict:
    """predict() tests using a real tiny pipeline from the session fixture."""

    def test_returns_dict(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        required = {
            "request_id", "prediction", "probability",
            "confidence", "risk_level", "status",
            "recommendations", "top_factors",
        }
        assert required.issubset(result.keys())

    def test_prediction_is_binary(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert result["prediction"] in (0, 1)

    def test_probability_in_unit_interval(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert 0.0 <= result["probability"] <= 1.0

    def test_confidence_in_unit_interval(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_risk_level_is_valid(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_request_id_is_uuid4(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        parsed = uuid.UUID(result["request_id"])
        assert parsed.version == 4

    def test_top_factors_count(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert len(result["top_factors"]) == 3

    def test_top_factors_importance_positive(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        for f in result["top_factors"]:
            assert f["importance"] >= 0

    def test_recommendations_non_empty(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD)
        assert len(result["recommendations"]) > 0

    def test_high_threshold_predicts_zero(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD, threshold=1.0)
        assert result["prediction"] == 0

    def test_zero_threshold_predicts_one(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            result = inf.predict(VALID_PAYLOAD, threshold=0.0)
        assert result["prediction"] == 1

    def test_invalid_input_raises_value_error(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            with pytest.raises(ValueError):
                inf.predict({**VALID_PAYLOAD, "Type": "Z"})

    def test_no_pipeline_raises_runtime_error(self):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", None):
            with pytest.raises(RuntimeError, match="degraded"):
                inf.predict(VALID_PAYLOAD)


class TestPredictBatch:
    """predict_batch() tests — list input, per-row success/failure isolation."""

    def test_returns_list(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            results = inf.predict_batch([VALID_PAYLOAD, VALID_PAYLOAD])
        assert isinstance(results, list)

    def test_length_matches_input(self, tiny_pipeline):
        from src.serving import inference as inf
        inputs = [VALID_PAYLOAD] * 5
        with patch.object(inf, "_pipeline", tiny_pipeline):
            results = inf.predict_batch(inputs)
        assert len(results) == 5

    def test_success_rows_have_required_keys(self, tiny_pipeline):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", tiny_pipeline):
            results = inf.predict_batch([VALID_PAYLOAD])
        row = results[0]
        assert row["success"] is True
        for key in ("request_id", "prediction", "probability", "risk_level"):
            assert key in row, f"Missing key: {key}"

    def test_invalid_row_is_isolated(self, tiny_pipeline):
        """A validation failure in one row must not abort the rest of the batch."""
        from src.serving import inference as inf
        inputs = [
            VALID_PAYLOAD,
            {**VALID_PAYLOAD, "Type": "INVALID"},
            VALID_PAYLOAD,
        ]
        with patch.object(inf, "_pipeline", tiny_pipeline):
            results = inf.predict_batch(inputs)
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"] is not None
        assert results[2]["success"] is True

    def test_no_pipeline_raises_runtime_error(self):
        from src.serving import inference as inf
        with patch.object(inf, "_pipeline", None):
            with pytest.raises(RuntimeError, match="degraded"):
                inf.predict_batch([VALID_PAYLOAD])
