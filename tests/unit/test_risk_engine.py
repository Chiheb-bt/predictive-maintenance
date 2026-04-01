"""Unit tests for src/core/risk_engine.py — no ML dependencies."""

from __future__ import annotations

import pytest

from src.core.risk_engine import (
    FeatureFactor,
    classify_risk,
    get_recommendations,
    top_factors,
)


class TestClassifyRisk:
    def test_low(self):
        assert classify_risk(0.0)  == "LOW"
        assert classify_risk(0.24) == "LOW"

    def test_medium_boundary(self):
        assert classify_risk(0.25) == "MEDIUM"
        assert classify_risk(0.49) == "MEDIUM"

    def test_high_boundary(self):
        assert classify_risk(0.50) == "HIGH"
        assert classify_risk(0.74) == "HIGH"

    def test_critical_boundary(self):
        assert classify_risk(0.75)  == "CRITICAL"
        assert classify_risk(1.0)   == "CRITICAL"

    @pytest.mark.parametrize("prob,expected", [
        (0.10, "LOW"),
        (0.30, "MEDIUM"),
        (0.60, "HIGH"),
        (0.90, "CRITICAL"),
    ])
    def test_parametrized(self, prob, expected):
        assert classify_risk(prob) == expected


class TestGetRecommendations:
    def test_returns_list(self):
        recs = get_recommendations("LOW")
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_all_risk_levels_have_recommendations(self):
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            assert len(get_recommendations(level)) > 0

    def test_factor_specific_prepended_for_known_factor(self):
        recs_with    = get_recommendations("HIGH", top_factor="Tool Wear")
        recs_without = get_recommendations("HIGH")
        assert len(recs_with) == len(recs_without) + 1
        assert "tool wear" in recs_with[0].lower()

    def test_unknown_factor_leaves_base_unchanged(self):
        recs_with    = get_recommendations("HIGH", top_factor="Unknown Sensor XYZ")
        recs_without = get_recommendations("HIGH")
        assert recs_with == recs_without

    def test_none_factor_leaves_base_unchanged(self):
        assert get_recommendations("LOW", top_factor=None) == get_recommendations("LOW")

    @pytest.mark.parametrize("factor", [
        "Tool Wear", "Torque", "Rotational Speed", "Air Temperature", "Process Temperature",
    ])
    def test_all_known_factors_prepend(self, factor):
        base = get_recommendations("MEDIUM")
        with_factor = get_recommendations("MEDIUM", top_factor=factor)
        assert len(with_factor) == len(base) + 1


class TestTopFactors:
    _NAMES = [
        "num__Air temperature [K]",
        "num__Process temperature [K]",
        "num__Rotational speed [rpm]",
        "num__Torque [Nm]",
        "num__Tool wear [min]",
    ]
    _LABELS = {
        "num__Air temperature [K]":     "Air Temperature",
        "num__Process temperature [K]": "Process Temperature",
        "num__Rotational speed [rpm]":  "Rotational Speed",
        "num__Torque [Nm]":             "Torque",
        "num__Tool wear [min]":         "Tool Wear",
    }
    _IMPORTANCES = [0.05, 0.08, 0.12, 0.27, 0.38]

    def test_returns_correct_count(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, self._LABELS, top_n=3)
        assert len(result) == 3

    def test_returns_feature_factors(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, self._LABELS, top_n=3)
        for f in result:
            assert isinstance(f, FeatureFactor)

    def test_sorted_descending_by_importance(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, self._LABELS, top_n=5)
        for i in range(len(result) - 1):
            assert result[i].importance >= result[i + 1].importance

    def test_ranks_start_at_one(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, self._LABELS, top_n=3)
        assert result[0].rank == 1

    def test_top_factor_is_tool_wear(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, self._LABELS, top_n=1)
        assert result[0].label == "Tool Wear"

    def test_importances_sum_to_one(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, self._LABELS, top_n=5)
        total = sum(f.importance for f in result)
        assert abs(total - 1.0) < 1e-3

    def test_label_lookup_fallback(self):
        result = top_factors(self._IMPORTANCES, self._NAMES, {}, top_n=1)
        assert result[0].label == self._NAMES[-1]

    def test_all_zero_importances_does_not_raise(self):
        zeros = [0.0] * len(self._IMPORTANCES)
        result = top_factors(zeros, self._NAMES, self._LABELS, top_n=3)
        assert len(result) == 3
