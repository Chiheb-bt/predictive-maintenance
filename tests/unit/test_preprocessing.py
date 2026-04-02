"""Unit tests for src/core/preprocessing.py — no model required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.core.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    FIELD_RANGES,
    INPUT_ALIASES,
    NUMERIC_FEATURES,
    VALID_TYPES,
    build_preprocessor,
)

_SAMPLE = pd.DataFrame([{
    "Air temperature [K]":     298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]":  1551.0,
    "Torque [Nm]":             42.8,
    "Tool wear [min]":         0.0,
    "Type":                    "M",
}])


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        pp = build_preprocessor()
        assert isinstance(pp, ColumnTransformer)

    def test_fit_transform_produces_float_array(self):
        pp  = build_preprocessor()
        out = pp.fit_transform(_SAMPLE)
        assert isinstance(out, np.ndarray)
        assert out.dtype.kind == "f"

    def test_output_column_count(self):
        pp  = build_preprocessor()
        out = pp.fit_transform(_SAMPLE)
        # 5 numeric + 2 one-hot (Type_L, Type_M; H is dropped reference)
        assert out.shape == (1, 7)

    def test_feature_names_out_count(self):
        pp = build_preprocessor()
        pp.fit(_SAMPLE)
        assert len(pp.get_feature_names_out()) == 7

    def test_h_type_dropped_as_reference(self):
        pp = build_preprocessor()
        pp.fit(_SAMPLE)
        names = list(pp.get_feature_names_out())
        assert "cat__Type_H" not in names
        assert "cat__Type_L" in names
        assert "cat__Type_M" in names

    def test_numeric_features_standardised(self):
        # Fit on a two-row dataset; the M-type row should have standardised numerics.
        df = pd.concat([_SAMPLE] * 3, ignore_index=True)
        pp  = build_preprocessor()
        out = pp.fit_transform(df)
        # After StandardScaler on identical rows the output should be all zeros for numerics.
        np.testing.assert_array_almost_equal(out[:, :5], 0.0)

    @pytest.mark.parametrize("machine_type", ["L", "M", "H"])
    def test_all_machine_types_accepted(self, machine_type):
        sample = _SAMPLE.copy()
        sample["Type"] = machine_type
        pp  = build_preprocessor()
        out = pp.fit_transform(sample)
        assert out.shape[1] == 7


class TestSchemaConstants:
    def test_numeric_features_count(self):
        assert len(NUMERIC_FEATURES) == 5

    def test_categorical_features_count(self):
        assert len(CATEGORICAL_FEATURES) == 1

    def test_feature_labels_count(self):
        assert len(FEATURE_LABELS) == 7

    def test_all_numeric_features_have_labels(self):
        for feat in NUMERIC_FEATURES:
            key = f"num__{feat}"
            assert key in FEATURE_LABELS, f"Missing label for {key}"

    def test_field_ranges_cover_all_numeric_features(self):
        for feat in NUMERIC_FEATURES:
            assert feat in FIELD_RANGES, f"Missing range for {feat}"

    def test_input_aliases_map_to_canonical_names(self):
        for _alias, canonical in INPUT_ALIASES.items():
            assert canonical in NUMERIC_FEATURES, f"{canonical} not in NUMERIC_FEATURES"

    def test_valid_types_set(self):
        assert VALID_TYPES == frozenset({"L", "M", "H"})

    def test_field_ranges_are_ordered(self):
        for feat, (lo, hi) in FIELD_RANGES.items():
            assert lo < hi, f"Range for {feat} is inverted"
