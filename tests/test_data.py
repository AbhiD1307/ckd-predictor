"""Tests for data loading and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ckd.data.loader import clean, _clean_target, _clean_mixed_column
from ckd.config import NUMERIC_FEATURES


class TestCleanTarget:
    def test_strips_tab(self):
        s = pd.Series(["ckd\t", "notckd", "ckd"])
        out = _clean_target(s)
        assert list(out) == ["ckd", "notckd", "ckd"]

    def test_lowercases(self):
        s = pd.Series(["CKD", "NOTCKD"])
        assert list(_clean_target(s)) == ["ckd", "notckd"]

    def test_strips_spaces(self):
        s = pd.Series(["  ckd  ", "notckd "])
        assert list(_clean_target(s)) == ["ckd", "notckd"]


class TestCleanMixedColumn:
    def test_converts_numeric_string(self):
        s = pd.Series(["44", "38", "31"])
        out = _clean_mixed_column(s)
        assert list(out) == [44.0, 38.0, 31.0]

    def test_noisy_tab_prefix(self):
        s = pd.Series(["\t43", "52"])
        out = _clean_mixed_column(s)
        assert out.iloc[0] == 43.0

    def test_question_mark_becomes_nan(self):
        s = pd.Series(["?", "44"])
        out = _clean_mixed_column(s)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == 44.0


class TestClean:
    def test_returns_X_and_y(self, synthetic_df):
        X, y = clean(synthetic_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_values(self, synthetic_df):
        _, y = clean(synthetic_df)
        assert set(y.unique()).issubset({"ckd", "notckd"})

    def test_no_classification_in_X(self, synthetic_df):
        X, _ = clean(synthetic_df)
        assert "classification" not in X.columns

    def test_lengths_match(self, synthetic_df):
        X, y = clean(synthetic_df)
        assert len(X) == len(y)

    def test_numeric_columns_are_float(self, synthetic_df):
        X, _ = clean(synthetic_df)
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                assert X[col].dtype in (np.float64, np.float32, float), col

    def test_drops_id_if_present(self):
        df = pd.DataFrame({
            "id": [1, 2],
            "age": [50.0, 40.0],
            "classification": ["ckd", "notckd"],
        })
        X, _ = clean(df)
        assert "id" not in X.columns
