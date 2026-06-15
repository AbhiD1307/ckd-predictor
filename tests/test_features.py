"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from ckd.features.pipeline import build_preprocessor, get_feature_names


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, synthetic_Xy):
        X, _ = synthetic_Xy
        preprocessor = build_preprocessor(X.columns.tolist())
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_shape(self, synthetic_Xy):
        X, _ = synthetic_Xy
        preprocessor = build_preprocessor(X.columns.tolist())
        X_t = preprocessor.fit_transform(X)
        assert X_t.shape[0] == len(X)
        assert X_t.shape[1] > 0

    def test_no_nans_after_transform(self, synthetic_Xy):
        X, _ = synthetic_Xy
        preprocessor = build_preprocessor(X.columns.tolist())
        X_t = preprocessor.fit_transform(X)
        assert not np.any(np.isnan(X_t))

    def test_fit_on_train_transform_test(self, synthetic_Xy):
        X, y = synthetic_Xy
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        preprocessor = build_preprocessor(X_train.columns.tolist())
        preprocessor.fit(X_train)
        X_test_t = preprocessor.transform(X_test)
        assert X_test_t.shape[0] == len(X_test)
        assert not np.any(np.isnan(X_test_t))

    def test_feature_names_length(self, synthetic_Xy):
        X, _ = synthetic_Xy
        preprocessor = build_preprocessor(X.columns.tolist())
        preprocessor.fit(X)
        names = get_feature_names(preprocessor)
        X_t   = preprocessor.transform(X)
        assert len(names) == X_t.shape[1]

    def test_handles_missing_columns(self, synthetic_Xy):
        X, _ = synthetic_Xy
        # Drop some columns — should not raise
        X_partial = X.drop(columns=["hemo", "dm"], errors="ignore")
        preprocessor = build_preprocessor(X_partial.columns.tolist())
        X_t = preprocessor.fit_transform(X_partial)
        assert X_t.shape[0] == len(X_partial)
