"""Tests for model training and evaluation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ckd.features.pipeline import build_preprocessor
from ckd.models.evaluate import compute_metrics
from ckd.models.train import encode_target


class TestEncodeTarget:
    def test_binary_output(self, synthetic_Xy):
        _, y = synthetic_Xy
        y_enc, le = encode_target(y)
        assert set(y_enc) == {0, 1}

    def test_classes_are_ckd_notckd(self, synthetic_Xy):
        _, y = synthetic_Xy
        _, le = encode_target(y)
        assert "ckd" in le.classes_
        assert "notckd" in le.classes_

    def test_length_preserved(self, synthetic_Xy):
        _, y = synthetic_Xy
        y_enc, _ = encode_target(y)
        assert len(y_enc) == len(y)


class TestComputeMetrics:
    def _fit_rf(self, X_train, y_train):
        pre  = build_preprocessor(X_train.columns.tolist())
        pipe = Pipeline([("preprocessor", pre),
                         ("clf", RandomForestClassifier(n_estimators=20, random_state=42))])
        pipe.fit(X_train, y_train)
        return pipe

    def test_keys_present(self, synthetic_Xy):
        X, y = synthetic_Xy
        y_enc, _ = encode_target(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.25, random_state=42
        )
        model  = self._fit_rf(X_train, y_train)
        result = compute_metrics(model, X_test, y_test)
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in result

    def test_metrics_in_range(self, synthetic_Xy):
        X, y = synthetic_Xy
        y_enc, _ = encode_target(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.25, random_state=42
        )
        model  = self._fit_rf(X_train, y_train)
        result = compute_metrics(model, X_test, y_test)
        for val in result.values():
            assert 0.0 <= val <= 1.0
