"""Tests for drift detection and data validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ckd.monitoring.drift import DriftDetector, _psi_one_feature
from ckd.data.validator import (
    LenientPatientInput,
    StrictPatientInput,
    validate_batch,
)


# ── PSI ────────────────────────────────────────────────────────────────────
class TestPSI:
    def test_identical_distributions_near_zero(self):
        arr = np.random.default_rng(42).uniform(0, 100, 200)
        psi = _psi_one_feature(arr, arr)
        assert psi < 0.01

    def test_very_different_distributions_high_psi(self):
        baseline = np.ones(200) * 10
        current  = np.ones(200) * 90
        psi = _psi_one_feature(baseline, current)
        assert psi > 0.10

    def test_returns_float(self):
        a = np.random.default_rng(0).normal(0, 1, 100)
        b = np.random.default_rng(1).normal(0, 1, 100)
        assert isinstance(_psi_one_feature(a, b), float)


# ── DriftDetector ──────────────────────────────────────────────────────────
class TestDriftDetector:
    def _make_df(self, n=100, seed=0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "age":  rng.uniform(20, 80, n),
            "hemo": rng.uniform(6, 17, n),
            "sc":   rng.uniform(0.4, 5, n),
        })

    def test_from_training_data(self):
        X = self._make_df()
        det = DriftDetector.from_training_data(X, list(X.columns))
        assert isinstance(det, DriftDetector)
        assert len(det._features) == 3

    def test_check_stable(self):
        X_train = self._make_df(n=200, seed=0)
        X_curr  = self._make_df(n=100, seed=1)   # same distribution
        det     = DriftDetector.from_training_data(X_train, list(X_train.columns))
        report  = det.check(X_curr)
        assert not report.needs_retraining

    def test_check_shifted(self):
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"age": rng.uniform(20, 40, 300)})
        X_curr  = pd.DataFrame({"age": rng.uniform(70, 90, 100)})   # huge shift
        det     = DriftDetector.from_training_data(X_train, ["age"])
        report  = det.check(X_curr)
        assert report.overall_psi > 0.10

    def test_save_load(self, tmp_path):
        X   = self._make_df()
        det = DriftDetector.from_training_data(X, list(X.columns))
        p   = tmp_path / "baseline.json"
        det.save(p)
        det2 = DriftDetector.load(p)
        assert det2._features == det._features

    def test_report_summary_string(self):
        X   = self._make_df(seed=42)
        det = DriftDetector.from_training_data(X, list(X.columns))
        r   = det.check(self._make_df(seed=99))
        assert isinstance(r.summary(), str)
        assert "PSI" in r.summary()

    def test_report_to_dict(self):
        X   = self._make_df()
        det = DriftDetector.from_training_data(X, list(X.columns))
        r   = det.check(self._make_df(seed=5))
        d   = r.to_dict()
        assert "overall_psi" in d
        assert "features" in d


# ── Validator ──────────────────────────────────────────────────────────────
class TestStrictValidator:
    def test_valid_input(self):
        p = StrictPatientInput(age=50, bp=80, htn="yes", dm="no")
        assert p.age == 50

    def test_invalid_age(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StrictPatientInput(age=200)

    def test_invalid_categorical(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StrictPatientInput(htn="maybe")

    def test_lowercases_string(self):
        p = StrictPatientInput(htn="YES")
        assert p.htn == "yes"

    def test_strips_whitespace(self):
        p = StrictPatientInput(rbc=" normal ")
        assert p.rbc == "normal"

    def test_empty_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StrictPatientInput()


class TestLenientValidator:
    def test_out_of_range_becomes_none(self):
        p = LenientPatientInput(age=999, bp=80)
        assert p.age is None    # coerced to None
        assert p.bp == 80

    def test_invalid_categorical_becomes_none(self):
        p = LenientPatientInput(htn="unknown")
        assert p.htn is None

    def test_valid_passes_through(self):
        p = LenientPatientInput(age=50, htn="yes")
        assert p.age == 50
        assert p.htn == "yes"


class TestValidateBatch:
    def test_all_valid(self):
        records = [{"age": 40}, {"age": 60, "htn": "yes"}]
        valid, report = validate_batch(records)
        assert len(valid)    == 2
        assert report.valid  == 2
        assert report.invalid == 0

    def test_partial_invalid(self):
        records = [{"age": 40}, {"age": 9999}]   # second is out of range
        valid, report = validate_batch(records, strict=True)
        assert report.invalid >= 1

    def test_empty_batch(self):
        valid, report = validate_batch([])
        assert report.total == 0
        assert report.valid == 0
