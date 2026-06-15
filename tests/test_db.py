"""Tests for SQLite prediction database."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ckd.db.database import (
    delete_all,
    get_dataframe,
    get_recent,
    get_stats,
    init_db,
    log_prediction,
)


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Return a fresh temp DB path."""
    db = tmp_path / "test_predictions.db"
    init_db(db)
    return db


class TestInitDb:
    def test_creates_file(self, tmp_path):
        db = tmp_path / "new.db"
        assert not db.exists()
        init_db(db)
        assert db.exists()

    def test_idempotent(self, tmp_db):
        init_db(tmp_db)   # second call must not raise
        assert tmp_db.exists()


class TestLogPrediction:
    def test_returns_request_id(self, tmp_db):
        rid = log_prediction(
            model_name="Ensemble",
            prediction="ckd",
            probability_ckd=0.92,
            risk_level="High",
            patient={"age": 65, "htn": "yes"},
            db_path=tmp_db,
        )
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_row_inserted(self, tmp_db):
        log_prediction("TestModel", "notckd", 0.1, "Low", {}, db_path=tmp_db)
        rows = get_recent(db_path=tmp_db)
        assert len(rows) == 1
        assert rows[0]["prediction"] == "notckd"

    def test_multiple_rows(self, tmp_db):
        for i in range(5):
            log_prediction(
                "Model", "ckd" if i % 2 == 0 else "notckd",
                0.8, "High", {"age": 50 + i}, db_path=tmp_db
            )
        rows = get_recent(db_path=tmp_db)
        assert len(rows) == 5

    def test_duplicate_request_id_ignored(self, tmp_db):
        rid = "fixed-uuid-1234"
        log_prediction("M", "ckd", 0.9, "High", {}, request_id=rid, db_path=tmp_db)
        log_prediction("M", "ckd", 0.9, "High", {}, request_id=rid, db_path=tmp_db)
        assert len(get_recent(db_path=tmp_db)) == 1


class TestGetStats:
    def test_empty_db(self, tmp_db):
        stats = get_stats(db_path=tmp_db)
        assert stats["total"] == 0

    def test_counts(self, tmp_db):
        log_prediction("M", "ckd",    0.9, "High",   {}, db_path=tmp_db)
        log_prediction("M", "ckd",    0.8, "High",   {}, db_path=tmp_db)
        log_prediction("M", "notckd", 0.1, "Low",    {}, db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert stats["total"]       == 3
        assert stats["ckd_count"]   == 2
        assert stats["notckd_count"]== 1

    def test_ckd_rate(self, tmp_db):
        log_prediction("M", "ckd",    0.9, "High", {}, db_path=tmp_db)
        log_prediction("M", "notckd", 0.1, "Low",  {}, db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert stats["ckd_rate_pct"] == 50.0


class TestGetDataframe:
    def test_returns_dataframe(self, tmp_db):
        import pandas as pd
        log_prediction("M", "ckd", 0.9, "High", {"age": 60}, db_path=tmp_db)
        df = get_dataframe(db_path=tmp_db)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_empty_returns_empty_df(self, tmp_db):
        import pandas as pd
        df = get_dataframe(db_path=tmp_db)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestDeleteAll:
    def test_requires_confirm(self, tmp_db):
        with pytest.raises(ValueError):
            delete_all(db_path=tmp_db)

    def test_deletes_rows(self, tmp_db):
        log_prediction("M", "ckd", 0.9, "High", {}, db_path=tmp_db)
        n = delete_all(db_path=tmp_db, confirm=True)
        assert n == 1
        assert get_stats(db_path=tmp_db)["total"] == 0
