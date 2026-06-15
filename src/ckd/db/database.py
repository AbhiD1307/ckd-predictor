"""
SQLite prediction database — logs every API prediction for auditing,
monitoring, and downstream analytics.

Schema
------
predictions
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    request_id      TEXT UNIQUE           — UUID per request
    created_at      TEXT                  — ISO timestamp
    model_name      TEXT
    prediction      TEXT                  — 'ckd' | 'notckd'
    probability_ckd REAL
    risk_level      TEXT
    age             REAL
    bp              REAL
    hemo            REAL
    sc              REAL
    htn             TEXT
    dm              TEXT
    patient_json    TEXT                  — full input as JSON
    shap_json       TEXT                  — SHAP values as JSON (nullable)
    source          TEXT                  — 'api' | 'dashboard' | 'batch'

Usage
-----
    from ckd.db.database import init_db, log_prediction, get_recent, get_stats

    init_db()
    log_prediction(request_id, model_name, label, prob, risk, patient_dict, source="api")
    rows = get_recent(limit=20)
    stats = get_stats()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from ckd.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

DB_PATH = ARTIFACTS_DIR / "predictions.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL UNIQUE,
    created_at      TEXT    NOT NULL,
    model_name      TEXT    NOT NULL,
    prediction      TEXT    NOT NULL,
    probability_ckd REAL    NOT NULL,
    risk_level      TEXT    NOT NULL,
    age             REAL,
    bp              REAL,
    hemo            REAL,
    sc              REAL,
    htn             TEXT,
    dm              TEXT,
    patient_json    TEXT,
    shap_json       TEXT,
    source          TEXT    DEFAULT 'api'
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_created_at  ON predictions(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_prediction  ON predictions(prediction);",
    "CREATE INDEX IF NOT EXISTS idx_risk_level  ON predictions(risk_level);",
    "CREATE INDEX IF NOT EXISTS idx_model_name  ON predictions(model_name);",
]


# ── Connection context manager ─────────────────────────────────────────────
@contextmanager
def _connect(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Lifecycle ──────────────────────────────────────────────────────────────
def init_db(db_path: Path | None = None) -> None:
    """Create the database and tables if they do not exist."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute(_CREATE_TABLE)
        for idx in _CREATE_INDEXES:
            conn.execute(idx)
    logger.info("Prediction database initialised at %s", path)


# ── Write ──────────────────────────────────────────────────────────────────
def log_prediction(
    model_name: str,
    prediction: str,
    probability_ckd: float,
    risk_level: str,
    patient: dict[str, Any],
    shap_values: dict[str, float] | None = None,
    source: str = "api",
    request_id: str | None = None,
    db_path: Path | None = None,
) -> str:
    """
    Insert one prediction record into the database.

    Returns the request_id (auto-generated UUID if not supplied).
    """
    path = db_path or DB_PATH
    if not path.exists():
        init_db(path)

    rid = request_id or str(uuid.uuid4())
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO predictions
                (request_id, created_at, model_name, prediction,
                 probability_ckd, risk_level,
                 age, bp, hemo, sc, htn, dm,
                 patient_json, shap_json, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rid,
                datetime.now(timezone.utc).isoformat(),
                model_name,
                prediction,
                round(probability_ckd, 6),
                risk_level,
                patient.get("age"),
                patient.get("bp"),
                patient.get("hemo"),
                patient.get("sc"),
                patient.get("htn"),
                patient.get("dm"),
                json.dumps(patient),
                json.dumps(shap_values) if shap_values else None,
                source,
            ),
        )
    return rid


# ── Read ───────────────────────────────────────────────────────────────────
def get_recent(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the most recent predictions as a list of dicts."""
    path = db_path or DB_PATH
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats(db_path: Path | None = None) -> dict[str, Any]:
    """Return aggregate statistics on logged predictions."""
    path = db_path or DB_PATH
    if not path.exists():
        return {"total": 0, "ckd_count": 0, "notckd_count": 0}
    with _connect(path) as conn:
        total      = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        ckd_count  = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction='ckd'"
        ).fetchone()[0]
        avg_prob   = conn.execute(
            "SELECT AVG(probability_ckd) FROM predictions"
        ).fetchone()[0]
        by_risk    = dict(
            conn.execute(
                "SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level"
            ).fetchall()
        )
        by_model   = dict(
            conn.execute(
                "SELECT model_name, COUNT(*) FROM predictions GROUP BY model_name"
            ).fetchall()
        )
        by_source  = dict(
            conn.execute(
                "SELECT source, COUNT(*) FROM predictions GROUP BY source"
            ).fetchall()
        )
    return {
        "total":            total,
        "ckd_count":        ckd_count,
        "notckd_count":     total - ckd_count,
        "ckd_rate_pct":     round(100 * ckd_count / max(total, 1), 2),
        "avg_probability":  round(avg_prob or 0.0, 4),
        "by_risk_level":    by_risk,
        "by_model":         by_model,
        "by_source":        by_source,
    }


def get_dataframe(db_path: Path | None = None):
    """Return all predictions as a pandas DataFrame."""
    import pandas as pd
    rows = get_recent(limit=100_000, db_path=db_path)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def delete_all(db_path: Path | None = None, confirm: bool = False) -> int:
    """Delete all rows. Returns row count deleted. Requires confirm=True."""
    if not confirm:
        raise ValueError("Pass confirm=True to delete all prediction logs.")
    path = db_path or DB_PATH
    if not path.exists():
        return 0
    with _connect(path) as conn:
        n = conn.execute("DELETE FROM predictions").rowcount
    logger.warning("Deleted %d prediction log rows", n)
    return n
