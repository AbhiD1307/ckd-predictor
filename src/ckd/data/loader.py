"""
Data loading and cleaning for the UCI Chronic Kidney Disease dataset.

Responsibilities:
  - Read raw CSV and validate schema
  - Clean noisy target labels ('ckd\\t' → 'ckd')
  - Fix mixed-type columns (pcv, wc, rc) stored as strings with embedded noise
  - Return a clean DataFrame ready for the feature pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ckd.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MIXED_STR_FEATURES,
    RAW_CSV,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

_NUMERIC_NOISE = {"?", "\\t?", "\t?", ""}


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read CSV and return DataFrame with the original column names intact."""
    csv_path = Path(path) if path else RAW_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            "Download from UCI ML Repository and place as 'Chronic_Kidney_Disease.csv'."
        )
    df = pd.read_csv(csv_path)
    logger.info("Loaded raw dataset: %s rows × %s cols from %s", *df.shape, csv_path)
    return df


def _clean_target(series: pd.Series) -> pd.Series:
    """Strip whitespace/tabs from target values ('ckd\\t' → 'ckd')."""
    return series.astype(str).str.strip().str.lower()


def _clean_mixed_column(series: pd.Series) -> pd.Series:
    """
    pcv / wc / rc are numeric but stored as object dtype in the raw CSV
    with noise values like '\\t43', '?', blanks.  Convert to float.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full cleaning pass.

    Returns
    -------
    X : pd.DataFrame   — feature matrix (ALL_FEATURES columns, raw dtype)
    y : pd.Series      — binary target, values {'ckd', 'notckd'}
    """
    df = df.copy()

    # ── Target ─────────────────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Got: {df.columns.tolist()}")
    y = _clean_target(df[TARGET_COL])
    unexpected = set(y.unique()) - {"ckd", "notckd"}
    if unexpected:
        logger.warning("Unexpected target values (dropping rows): %s", unexpected)
        mask = y.isin({"ckd", "notckd"})
        df = df[mask].copy()
        y  = y[mask]

    # ── Drop id (leaks row order, not a clinical feature) ──────────────────
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # ── Mixed-type columns → numeric ───────────────────────────────────────
    for col in MIXED_STR_FEATURES:
        if col in df.columns:
            df[col] = _clean_mixed_column(df[col])

    # ── Categorical: strip whitespace ──────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", np.nan)

    # ── Subset to known feature columns ────────────────────────────────────
    available = [c for c in ALL_FEATURES if c in df.columns]
    X = df[available]

    missing_cols = set(ALL_FEATURES) - set(available)
    if missing_cols:
        logger.warning("Feature columns absent from CSV (will be zero-filled): %s", missing_cols)

    logger.info("Clean dataset: %s rows × %s features", len(X), len(X.columns))
    return X, y.reset_index(drop=True)


def load_clean(path: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience: load_raw + clean in one call."""
    return clean(load_raw(path))
