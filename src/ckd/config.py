"""
Central configuration — paths, hyperparameter grids, feature schemas.
Import this module everywhere instead of scattering magic strings/numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parents[2]   # project root
DATA_DIR      = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR   = ROOT_DIR / "reports"

RAW_CSV       = ROOT_DIR / "Chronic_Kidney_Disease.csv"  # existing file

for _d in (ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE:    float = 0.20
CV_FOLDS:     int = 5

# ── Feature schema ─────────────────────────────────────────────────────────
TARGET_COL = "classification"

NUMERIC_FEATURES = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo"]

CATEGORICAL_FEATURES = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

# pcv / wc / rc are stored as strings in the raw CSV (mixed type column)
MIXED_STR_FEATURES = ["pcv", "wc", "rc"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + MIXED_STR_FEATURES

CATEGORICAL_VALUES: dict[str, list[str]] = {
    "rbc":   ["normal", "abnormal"],
    "pc":    ["normal", "abnormal"],
    "pcc":   ["notpresent", "present"],
    "ba":    ["notpresent", "present"],
    "htn":   ["no", "yes"],
    "dm":    ["no", "yes"],
    "cad":   ["no", "yes"],
    "appet": ["good", "poor"],
    "pe":    ["no", "yes"],
    "ane":   ["no", "yes"],
}

# ── Model hyperparameter grids (GridSearchCV) ──────────────────────────────
RF_PARAM_GRID: dict[str, Any] = {
    "n_estimators":    [200, 400],
    "max_depth":       [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "max_features":    ["sqrt", "log2"],
}

SVM_PARAM_GRID: dict[str, Any] = {
    "C":      [0.5, 1, 2],
    "gamma":  ["scale", 0.1],
    "kernel": ["rbf"],
}

XGB_PARAM_GRID: dict[str, Any] = {
    "n_estimators":    [200, 300],
    "learning_rate":   [0.05, 0.1],
    "max_depth":       [3, 4, 6],
    "subsample":       [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}

# ── Artifact filenames ─────────────────────────────────────────────────────
ARTIFACT_PIPELINE  = "full_pipeline.joblib"
ARTIFACT_METRICS   = "metrics.json"

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Claude AI ──────────────────────────────────────────────────────────────
ANTHROPIC_MODEL   = "claude-sonnet-4-6"
ANTHROPIC_MAX_TOKENS = 512
