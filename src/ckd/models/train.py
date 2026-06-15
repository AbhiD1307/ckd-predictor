"""
Model training: builds an end-to-end sklearn Pipeline per algorithm,
runs GridSearchCV with stratified k-fold, selects the best estimator,
wraps the top models in a Soft Voting Ensemble, and serialises artifacts.

Algorithms
----------
- Logistic Regression   (baseline, highly interpretable)
- Random Forest         (tree ensemble, feature importance)
- SVM RBF               (kernel, good on small datasets)
- XGBoost               (gradient boosting, SHAP-compatible)
- Soft Voting Ensemble  (RF + XGB, final production model)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ckd.config import (
    ARTIFACT_METRICS,
    ARTIFACT_PIPELINE,
    ARTIFACTS_DIR,
    CV_FOLDS,
    RANDOM_STATE,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    TEST_SIZE,
    XGB_PARAM_GRID,
)
from ckd.data.loader import load_clean
from ckd.features.pipeline import build_preprocessor

logger = logging.getLogger(__name__)


# ── Label encoding ──────────────────────────────────────────────────────────
def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    return le.fit_transform(y), le  # 'ckd'→0 , 'notckd'→1  (alphabetical)


# ── Build per-model end-to-end Pipelines ───────────────────────────────────
def _make_pipeline(preprocessor: Any, estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("clf", estimator)])


def _wrap_grid(
    pipeline: Pipeline,
    param_grid: dict[str, Any],
    cv: StratifiedKFold,
    scoring: str = "f1",
) -> GridSearchCV:
    prefixed = {f"clf__{k}": v for k, v in param_grid.items()}
    return GridSearchCV(
        pipeline,
        prefixed,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


# ── Main training function ──────────────────────────────────────────────────
def train(
    data_path: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Full training run.

    Returns a metrics dict that is also saved as metrics.json.
    """
    art_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    art_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & split ──────────────────────────────────────────────────────
    logger.info("Loading dataset…")
    X, y_raw = load_clean(data_path)
    y, le    = encode_target(y_raw)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Split: train=%d  test=%d", len(X_train), len(X_test))

    # 2. Preprocessor (fit only on train inside each CV fold) ─────────────
    preprocessor = build_preprocessor(X_train.columns.tolist())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 3. Define base estimators ───────────────────────────────────────────
    base_estimators: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(
                                   n_estimators=300, learning_rate=0.05, max_depth=4,
                                   subsample=0.9, colsample_bytree=0.9,
                                   random_state=RANDOM_STATE, eval_metric="logloss",
                               ),
    }

    # 4. Baseline cross-validation (no tuning) ────────────────────────────
    logger.info("Running baseline cross-validation…")
    cv_results: dict[str, dict[str, float]] = {}
    baseline_pipelines: dict[str, Pipeline] = {}

    for name, est in base_estimators.items():
        import copy
        pipe = _make_pipeline(copy.deepcopy(preprocessor), est)
        t0   = time.time()
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1,
        )
        elapsed = time.time() - t0
        cv_results[name] = {
            "cv_accuracy": float(np.mean(scores["test_accuracy"])),
            "cv_f1":       float(np.mean(scores["test_f1"])),
            "cv_auc":      float(np.mean(scores["test_roc_auc"])),
            "cv_time_sec": round(elapsed, 3),
        }
        logger.info("  %-22s  F1=%.4f  AUC=%.4f", name, cv_results[name]["cv_f1"], cv_results[name]["cv_auc"])
        # Refit on full train for test evaluation
        pipe.fit(X_train, y_train)
        baseline_pipelines[name] = pipe

    # 5. Hyperparameter tuning ────────────────────────────────────────────
    logger.info("Tuning Random Forest…")
    rf_gs = _wrap_grid(
        _make_pipeline(build_preprocessor(X_train.columns.tolist()),
                       RandomForestClassifier(random_state=RANDOM_STATE)),
        RF_PARAM_GRID, cv,
    )
    rf_gs.fit(X_train, y_train)
    logger.info("  RF best CV F1: %.4f  params: %s", rf_gs.best_score_, rf_gs.best_params_)

    logger.info("Tuning XGBoost…")
    xgb_gs = _wrap_grid(
        _make_pipeline(build_preprocessor(X_train.columns.tolist()),
                       XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
        XGB_PARAM_GRID, cv,
    )
    xgb_gs.fit(X_train, y_train)
    logger.info("  XGB best CV F1: %.4f  params: %s", xgb_gs.best_score_, xgb_gs.best_params_)

    logger.info("Tuning SVM…")
    svm_gs = _wrap_grid(
        _make_pipeline(build_preprocessor(X_train.columns.tolist()),
                       SVC(probability=True, random_state=RANDOM_STATE)),
        SVM_PARAM_GRID, cv,
    )
    svm_gs.fit(X_train, y_train)
    logger.info("  SVM best CV F1: %.4f", svm_gs.best_score_)

    tuned_pipelines: dict[str, Pipeline] = {
        "Random Forest (Tuned)": rf_gs.best_estimator_,
        "XGBoost (Tuned)":       xgb_gs.best_estimator_,
        "SVM (Tuned)":           svm_gs.best_estimator_,
    }

    # 6. Soft Voting Ensemble ─────────────────────────────────────────────
    logger.info("Building Soft Voting Ensemble (RF + XGB)…")
    ensemble = VotingClassifier(
        estimators=[
            ("rf",  rf_gs.best_estimator_),
            ("xgb", xgb_gs.best_estimator_),
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)

    # 7. Evaluate all on hold-out test set ────────────────────────────────
    from ckd.models.evaluate import compute_metrics

    all_pipelines: dict[str, Any] = {
        **baseline_pipelines,
        **tuned_pipelines,
        "Ensemble (Voting)": ensemble,
    }
    test_results: dict[str, dict[str, float]] = {}
    for name, pipe in all_pipelines.items():
        test_results[name] = compute_metrics(pipe, X_test, y_test)
        logger.info(
            "  %-28s  F1=%.4f  AUC=%.4f  Acc=%.4f",
            name,
            test_results[name]["f1"],
            test_results[name]["auc"],
            test_results[name]["accuracy"],
        )

    # 8. Identify & save final model ──────────────────────────────────────
    best_name = max(
        ["Ensemble (Voting)", "Random Forest (Tuned)", "XGBoost (Tuned)"],
        key=lambda n: test_results[n]["f1"],
    )
    final_model = all_pipelines[best_name]
    logger.info("Final model selected: %s", best_name)

    # Save the XGBoost tuned pipeline separately for SHAP (needs the clf step directly)
    xgb_pipeline_path = art_dir / "xgb_pipeline.joblib"
    joblib.dump(xgb_gs.best_estimator_, xgb_pipeline_path)

    # Save final production pipeline
    artifact = {
        "model":         final_model,
        "label_encoder": le,
        "feature_cols":  X_train.columns.tolist(),
        "model_name":    best_name,
        "all_models":    all_pipelines,
    }
    joblib.dump(artifact, art_dir / ARTIFACT_PIPELINE)
    logger.info("Saved pipeline artifact → %s", art_dir / ARTIFACT_PIPELINE)

    # 9. Save metrics JSON ────────────────────────────────────────────────
    metrics_payload = {
        "random_state":   RANDOM_STATE,
        "test_size":      TEST_SIZE,
        "cv_folds":       CV_FOLDS,
        "feature_cols":   X_train.columns.tolist(),
        "best_model":     best_name,
        "cv_results":     cv_results,
        "test_results":   test_results,
        "rf_best_params": rf_gs.best_params_,
        "xgb_best_params": xgb_gs.best_params_,
        "svm_best_params": svm_gs.best_params_,
        "label_classes":  le.classes_.tolist(),
    }
    with open(art_dir / ARTIFACT_METRICS, "w") as f:
        json.dump(metrics_payload, f, indent=2, default=str)
    logger.info("Saved metrics → %s", art_dir / ARTIFACT_METRICS)

    return metrics_payload
