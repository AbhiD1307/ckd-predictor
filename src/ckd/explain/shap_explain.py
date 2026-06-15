"""
SHAP-based explanations for the XGBoost estimator embedded in an sklearn Pipeline.

Provides:
  - global_summary()  → beeswarm / bar summary plot (training set)
  - local_waterfall() → single-sample waterfall chart
  - local_values()    → dict of {feature: shap_value} for API / AI layer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

_N_DISPLAY = 12   # features shown in local waterfall


def _extract_xgb(pipeline: Pipeline) -> tuple[Any, np.ndarray]:
    """
    Extract the XGBoost classifier and preprocessed X from a fitted Pipeline.
    Returns (xgb_clf, transform_fn) where transform_fn(X) → np.ndarray.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf          = pipeline.named_steps["clf"]
    return clf, preprocessor


def _transform(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].transform(X)


def build_explainer(pipeline: Pipeline, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer backed by the XGB clf inside the pipeline."""
    clf = pipeline.named_steps["clf"]
    X_t = _transform(pipeline, X_background)
    explainer = shap.TreeExplainer(clf, data=X_t, feature_perturbation="interventional")
    logger.info("SHAP TreeExplainer built (background n=%d)", len(X_background))
    return explainer


def _shap_values_1d(explainer: shap.TreeExplainer, X_t: np.ndarray) -> np.ndarray:
    """Return 1-D SHAP values (works for both binary list output and 2-D array)."""
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        return sv[0]            # class 0 = CKD
    return sv if sv.ndim == 1 else sv[:, 0]


def global_summary(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    feature_names: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """Beeswarm SHAP summary plot over the training set."""
    clf = pipeline.named_steps["clf"]
    X_t = _transform(pipeline, X_train)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_t, feature_names=feature_names, show=False, plot_size=None)
    plt.title("SHAP Global Feature Importance (XGBoost)", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved global SHAP summary → %s", save_path)
    return fig


def local_waterfall(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    title: str = "Feature Impact on This Prediction",
    save_path: Path | None = None,
) -> plt.Figure:
    """Waterfall chart showing top-N feature impacts for a single sample."""
    clf        = pipeline.named_steps["clf"]
    X_t        = _transform(pipeline, X_sample)
    explainer  = shap.TreeExplainer(clf)
    sv_1d      = _shap_values_1d(explainer, X_t)[0]    # shape (n_features,)

    pairs  = sorted(zip(np.abs(sv_1d), sv_1d, feature_names), reverse=True)
    top    = pairs[:_N_DISPLAY]
    _, vals, names = zip(*top)

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (impact on log-odds of CKD)")
    ax.set_title(title, fontsize=12)
    red_p  = mpatches.Patch(color="#d62728", label="→ increases CKD risk")
    blue_p = mpatches.Patch(color="#1f77b4", label="→ decreases CKD risk")
    ax.legend(handles=[red_p, blue_p], fontsize=9, loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def local_values(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Return {feature_name: shap_value} for the sample — used by the AI explainer.
    Sorted by absolute magnitude (largest first).
    """
    clf       = pipeline.named_steps["clf"]
    X_t       = _transform(pipeline, X_sample)
    explainer = shap.TreeExplainer(clf)
    sv_1d     = _shap_values_1d(explainer, X_t)[0]

    pairs = sorted(zip(feature_names, sv_1d.tolist()), key=lambda t: abs(t[1]), reverse=True)
    return dict(pairs)
