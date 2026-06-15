"""
Model evaluation utilities: metrics, ROC curves, confusion matrices, calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from ckd.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    pos_label: int = 0,
) -> dict[str, float]:
    """Return a flat dict of accuracy / precision / recall / F1 / AUC."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, pos_label]
    else:
        y_prob = model.decision_function(X)

    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=pos_label)
    roc_auc     = auc(fpr, tpr)

    return {
        "accuracy":  round(accuracy_score(y, y_pred), 6),
        "precision": round(precision_score(y, y_pred, pos_label=pos_label, zero_division=0), 6),
        "recall":    round(recall_score(y, y_pred, pos_label=pos_label, zero_division=0), 6),
        "f1":        round(f1_score(y, y_pred, pos_label=pos_label, zero_division=0), 6),
        "auc":       round(roc_auc, 6),
    }


def plot_confusion_matrix(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    y_pred = model.predict(X)
    cm     = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved confusion matrix → %s", save_path)
    return fig


def plot_roc_curves(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y, y_prob, pos_label=1)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved ROC curves → %s", save_path)
    return fig


def plot_calibration(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Reliability / calibration diagram."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        prob_pos = model.predict_proba(X)[:, 0]   # P(ckd)
        frac_pos, mean_pred = calibration_curve(y == 0, prob_pos, n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", label=name, markersize=4)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def print_classification_report(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: list[str],
    title: str = "",
) -> None:
    y_pred = model.predict(X)
    if title:
        print(f"\n{'='*50}\n{title}\n{'='*50}")
    print(classification_report(y, y_pred, target_names=class_names, digits=4))


def generate_all_reports(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    out_dir: Path | None = None,
) -> None:
    """Save confusion matrices + ROC + calibration plots for all models."""
    out = out_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            model, X_test, y_test, class_names,
            title=f"Confusion Matrix – {name}",
            save_path=out / f"cm_{safe_name}.png",
        )
        plt.close("all")

    plot_roc_curves(models, X_test, y_test, save_path=out / "roc_curves.png")
    plt.close("all")

    plot_calibration(models, X_test, y_test, save_path=out / "calibration.png")
    plt.close("all")

    logger.info("All evaluation plots saved to %s", out)
