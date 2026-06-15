"""
MLflow experiment tracking integration.

Wraps the training pipeline to log:
  - Parameters   : hyperparameter grids, CV folds, random state, test size
  - Metrics      : accuracy, F1, AUC, precision, recall per model
  - Artifacts    : model pipeline .joblib, metrics.json, evaluation plots
  - Tags         : author, dataset, model name, git commit (if available)

Usage
-----
    from ckd.monitoring.mlflow_tracking import MLflowTracker

    tracker = MLflowTracker(experiment_name="ckd_experiment")
    with tracker.start_run(run_name="ensemble_v2") as run:
        metrics = train(...)
        tracker.log_training_result(metrics, artifact_dir)

    # Or simply pass track=True to scripts/train.py:
    python scripts/train.py --track

MLflow UI:
    mlflow ui --port 5000
    → http://localhost:5000
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("mlflow not installed — experiment tracking disabled. pip install mlflow")


def _git_commit() -> str | None:
    """Return the current git commit hash, or None."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


class MLflowTracker:
    """Thin wrapper around MLflow for CKD training runs."""

    def __init__(
        self,
        experiment_name: str = "ckd_prediction",
        tracking_uri: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri    = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self._run            = None

        if not _MLFLOW_AVAILABLE:
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info("MLflow tracking URI: %s  experiment: %s", self.tracking_uri, self.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator["MLflowTracker", None, None]:
        """Context manager wrapping an MLflow run."""
        if not _MLFLOW_AVAILABLE:
            logger.info("MLflow not available — skipping tracking.")
            yield self
            return

        default_tags = {
            "author":   "Abhishek Ashok Deshmukh",
            "course":   "CSS 581 ML",
            "dataset":  "UCI CKD",
        }
        commit = _git_commit()
        if commit:
            default_tags["git_commit"] = commit
        default_tags.update(tags or {})

        with mlflow.start_run(run_name=run_name, tags=default_tags) as run:
            self._run = run
            logger.info("MLflow run started: %s  (id=%s)", run_name or "auto", run.info.run_id)
            try:
                yield self
            finally:
                self._run = None
                logger.info("MLflow run finished: %s", run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        if not _MLFLOW_AVAILABLE or not mlflow.active_run():
            return
        flat = {k: str(v)[:250] for k, v in params.items()}
        mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not _MLFLOW_AVAILABLE or not mlflow.active_run():
            return
        num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(num_metrics, step=step)

    def log_artifact(self, path: Path | str) -> None:
        if not _MLFLOW_AVAILABLE or not mlflow.active_run():
            return
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
        else:
            logger.warning("Artifact not found, skipping: %s", p)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if not _MLFLOW_AVAILABLE or not mlflow.active_run():
            return
        try:
            mlflow.sklearn.log_model(model, artifact_path)
        except Exception as exc:
            logger.warning("mlflow.sklearn.log_model failed: %s", exc)

    def log_training_result(
        self,
        metrics_payload: dict[str, Any],
        artifact_dir: Path,
    ) -> None:
        """
        Log a complete training result: params, all model metrics, artifacts.
        Call inside an active MLflow run.
        """
        if not _MLFLOW_AVAILABLE or not mlflow.active_run():
            return

        # Params
        self.log_params({
            "random_state":    metrics_payload.get("random_state"),
            "test_size":       metrics_payload.get("test_size"),
            "cv_folds":        metrics_payload.get("cv_folds"),
            "best_model":      metrics_payload.get("best_model"),
            "n_features":      len(metrics_payload.get("feature_cols", [])),
            **{f"rf_{k}": v for k, v in (metrics_payload.get("rf_best_params") or {}).items()},
            **{f"xgb_{k}": v for k, v in (metrics_payload.get("xgb_best_params") or {}).items()},
        })

        # Test metrics for each model
        for model_name, result in (metrics_payload.get("test_results") or {}).items():
            prefix = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            self.log_metrics({
                f"{prefix}_accuracy":  result.get("accuracy", 0),
                f"{prefix}_f1":        result.get("f1", 0),
                f"{prefix}_auc":       result.get("auc", 0),
                f"{prefix}_precision": result.get("precision", 0),
                f"{prefix}_recall":    result.get("recall", 0),
            })

        # Best model CV metrics
        best = metrics_payload.get("best_model", "")
        cv   = (metrics_payload.get("cv_results") or {}).get(best, {})
        if cv:
            self.log_metrics({
                "best_cv_f1":  cv.get("cv_f1", 0),
                "best_cv_auc": cv.get("cv_auc", 0),
            })

        # Artifacts
        for fname in ["full_pipeline.joblib", "metrics.json"]:
            self.log_artifact(artifact_dir / fname)

        from ckd.config import REPORTS_DIR
        for img in REPORTS_DIR.glob("*.png"):
            self.log_artifact(img)

        logger.info("MLflow: logged params, metrics, and artifacts.")
