#!/usr/bin/env python3
"""
CLI entry point for training the full CKD prediction pipeline.

Usage
-----
    python scripts/train.py                      # default paths from config
    python scripts/train.py --data path/to.csv   # custom CSV
    python scripts/train.py --artifacts out/      # custom artifacts dir
    python scripts/train.py --reports reports/    # save evaluation plots
    python scripts/train.py --verbose             # debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root without `pip install -e .`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CKD prediction models and save artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      type=Path, default=None, help="Path to raw CSV")
    p.add_argument("--artifacts", type=Path, default=None, help="Artifacts output directory")
    p.add_argument("--reports",   type=Path, default=None, help="Evaluation plots directory")
    p.add_argument("--verbose",   action="store_true",     help="Enable DEBUG logging")
    p.add_argument("--no-plots",  action="store_true",     help="Skip evaluation plot generation")
    p.add_argument("--track",     action="store_true",     help="Enable MLflow experiment tracking")
    p.add_argument("--run-name",  type=str, default=None,  help="MLflow run name (requires --track)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger("train")

    logger.info("=" * 60)
    logger.info("CKD Prediction — Training Pipeline")
    logger.info("=" * 60)

    t_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────
    from ckd.models.train import train
    from ckd.monitoring.mlflow_tracking import MLflowTracker
    from ckd.models.registry import save as registry_save

    tracker  = MLflowTracker() if args.track else None
    art_dir  = args.artifacts or None

    def _do_train():
        m = train(data_path=args.data, artifacts_dir=art_dir)
        if tracker:
            from ckd.config import ARTIFACTS_DIR
            tracker.log_training_result(m, art_dir or ARTIFACTS_DIR)
        return m

    if args.track and tracker:
        with tracker.start_run(run_name=args.run_name):
            metrics = _do_train()
    else:
        metrics = _do_train()

    # Register versioned model in the model registry
    import joblib as _jl
    from ckd.config import ARTIFACT_PIPELINE, ARTIFACTS_DIR as _ART
    _art_path = (art_dir or _ART) / ARTIFACT_PIPELINE
    if _art_path.exists():
        artifact = _jl.load(_art_path)
        vtag = registry_save(artifact, metrics, notes=f"run via scripts/train.py")
        logger.info("Registered model version: %s", vtag)

    logger.info("-" * 60)
    logger.info("Training complete in %.1f sec", time.time() - t_start)
    logger.info("Best model : %s", metrics["best_model"])

    test = metrics.get("test_results", {}).get(metrics["best_model"], {})
    if test:
        logger.info(
            "Test results  —  Acc=%.4f  F1=%.4f  AUC=%.4f",
            test["accuracy"], test["f1"], test["auc"],
        )

    # ── Evaluation plots ───────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("Generating evaluation plots…")
        import joblib
        from ckd.config import ARTIFACT_PIPELINE, ARTIFACTS_DIR
        from ckd.data.loader import load_clean
        from ckd.models.evaluate import generate_all_reports

        art_dir  = args.artifacts or ARTIFACTS_DIR
        artifact = joblib.load(Path(art_dir) / ARTIFACT_PIPELINE)
        le       = artifact["label_encoder"]
        feat_cols = artifact["feature_cols"]

        from sklearn.model_selection import train_test_split
        from ckd.config import RANDOM_STATE, TEST_SIZE

        X, y_raw = load_clean(args.data)
        from ckd.models.train import encode_target
        y, _   = encode_target(y_raw)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        generate_all_reports(
            models=artifact["all_models"],
            X_test=X_test,
            y_test=y_test,
            class_names=le.classes_.tolist(),
            out_dir=args.reports,
        )
        logger.info("Plots saved.")

    logger.info("=" * 60)
    logger.info("Done. Artifacts: %s", args.artifacts or "artifacts/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
