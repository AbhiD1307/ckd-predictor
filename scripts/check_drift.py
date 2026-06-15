#!/usr/bin/env python3
"""
Data drift check CLI — compare recent prediction inputs against training baseline.

Usage
-----
    python scripts/check_drift.py                         # uses DB prediction history
    python scripts/check_drift.py --input data/new.csv    # use a new CSV
    python scripts/check_drift.py --alert-on-high         # exit 1 if high drift found
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check data drift vs. training baseline.")
    p.add_argument("--input",        type=Path, default=None, help="New data CSV (uses DB if not set)")
    p.add_argument("--baseline",     type=Path, default=None, help="Path to drift_baseline.json")
    p.add_argument("--alert-on-high",action="store_true",     help="Exit code 1 if retraining needed")
    p.add_argument("--save-report",  type=Path, default=None, help="Save report JSON here")
    p.add_argument("--verbose",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("check_drift")

    import pandas as pd
    from ckd.monitoring.drift import DriftDetector
    from ckd.config import NUMERIC_FEATURES, MIXED_STR_FEATURES

    numeric_cols = NUMERIC_FEATURES + MIXED_STR_FEATURES

    # ── Load current data ────────────────────────────────────────────────
    if args.input:
        if not args.input.exists():
            logger.error("Input file not found: %s", args.input)
            sys.exit(1)
        X_current = pd.read_csv(args.input)
        logger.info("Loaded current data from %s (%d rows)", args.input, len(X_current))
    else:
        from ckd.db.database import get_dataframe
        df_hist = get_dataframe()
        if df_hist.empty:
            logger.warning("No prediction history in DB. Run some predictions first.")
            sys.exit(0)
        X_current = df_hist
        logger.info("Loaded %d rows from prediction DB", len(X_current))

    # Keep only numeric columns present
    X_current = X_current[[c for c in numeric_cols if c in X_current.columns]]

    # ── Load drift detector ──────────────────────────────────────────────
    try:
        detector = DriftDetector.load(args.baseline)
    except FileNotFoundError:
        logger.warning("No drift baseline found — building from training data…")
        from ckd.data.loader import load_clean
        X_train, _ = load_clean()
        detector = DriftDetector.from_training_data(X_train, list(X_current.columns))
        detector.save()

    # ── Check drift ──────────────────────────────────────────────────────
    report = detector.check(X_current)
    print(report.summary())

    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Report saved → %s", args.save_report)

    if args.alert_on_high and report.needs_retraining:
        logger.warning("⚠️  High drift detected — retraining recommended!")
        sys.exit(1)


if __name__ == "__main__":
    main()
