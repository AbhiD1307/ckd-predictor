#!/usr/bin/env python3
"""
Batch inference CLI — predict CKD for many patients from a CSV file.

Usage
-----
    python scripts/predict.py --input data/patients.csv --output results.csv
    python scripts/predict.py --input data/patients.csv --model "XGBoost (Tuned)"
    python scripts/predict.py --input data/patients.csv --explain      # adds SHAP top-3
    python scripts/predict.py --input data/patients.csv --format jsonl # JSON Lines output

Input CSV must have at least some of the 24 clinical feature columns.
Missing columns default to training-set median/mode via the preprocessing pipeline.

Output columns added:
    prediction          — 'ckd' or 'notckd'
    probability_ckd     — float [0, 1]
    probability_not_ckd — float [0, 1]
    risk_level          — 'High' / 'Medium' / 'Low'
    shap_top_feature_1  — (if --explain) top SHAP feature name
    shap_top_value_1    — (if --explain) SHAP value
    ... up to shap_top_3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch CKD prediction from CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     type=Path, required=True,  help="Input CSV path")
    p.add_argument("--output",    type=Path, default=None,   help="Output path (default: <input>_predictions.csv)")
    p.add_argument("--artifacts", type=Path, default=None,   help="Artifacts directory")
    p.add_argument("--model",     type=str,  default=None,   help="Model name to use")
    p.add_argument("--explain",   action="store_true",       help="Add top-3 SHAP features to output")
    p.add_argument("--format",    choices=["csv", "jsonl"],  default="csv")
    p.add_argument("--threshold", type=float, default=0.5,   help="CKD classification threshold")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def _risk_level(p: float) -> str:
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger("predict")

    import joblib
    import numpy as np
    import pandas as pd

    from ckd.config import ARTIFACT_PIPELINE, ARTIFACTS_DIR

    art_dir  = args.artifacts or ARTIFACTS_DIR
    pipeline_path = Path(art_dir) / ARTIFACT_PIPELINE
    if not pipeline_path.exists():
        logger.error("No trained model at %s — run 'make train' first.", pipeline_path)
        sys.exit(1)

    logger.info("Loading model artifact…")
    artifact  = joblib.load(pipeline_path)
    le        = artifact["label_encoder"]
    feat_cols = artifact["feature_cols"]
    all_models = artifact["all_models"]

    model_name = args.model or artifact["model_name"]
    if model_name not in all_models:
        logger.error("Unknown model '%s'. Available: %s", model_name, list(all_models))
        sys.exit(1)
    model = all_models[model_name]
    logger.info("Using model: %s", model_name)

    # ── Load input ─────────────────────────────────────────────────────────
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    df_in = pd.read_csv(args.input)
    logger.info("Loaded %d patients from %s", len(df_in), args.input)

    # Ensure all feature cols exist (fill missing with NaN)
    for col in feat_cols:
        if col not in df_in.columns:
            df_in[col] = np.nan

    # ── Predict ────────────────────────────────────────────────────────────
    t0 = time.time()
    X = df_in[feat_cols]

    pred_enc = model.predict(X)
    proba    = model.predict_proba(X)
    ckd_idx  = list(le.classes_).index("ckd")

    labels    = le.inverse_transform(pred_enc)
    prob_ckd  = proba[:, ckd_idx]

    # Apply custom threshold
    if args.threshold != 0.5:
        labels = np.where(prob_ckd >= args.threshold, "ckd", "notckd")

    df_out = df_in.copy()
    df_out["prediction"]          = labels
    df_out["probability_ckd"]     = prob_ckd.round(4)
    df_out["probability_not_ckd"] = (1 - prob_ckd).round(4)
    df_out["risk_level"]          = [_risk_level(p) for p in prob_ckd]
    df_out["model_used"]          = model_name

    logger.info("Predicted %d patients in %.2f sec", len(df_out), time.time() - t0)

    # ── Optional SHAP top-3 ────────────────────────────────────────────────
    if args.explain:
        logger.info("Computing SHAP explanations…")
        xgb_name = next((n for n in ("XGBoost (Tuned)", "XGBoost") if n in all_models), None)
        if xgb_name:
            try:
                from ckd.explain.shap_explain import local_values
                for rank in range(1, 4):
                    df_out[f"shap_top_feature_{rank}"] = ""
                    df_out[f"shap_top_value_{rank}"]   = np.nan

                xgb_pipe = all_models[xgb_name]
                for i, row in df_in[feat_cols].iterrows():
                    sv = local_values(xgb_pipe, pd.DataFrame([row]), feat_cols)
                    for rank, (fname, fval) in enumerate(list(sv.items())[:3], start=1):
                        df_out.at[i, f"shap_top_feature_{rank}"] = fname
                        df_out.at[i, f"shap_top_value_{rank}"]   = round(fval, 5)
            except Exception as exc:
                logger.warning("SHAP failed: %s", exc)

    # ── Summary stats ──────────────────────────────────────────────────────
    n_ckd    = int((labels == "ckd").sum())
    n_notckd = int((labels == "notckd").sum())
    logger.info("Results: %d CKD  |  %d Not-CKD  |  %.1f%% positive rate",
                n_ckd, n_notckd, 100 * n_ckd / max(len(labels), 1))

    # ── Write output ───────────────────────────────────────────────────────
    out_path = args.output or args.input.with_name(args.input.stem + "_predictions.csv")

    if args.format == "jsonl":
        out_path = out_path.with_suffix(".jsonl")
        with open(out_path, "w") as f:
            for record in df_out.to_dict(orient="records"):
                f.write(json.dumps(record) + "\n")
    else:
        df_out.to_csv(out_path, index=False)

    logger.info("Saved predictions → %s", out_path)

    # Print summary table
    print("\n" + "=" * 55)
    print(f"  Batch Prediction Summary")
    print("=" * 55)
    print(f"  Total patients : {len(df_out)}")
    print(f"  CKD positive   : {n_ckd}  ({100*n_ckd/max(len(labels),1):.1f}%)")
    print(f"  Not-CKD        : {n_notckd}")
    print(f"  Model used     : {model_name}")
    print(f"  Output file    : {out_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
