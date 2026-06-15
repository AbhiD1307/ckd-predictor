"""
Data drift detection — compares incoming prediction requests against
the training distribution to alert when the model may be degrading.

Two complementary detectors
----------------------------
1. Statistical (PSI)      — Population Stability Index per numeric feature
2. Prediction drift       — monitors shift in CKD positive rate over time

PSI interpretation
------------------
  PSI < 0.10  → No significant change
  PSI < 0.25  → Moderate change — investigate
  PSI >= 0.25 → Significant shift — consider retraining

Usage
-----
    from ckd.monitoring.drift import DriftDetector

    detector = DriftDetector.from_training_data(X_train, feature_names)
    detector.save()                              # persist baseline stats

    # Later, after serving N predictions:
    report = detector.check(X_new)
    print(report.summary())
    if report.needs_retraining:
        trigger_retrain()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ckd.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

DRIFT_STATS_PATH = ARTIFACTS_DIR / "drift_baseline.json"

PSI_LOW      = 0.10
PSI_MODERATE = 0.25
N_BINS       = 10


# ── PSI computation ────────────────────────────────────────────────────────
def _psi_one_feature(baseline: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """Compute PSI for one numeric feature."""
    # Build bins from baseline
    _, bin_edges = np.histogram(baseline, bins=n_bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    def _bucket_probs(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=bin_edges)
        probs = counts / max(len(arr), 1)
        return np.clip(probs, 1e-6, None)   # avoid log(0)

    p_base = _bucket_probs(baseline)
    p_curr = _bucket_probs(current)
    psi    = float(np.sum((p_curr - p_base) * np.log(p_curr / p_base)))
    return round(psi, 5)


def _psi_label(psi: float) -> str:
    if psi < PSI_LOW:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "high"


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class FeatureDrift:
    name:       str
    psi:        float
    status:     str          # 'stable' | 'moderate' | 'high'
    train_mean: float
    current_mean: float
    mean_shift_pct: float


@dataclass
class DriftReport:
    n_train:           int
    n_current:         int
    feature_drifts:    list[FeatureDrift]
    prediction_drift:  dict[str, Any]
    overall_psi:       float
    needs_retraining:  bool
    generated_at:      str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Data Drift Report",
            f"  Training samples : {self.n_train}",
            f"  Current samples  : {self.n_current}",
            f"  Overall PSI      : {self.overall_psi:.4f}",
            f"  Needs retraining : {'⚠️  YES' if self.needs_retraining else '✅ NO'}",
            f"{'='*55}",
            f"  {'Feature':<18} {'PSI':>8}  Status",
            f"  {'-'*42}",
        ]
        for fd in sorted(self.feature_drifts, key=lambda f: -f.psi):
            icon = "🔴" if fd.status == "high" else "🟠" if fd.status == "moderate" else "🟢"
            lines.append(f"  {fd.name:<18} {fd.psi:>8.4f}  {icon} {fd.status}")
        if self.prediction_drift:
            lines.append(f"\n  Prediction Drift:")
            for k, v in self.prediction_drift.items():
                lines.append(f"    {k}: {v}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train":          self.n_train,
            "n_current":        self.n_current,
            "overall_psi":      self.overall_psi,
            "needs_retraining": self.needs_retraining,
            "generated_at":     self.generated_at,
            "features": [
                {
                    "name":           fd.name,
                    "psi":            fd.psi,
                    "status":         fd.status,
                    "train_mean":     fd.train_mean,
                    "current_mean":   fd.current_mean,
                    "mean_shift_pct": fd.mean_shift_pct,
                }
                for fd in self.feature_drifts
            ],
            "prediction_drift": self.prediction_drift,
        }


# ── Detector ───────────────────────────────────────────────────────────────
class DriftDetector:
    """Stateful drift detector backed by training-set statistics."""

    def __init__(
        self,
        feature_stats: dict[str, dict],   # {feature: {mean, std, values: list}}
        feature_names: list[str],
    ):
        self._stats   = feature_stats
        self._features = feature_names

    @classmethod
    def from_training_data(
        cls,
        X_train: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> "DriftDetector":
        """Build baseline statistics from the training DataFrame."""
        cols   = feature_names or X_train.select_dtypes(include=np.number).columns.tolist()
        stats  = {}
        for col in cols:
            if col in X_train.columns:
                arr = X_train[col].dropna().to_numpy()
                stats[col] = {
                    "mean":   float(arr.mean()) if len(arr) else 0.0,
                    "std":    float(arr.std())  if len(arr) else 0.0,
                    "min":    float(arr.min())  if len(arr) else 0.0,
                    "max":    float(arr.max())  if len(arr) else 0.0,
                    "values": arr.tolist()[:5000],   # cap to 5K for storage
                }
        logger.info("DriftDetector baseline built from %d training samples", len(X_train))
        return cls(stats, list(stats.keys()))

    def save(self, path: Path | None = None) -> None:
        """Persist baseline stats to JSON."""
        target = path or DRIFT_STATS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump({"features": self._features, "stats": self._stats}, f)
        logger.info("Drift baseline saved → %s", target)

    @classmethod
    def load(cls, path: Path | None = None) -> "DriftDetector":
        """Restore from saved JSON."""
        target = path or DRIFT_STATS_PATH
        if not target.exists():
            raise FileNotFoundError(
                f"Drift baseline not found at {target}. "
                "Run DriftDetector.from_training_data(...).save() after training."
            )
        with open(target) as f:
            data = json.load(f)
        return cls(data["stats"], data["features"])

    def check(
        self,
        X_current: pd.DataFrame,
        current_ckd_rate: float | None = None,
        train_ckd_rate: float | None = None,
    ) -> DriftReport:
        """
        Compare current data distribution against training baseline.

        Parameters
        ----------
        X_current        : new feature matrix (numeric columns only)
        current_ckd_rate : optional — fraction of CKD predictions in current window
        train_ckd_rate   : optional — fraction of CKD in training labels
        """
        feature_drifts: list[FeatureDrift] = []

        for feat, stat in self._stats.items():
            if feat not in X_current.columns:
                continue
            curr_arr  = X_current[feat].dropna().to_numpy()
            train_arr = np.array(stat["values"])
            if len(curr_arr) < 5 or len(train_arr) < 5:
                continue

            psi           = _psi_one_feature(train_arr, curr_arr)
            curr_mean     = float(curr_arr.mean())
            train_mean    = stat["mean"]
            shift_pct     = round(
                100 * abs(curr_mean - train_mean) / max(abs(train_mean), 1e-9), 2
            )
            feature_drifts.append(FeatureDrift(
                name=feat,
                psi=psi,
                status=_psi_label(psi),
                train_mean=round(train_mean, 4),
                current_mean=round(curr_mean, 4),
                mean_shift_pct=shift_pct,
            ))

        overall_psi = round(
            float(np.mean([fd.psi for fd in feature_drifts])) if feature_drifts else 0.0, 5
        )

        # Prediction drift
        pred_drift: dict[str, Any] = {}
        if current_ckd_rate is not None and train_ckd_rate is not None:
            delta = abs(current_ckd_rate - train_ckd_rate)
            pred_drift = {
                "train_ckd_rate":   round(train_ckd_rate, 4),
                "current_ckd_rate": round(current_ckd_rate, 4),
                "absolute_delta":   round(delta, 4),
                "status":           "high" if delta > 0.15 else "moderate" if delta > 0.07 else "stable",
            }

        high_drift = any(fd.status == "high" for fd in feature_drifts)
        pred_alert = pred_drift.get("status") == "high"

        return DriftReport(
            n_train=len(next(iter(self._stats.values()), {}).get("values", [])),
            n_current=len(X_current),
            feature_drifts=feature_drifts,
            prediction_drift=pred_drift,
            overall_psi=overall_psi,
            needs_retraining=high_drift or pred_alert,
        )
