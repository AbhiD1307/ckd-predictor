"""
Model Registry — version-controlled artifact management.

Every training run is saved with a version tag (timestamp-based).
Supports:
  - save(artifact, tag)     → save a versioned artifact
  - load(tag="latest")      → load a specific or latest version
  - list_versions()         → all saved versions with metadata
  - compare(tag_a, tag_b)   → side-by-side metric comparison
  - promote(tag, "production") → mark a version as production
  - rollback()              → revert production to previous version

Storage layout:
  artifacts/
    registry/
      v20240101_120000/
        pipeline.joblib
        metrics.json
        meta.json          ← version metadata + tags
      v20240102_093000/
        ...
      production -> v20240102_093000   (symlink)
      latest     -> v20240102_093000   (symlink)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from ckd.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

REGISTRY_DIR = ARTIFACTS_DIR / "registry"


def _version_tag() -> str:
    return "v" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _version_dir(tag: str) -> Path:
    return REGISTRY_DIR / tag


def _update_symlink(link: Path, target: Path) -> None:
    """Create or update a symlink (works on macOS/Linux)."""
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)


# ── Public API ─────────────────────────────────────────────────────────────
def save(
    artifact: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    tag: str | None = None,
    notes: str = "",
) -> str:
    """
    Save a trained artifact to the registry under a new version.

    Parameters
    ----------
    artifact : dict — the full pipeline artifact (model, label_encoder, etc.)
    metrics  : dict — training/test metrics
    tag      : str  — custom version tag (auto-generated if None)
    notes    : str  — free-text notes attached to this version

    Returns
    -------
    version tag string
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    vtag = tag or _version_tag()
    vdir = _version_dir(vtag)
    vdir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(artifact, vdir / "pipeline.joblib")

    # Save metrics
    if metrics:
        with open(vdir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    # Save metadata
    meta = {
        "version":    vtag,
        "created_at": datetime.now().isoformat(),
        "model_name": artifact.get("model_name", "unknown"),
        "notes":      notes,
        "promoted_to": [],
        "test_f1":    _extract_f1(metrics),
        "test_auc":   _extract_auc(metrics),
    }
    with open(vdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Update "latest" symlink
    _update_symlink(REGISTRY_DIR / "latest", vdir)

    logger.info("Saved model version %s → %s", vtag, vdir)
    return vtag


def load(tag: str = "latest") -> dict[str, Any]:
    """
    Load a versioned artifact.

    Parameters
    ----------
    tag : 'latest', 'production', or a specific version tag like 'v20240101_120000'
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    target = REGISTRY_DIR / tag

    # Resolve symlink
    if target.is_symlink():
        target = target.resolve()
    elif not target.exists():
        raise FileNotFoundError(
            f"Version '{tag}' not found in registry at {REGISTRY_DIR}. "
            f"Available: {list_versions()}"
        )

    pipeline_path = target / "pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"pipeline.joblib missing in version {tag}")

    artifact = joblib.load(pipeline_path)
    logger.info("Loaded model version %s from %s", tag, target)
    return artifact


def list_versions() -> list[dict[str, Any]]:
    """Return metadata for all registered versions, newest first."""
    if not REGISTRY_DIR.exists():
        return []
    versions = []
    for d in sorted(REGISTRY_DIR.iterdir(), reverse=True):
        if d.is_dir() and not d.is_symlink() and d.name.startswith("v"):
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    versions.append(json.load(f))
    return versions


def promote(tag: str, stage: str = "production") -> None:
    """
    Promote a version to a named stage (e.g. 'production', 'staging').
    Creates a symlink REGISTRY_DIR/<stage> → version_dir.
    """
    vdir = _version_dir(tag)
    if not vdir.exists():
        raise ValueError(f"Version '{tag}' does not exist in registry.")

    # Update symlink
    link = REGISTRY_DIR / stage
    _update_symlink(link, vdir)

    # Update meta
    meta_path = vdir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if stage not in meta.get("promoted_to", []):
            meta.setdefault("promoted_to", []).append(stage)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    logger.info("Promoted %s → %s", tag, stage)


def rollback(stage: str = "production") -> str:
    """
    Roll back a stage to the second-most-recent version promoted to it.
    Returns the version tag it was rolled back to.
    """
    versions = list_versions()
    candidates = [
        v for v in versions
        if stage in v.get("promoted_to", []) and v["version"] != _current_tag(stage)
    ]
    if not candidates:
        raise RuntimeError(f"No previous version to roll back to for stage '{stage}'.")
    target_tag = candidates[0]["version"]
    promote(target_tag, stage)
    logger.warning("Rolled back '%s' to %s", stage, target_tag)
    return target_tag


def compare(tag_a: str, tag_b: str) -> dict[str, Any]:
    """Side-by-side metric comparison between two versions."""
    def _load_meta(tag: str) -> dict:
        p = _version_dir(tag) / "meta.json"
        if not p.exists():
            raise FileNotFoundError(f"No meta.json for version '{tag}'")
        return json.load(open(p))

    meta_a = _load_meta(tag_a)
    meta_b = _load_meta(tag_b)

    return {
        "version_a": {
            "tag":      tag_a,
            "created":  meta_a.get("created_at"),
            "model":    meta_a.get("model_name"),
            "test_f1":  meta_a.get("test_f1"),
            "test_auc": meta_a.get("test_auc"),
            "notes":    meta_a.get("notes"),
        },
        "version_b": {
            "tag":      tag_b,
            "created":  meta_b.get("created_at"),
            "model":    meta_b.get("model_name"),
            "test_f1":  meta_b.get("test_f1"),
            "test_auc": meta_b.get("test_auc"),
            "notes":    meta_b.get("notes"),
        },
        "winner": tag_a if (meta_a.get("test_f1") or 0) >= (meta_b.get("test_f1") or 0) else tag_b,
    }


def delete(tag: str, confirm: bool = False) -> None:
    """Delete a versioned artifact (requires confirm=True)."""
    if not confirm:
        raise ValueError("Pass confirm=True to delete a registered model version.")
    vdir = _version_dir(tag)
    if not vdir.exists():
        raise FileNotFoundError(f"Version '{tag}' not found.")
    shutil.rmtree(vdir)
    logger.warning("Deleted model version %s", tag)


# ── Helpers ────────────────────────────────────────────────────────────────
def _current_tag(stage: str) -> str | None:
    link = REGISTRY_DIR / stage
    if link.is_symlink():
        return link.resolve().name
    return None


def _extract_f1(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    best = metrics.get("best_model")
    if best:
        return metrics.get("test_results", {}).get(best, {}).get("f1")
    return None


def _extract_auc(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    best = metrics.get("best_model")
    if best:
        return metrics.get("test_results", {}).get(best, {}).get("auc")
    return None
