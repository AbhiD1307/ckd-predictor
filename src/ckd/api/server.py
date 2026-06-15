"""
FastAPI REST API for the CKD Prediction service.

Features
--------
- API key authentication (header: X-API-Key)
- Every prediction logged to SQLite (ckd.db.database)
- Input validated via ckd.data.validator
- SHAP + Claude AI on /predict/explain
- Batch endpoint (up to 500 patients)
- /health, /models, /stats, /predictions/history endpoints
- OpenAPI docs at /docs

Run locally:
    uvicorn ckd.api.server:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ckd.config import ARTIFACT_METRICS, ARTIFACT_PIPELINE, ARTIFACTS_DIR
from ckd.data.validator import LenientPatientInput, validate_batch
from ckd.db.database import get_recent, get_stats, init_db, log_prediction
from ckd.explain.shap_explain import local_values
from ckd.features.pipeline import get_feature_names

logger = logging.getLogger(__name__)

# ── Initialise DB on startup ───────────────────────────────────────────────
init_db()

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CKD Prediction API",
    description=(
        "**Production REST API** for Chronic Kidney Disease prediction.\n\n"
        "Powered by a Soft Voting Ensemble (Tuned Random Forest + Tuned XGBoost) "
        "trained on the UCI CKD dataset (400 patients, 25 features).\n\n"
        "**Authentication:** pass your API key in the `X-API-Key` header.  \n"
        "Set `CKD_API_KEY` in your `.env` file (leave blank to disable auth).\n\n"
        "**Author:** Abhishek Ashok Deshmukh · CSS 581 ML · UW Bothell"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key auth ───────────────────────────────────────────────────────────
_API_KEY = os.getenv("CKD_API_KEY", "")   # empty = auth disabled


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not _API_KEY:
        return   # auth disabled
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


ApiKeyDep = Annotated[None, Depends(verify_api_key)]


# ── Artifact loading (singleton) ───────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_artifact(path: str = str(ARTIFACTS_DIR / ARTIFACT_PIPELINE)) -> dict[str, Any]:
    if not Path(path).exists():
        raise RuntimeError(
            f"Model not found at {path}. Run 'make train' first."
        )
    return joblib.load(path)


def get_artifact() -> dict[str, Any]:
    try:
        return _load_artifact()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Request timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ── Pydantic response schemas ──────────────────────────────────────────────
class PredictionResponse(BaseModel):
    request_id:          str
    prediction:          str
    probability_ckd:     float
    probability_not_ckd: float
    risk_level:          str
    model_used:          str
    response_time_ms:    float | None = None


class ExplainResponse(PredictionResponse):
    shap_values:    dict[str, float]
    ai_explanation: str


class BatchPredictionResponse(BaseModel):
    total:         int
    results:       list[PredictionResponse]
    summary:       dict[str, Any]


class HealthResponse(BaseModel):
    status:       str
    version:      str
    model_loaded: bool
    db_ready:     bool


# ── Inference helpers ──────────────────────────────────────────────────────
def _risk_level(p: float) -> str:
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def _run_inference(
    patient_dict: dict[str, Any],
    artifact: dict[str, Any],
    model_name: str | None = None,
) -> tuple[str, float, str, pd.DataFrame]:
    le          = artifact["label_encoder"]
    feat_cols   = artifact["feature_cols"]
    all_models  = artifact["all_models"]
    chosen      = model_name or artifact["model_name"]

    if chosen not in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{chosen}'. Available: {list(all_models.keys())}",
        )
    model = all_models[chosen]

    df = pd.DataFrame([patient_dict])
    for col in feat_cols:
        if col not in df.columns:
            df[col] = np.nan

    pred_enc  = model.predict(df[feat_cols])[0]
    proba     = model.predict_proba(df[feat_cols])[0]
    label     = le.inverse_transform([pred_enc])[0]
    ckd_idx   = list(le.classes_).index("ckd")
    prob_ckd  = float(proba[ckd_idx])

    return label, prob_ckd, chosen, df[feat_cols]


def _get_shap(pipeline, df_proc: pd.DataFrame, feat_cols: list[str]) -> dict[str, float]:
    try:
        return local_values(pipeline, df_proc, feat_cols)
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return {}


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    model_loaded = (ARTIFACTS_DIR / ARTIFACT_PIPELINE).exists()
    db_ready     = True
    try:
        get_stats()
    except Exception:
        db_ready = False
    return HealthResponse(
        status="ok",
        version=app.version,
        model_loaded=model_loaded,
        db_ready=db_ready,
    )


@app.get("/models", tags=["System"])
def list_models(_: ApiKeyDep) -> dict[str, Any]:
    artifact = get_artifact()
    metrics: dict = {}
    mp = ARTIFACTS_DIR / ARTIFACT_METRICS
    if mp.exists():
        import json
        metrics = json.load(open(mp))
    return {
        "available_models": list(artifact["all_models"].keys()),
        "default_model":    artifact["model_name"],
        "test_results":     metrics.get("test_results", {}),
        "cv_results":       metrics.get("cv_results", {}),
    }


@app.get("/stats", tags=["Analytics"])
def prediction_stats(_: ApiKeyDep) -> dict[str, Any]:
    """Aggregate stats from the prediction database."""
    return get_stats()


@app.get("/predictions/history", tags=["Analytics"])
def prediction_history(
    _: ApiKeyDep,
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Return recent predictions from the database."""
    return get_recent(limit=limit)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    patient: LenientPatientInput,
    _: ApiKeyDep,
    model: Optional[str] = Query(None, description="Override model name"),
) -> PredictionResponse:
    t0       = time.perf_counter()
    artifact = get_artifact()
    feat_cols = artifact["feature_cols"]
    label, prob_ckd, model_used, df_proc = _run_inference(
        patient.model_dump(), artifact, model
    )
    rid = log_prediction(
        model_name=model_used,
        prediction=label,
        probability_ckd=prob_ckd,
        risk_level=_risk_level(prob_ckd),
        patient=patient.model_dump(exclude_none=True),
        source="api",
    )
    return PredictionResponse(
        request_id=rid,
        prediction=label,
        probability_ckd=round(prob_ckd, 4),
        probability_not_ckd=round(1 - prob_ckd, 4),
        risk_level=_risk_level(prob_ckd),
        model_used=model_used,
        response_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    patients: list[LenientPatientInput],
    _: ApiKeyDep,
    model: Optional[str] = Query(None),
) -> BatchPredictionResponse:
    if len(patients) > 500:
        raise HTTPException(status_code=400, detail="Batch limit is 500 patients.")

    artifact  = get_artifact()
    results   = []
    ckd_count = 0

    for pat in patients:
        label, prob_ckd, model_used, df_proc = _run_inference(
            pat.model_dump(), artifact, model
        )
        rid = log_prediction(
            model_name=model_used,
            prediction=label,
            probability_ckd=prob_ckd,
            risk_level=_risk_level(prob_ckd),
            patient=pat.model_dump(exclude_none=True),
            source="api_batch",
        )
        results.append(PredictionResponse(
            request_id=rid,
            prediction=label,
            probability_ckd=round(prob_ckd, 4),
            probability_not_ckd=round(1 - prob_ckd, 4),
            risk_level=_risk_level(prob_ckd),
            model_used=model_used,
        ))
        if label == "ckd":
            ckd_count += 1

    return BatchPredictionResponse(
        total=len(results),
        results=results,
        summary={
            "ckd_count":     ckd_count,
            "notckd_count":  len(results) - ckd_count,
            "ckd_rate_pct":  round(100 * ckd_count / max(len(results), 1), 2),
        },
    )


@app.post("/predict/explain", response_model=ExplainResponse, tags=["Prediction"])
def predict_with_explanation(
    patient: LenientPatientInput,
    _: ApiKeyDep,
    use_ai: bool = Query(True, description="Call Claude AI for narrative explanation"),
) -> ExplainResponse:
    t0       = time.perf_counter()
    artifact = get_artifact()
    feat_cols = artifact["feature_cols"]
    all_models = artifact["all_models"]

    label, prob_ckd, model_used, df_proc = _run_inference(
        patient.model_dump(), artifact, None
    )

    # SHAP — prefer XGBoost pipeline
    xgb_name  = next((n for n in ("XGBoost (Tuned)", "XGBoost") if n in all_models), model_used)
    shap_dict = _get_shap(all_models[xgb_name], df_proc, feat_cols)

    # AI explanation
    ai_text = ""
    if use_ai:
        from ckd.ai.claude_explain import explain
        ai_text = explain(
            prediction=label,
            prob_ckd=prob_ckd,
            shap_top=shap_dict,
            patient_context=patient.model_dump(exclude_none=True),
        )

    rid = log_prediction(
        model_name=model_used,
        prediction=label,
        probability_ckd=prob_ckd,
        risk_level=_risk_level(prob_ckd),
        patient=patient.model_dump(exclude_none=True),
        shap_values=shap_dict,
        source="api_explain",
    )
    return ExplainResponse(
        request_id=rid,
        prediction=label,
        probability_ckd=round(prob_ckd, 4),
        probability_not_ckd=round(1 - prob_ckd, 4),
        risk_level=_risk_level(prob_ckd),
        model_used=model_used,
        shap_values=shap_dict,
        ai_explanation=ai_text,
        response_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/validate", tags=["Utilities"])
def validate_input(patient: dict) -> dict[str, Any]:
    """Validate patient data and return any warnings without running prediction."""
    from ckd.data.validator import StrictPatientInput
    from pydantic import ValidationError as PydanticError
    try:
        validated = StrictPatientInput(**patient)
        return {"valid": True, "warnings": [], "cleaned": validated.model_dump(exclude_none=True)}
    except PydanticError as exc:
        return {"valid": False, "warnings": exc.errors(), "cleaned": {}}
