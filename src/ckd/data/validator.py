"""
Input validation layer — Pydantic v2 models that enforce clinical value ranges
before data reaches the ML pipeline.

Two validation modes
--------------------
StrictPatientInput   — hard limits, raises ValidationError on bad values
LenientPatientInput  — soft warnings only, coerces to None for out-of-range

Also provides:
  validate_batch(records)  → (valid_list, error_list)
  ValidationReport         — summary of batch validation results
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ── Clinical reference ranges ──────────────────────────────────────────────
_RANGES: dict[str, tuple[float, float]] = {
    "age":  (1,   120),
    "bp":   (40,  200),
    "sg":   (1.000, 1.030),
    "al":   (0,   5),
    "su":   (0,   5),
    "bgr":  (20,  500),
    "bu":   (1,   400),
    "sc":   (0.1, 80),
    "sod":  (100, 170),
    "pot":  (2.0, 50),
    "hemo": (2.0, 20),
    "pcv":  (9,   65),
    "wc":   (1000, 25000),
    "rc":   (1.5, 8.0),
}

_CATEGORICAL_VALID: dict[str, set[str]] = {
    "rbc":   {"normal", "abnormal"},
    "pc":    {"normal", "abnormal"},
    "pcc":   {"notpresent", "present"},
    "ba":    {"notpresent", "present"},
    "htn":   {"no", "yes"},
    "dm":    {"no", "yes"},
    "cad":   {"no", "yes"},
    "appet": {"good", "poor"},
    "pe":    {"no", "yes"},
    "ane":   {"no", "yes"},
}


# ── Shared base (no field-level constraints — allows lenient override) ─────
class _BasePatientInput(BaseModel):
    """All fields unconstrained — constraints added by subclasses via validators."""

    age:   Optional[float] = Field(None, description="Age in years")
    bp:    Optional[float] = Field(None, description="Blood pressure mm Hg")
    sg:    Optional[float] = Field(None, description="Specific gravity")
    al:    Optional[float] = Field(None, description="Albumin 0–5")
    su:    Optional[float] = Field(None, description="Sugar 0–5")
    bgr:   Optional[float] = Field(None, description="Blood glucose random")
    bu:    Optional[float] = Field(None, description="Blood urea")
    sc:    Optional[float] = Field(None, description="Serum creatinine")
    sod:   Optional[float] = Field(None, description="Sodium mEq/L")
    pot:   Optional[float] = Field(None, description="Potassium mEq/L")
    hemo:  Optional[float] = Field(None, description="Hemoglobin gms")
    pcv:   Optional[float] = Field(None, description="Packed cell volume")
    wc:    Optional[float] = Field(None, description="WBC count")
    rc:    Optional[float] = Field(None, description="RBC count millions")

    rbc:   Optional[str]   = None
    pc:    Optional[str]   = None
    pcc:   Optional[str]   = None
    ba:    Optional[str]   = None
    htn:   Optional[str]   = None
    dm:    Optional[str]   = None
    cad:   Optional[str]   = None
    appet: Optional[str]   = None
    pe:    Optional[str]   = None
    ane:   Optional[str]   = None

    model_config = {"extra": "allow", "str_to_lower": True, "str_strip_whitespace": True}

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# ── Strict model ───────────────────────────────────────────────────────────
class StrictPatientInput(_BasePatientInput):
    """Hard validation — raises on any range or value violation."""

    @model_validator(mode="after")
    def _validate_ranges(self) -> "StrictPatientInput":
        for feat, (lo, hi) in _RANGES.items():
            val = getattr(self, feat, None)
            if val is not None and not (lo <= val <= hi):
                raise ValueError(
                    f"'{feat}'={val} is outside clinical range [{lo}, {hi}]"
                )
        return self

    @model_validator(mode="after")
    def _validate_categoricals(self) -> "StrictPatientInput":
        for col, valid_set in _CATEGORICAL_VALID.items():
            val = getattr(self, col, None)
            if val is not None and val not in valid_set:
                raise ValueError(
                    f"'{col}' must be one of {sorted(valid_set)}, got '{val}'"
                )
        return self

    @model_validator(mode="after")
    def _check_at_least_one_feature(self) -> "StrictPatientInput":
        non_none = [v for v in self.model_dump().values() if v is not None]
        if not non_none:
            raise ValueError("At least one clinical feature must be provided.")
        return self


# ── Lenient model ──────────────────────────────────────────────────────────
class LenientPatientInput(_BasePatientInput):
    """
    Soft validation — logs warnings for out-of-range / unknown values instead of raising.
    Coerces invalid values to None so the pipeline can impute them.
    """

    @model_validator(mode="after")
    def _soft_range_check(self) -> "LenientPatientInput":
        for feat, (lo, hi) in _RANGES.items():
            val = getattr(self, feat, None)
            if val is not None and not (lo <= val <= hi):
                logger.warning(
                    "Feature '%s'=%s outside clinical range [%s, %s] — will be imputed.",
                    feat, val, lo, hi,
                )
                setattr(self, feat, None)
        return self

    @model_validator(mode="after")
    def _soft_categorical_check(self) -> "LenientPatientInput":
        for col, valid_set in _CATEGORICAL_VALID.items():
            val = getattr(self, col, None)
            if val is not None and val not in valid_set:
                logger.warning(
                    "Feature '%s'='%s' not in %s — will be imputed.",
                    col, val, sorted(valid_set),
                )
                setattr(self, col, None)
        return self


# ── Batch validation ───────────────────────────────────────────────────────
class ValidationError(BaseModel):
    row_index: int
    error:     str
    raw_data:  dict[str, Any]


class ValidationReport(BaseModel):
    total:    int
    valid:    int
    invalid:  int
    errors:   list[ValidationError]

    def summary(self) -> str:
        return (
            f"Validation: {self.valid}/{self.total} valid "
            f"({self.invalid} rejected)\n"
            + "\n".join(
                f"  Row {e.row_index}: {e.error}" for e in self.errors[:10]
            )
        )


def validate_batch(
    records: list[dict[str, Any]],
    strict: bool = True,
) -> tuple[list[StrictPatientInput], ValidationReport]:
    """
    Validate a list of raw patient dicts.

    Returns
    -------
    (valid_inputs, report)
      valid_inputs — list of validated PatientInput objects
      report       — full ValidationReport with error details
    """
    Model = StrictPatientInput if strict else LenientPatientInput
    valid: list[StrictPatientInput] = []
    errors: list[ValidationError] = []

    for i, record in enumerate(records):
        try:
            valid.append(Model(**record))
        except Exception as exc:
            errors.append(ValidationError(row_index=i, error=str(exc), raw_data=record))

    report = ValidationReport(
        total=len(records), valid=len(valid), invalid=len(errors), errors=errors
    )
    if errors:
        logger.warning("Batch validation: %d/%d records failed", len(errors), len(records))
    return valid, report
