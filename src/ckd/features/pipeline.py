"""
Sklearn preprocessing pipeline built with ColumnTransformer.

Design:
  - Numeric columns  → median imputation → StandardScaler
  - Categorical cols → most-frequent imputation → OrdinalEncoder
  - Mixed-type cols  → same as numeric (already cleaned to float by loader)

All transformers are fit ONLY on training data (no leakage).
The fitted pipeline is serialised as part of the end-to-end model artifact.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ckd.config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES,
    MIXED_STR_FEATURES,
    NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)


def _numeric_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])


def _categorical_pipe(cols: list[str]) -> Pipeline:
    # Build categories list only for the cols actually present
    categories = [CATEGORICAL_VALUES.get(col, "auto") for col in cols]
    return Pipeline([
        ("impute",  SimpleImputer(strategy="most_frequent")),
        ("encode",  OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the given feature columns.
    Adapts to which columns are actually present (guards against missing columns).
    """
    num_present  = [c for c in NUMERIC_FEATURES  + MIXED_STR_FEATURES if c in feature_cols]
    cat_present  = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if num_present:
        transformers.append(("num", _numeric_pipe(), num_present))
    if cat_present:
        transformers.append(("cat", _categorical_pipe(cat_present), cat_present))

    logger.debug("Preprocessor: %d numeric, %d categorical", len(num_present), len(cat_present))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return ordered feature names after ColumnTransformer.fit_transform."""
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        names.extend(cols)
    return names
