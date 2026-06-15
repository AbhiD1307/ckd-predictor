"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/ to path so tests can import `ckd.*` without installing
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ── Tiny synthetic CKD-like dataframe for unit tests ──────────────────────
N_CKD     = 30
N_NOTCKD  = 20
RNG       = np.random.default_rng(42)


def _make_synthetic(n_ckd: int = N_CKD, n_notckd: int = N_NOTCKD) -> pd.DataFrame:
    n = n_ckd + n_notckd
    df = pd.DataFrame({
        "age":   RNG.uniform(20, 80, n),
        "bp":    RNG.uniform(60, 140, n),
        "sg":    RNG.choice([1.005, 1.010, 1.015, 1.020, 1.025], n),
        "al":    RNG.integers(0, 5, n).astype(float),
        "su":    RNG.integers(0, 4, n).astype(float),
        "bgr":   RNG.uniform(70, 250, n),
        "bu":    RNG.uniform(10, 120, n),
        "sc":    RNG.uniform(0.4, 5.0, n),
        "sod":   RNG.uniform(120, 155, n),
        "pot":   RNG.uniform(3.0, 7.0, n),
        "hemo":  RNG.uniform(6.0, 17.5, n),
        "pcv":   RNG.uniform(20, 52, n),
        "wc":    RNG.uniform(3000, 15000, n),
        "rc":    RNG.uniform(2.4, 6.0, n),
        "rbc":   RNG.choice(["normal", "abnormal"], n),
        "pc":    RNG.choice(["normal", "abnormal"], n),
        "pcc":   RNG.choice(["notpresent", "present"], n),
        "ba":    RNG.choice(["notpresent", "present"], n),
        "htn":   RNG.choice(["no", "yes"], n),
        "dm":    RNG.choice(["no", "yes"], n),
        "cad":   RNG.choice(["no", "yes"], n),
        "appet": RNG.choice(["good", "poor"], n),
        "pe":    RNG.choice(["no", "yes"], n),
        "ane":   RNG.choice(["no", "yes"], n),
        "classification": ["ckd"] * n_ckd + ["notckd"] * n_notckd,
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic()


@pytest.fixture(scope="session")
def synthetic_Xy(synthetic_df):
    from ckd.data.loader import clean
    return clean(synthetic_df)
