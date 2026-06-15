"""
Streamlit dashboard for CKD Prediction.

Tabs:
  1. Predict       — patient form → prediction + gauge + SHAP + Claude AI
  2. Model Metrics — CV results, test-set tables, ROC, calibration plots
  3. Data Explorer — dataset overview, distributions, missing-value heatmap
  4. About         — project info, architecture diagram

Run:
    streamlit run src/ckd/app/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running directly without installing the package
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ckd.config import (
    ANTHROPIC_MODEL,
    ARTIFACT_METRICS,
    ARTIFACT_PIPELINE,
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES,
    MIXED_STR_FEATURES,
    NUMERIC_FEATURES,
    RAW_CSV,
    RANDOM_STATE,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Predictor · Abhishek Deshmukh",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifact():
    path = ARTIFACTS_DIR / ARTIFACT_PIPELINE
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = ARTIFACTS_DIR / ARTIFACT_METRICS
    if not path.exists():
        return {}
    return json.load(open(path))

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RAW_CSV)

artifact  = load_artifact()
metrics   = load_metrics()
df_raw    = load_raw_data()

MODEL_READY = artifact is not None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/UW_Bothell_logo.png/200px-UW_Bothell_logo.png",
             width=120)
    st.markdown("## CKD Predictor")
    st.caption("Abhishek Ashok Deshmukh\nCSS 581 · UW Bothell")
    st.divider()
    if MODEL_READY:
        st.success(f"Model loaded: **{artifact['model_name']}**")
    else:
        st.warning("No model found. Run `python scripts/train.py` first.")
    st.divider()
    api_key = st.text_input("🔑 Anthropic API Key (for AI explanations)",
                             value=os.getenv("ANTHROPIC_API_KEY", ""),
                             type="password",
                             help="Get yours at console.anthropic.com")
    st.caption(f"AI model: `{ANTHROPIC_MODEL}`")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_explore, tab_about = st.tabs(
    ["🔍 Predict", "📊 Metrics", "🗂 Data Explorer", "ℹ️ About"]
)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _risk_color(p: float) -> str:
    return "#d62728" if p >= 0.7 else "#ff7f0e" if p >= 0.4 else "#2ca02c"

def _risk_label(p: float) -> str:
    return "🔴 High" if p >= 0.7 else "🟠 Medium" if p >= 0.4 else "🟢 Low"

def _render_gauge(prob_ckd: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.15, 1.4); ax.axis("off")
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#e8e8e8", linewidth=20, solid_capstyle="round")
    fill = np.linspace(np.pi, np.pi - prob_ckd * np.pi, 300)
    color = _risk_color(prob_ckd)
    ax.plot(np.cos(fill), np.sin(fill), color=color, linewidth=20, solid_capstyle="round")
    angle = np.pi - prob_ckd * np.pi
    ax.annotate("", xy=(0.78 * np.cos(angle), 0.78 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=2.5))
    ax.add_patch(plt.Circle((0, 0), 0.09, color="#222", zorder=5))
    ax.text(0, 0.12, f"{prob_ckd*100:.1f}%", ha="center", va="bottom",
            fontsize=20, fontweight="bold", color=color)
    ax.text(-1.2, -0.08, "0%", ha="center", fontsize=9, color="#888")
    ax.text( 1.2, -0.08, "100%", ha="center", fontsize=9, color="#888")
    ax.text(0, 1.25, "CKD Risk Probability", ha="center", fontsize=10, color="#444")
    plt.tight_layout(pad=0.1)
    return fig

def _render_waterfall(shap_dict: dict[str, float]) -> plt.Figure:
    pairs = sorted(shap_dict.items(), key=lambda t: abs(t[1]), reverse=True)[:12]
    names, vals = zip(*pairs)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("SHAP value  (impact on CKD log-odds)")
    ax.set_title("Feature Impact — XGBoost SHAP", fontsize=12, fontweight="bold")
    red_p  = mpatches.Patch(color="#d62728", label="Increases CKD risk")
    blue_p = mpatches.Patch(color="#1f77b4", label="Decreases CKD risk")
    ax.legend(handles=[red_p, blue_p], fontsize=9)
    plt.tight_layout()
    return fig

def _preprocess_and_predict(sample: dict) -> tuple[str, float, pd.DataFrame]:
    if not MODEL_READY:
        st.error("Model not loaded.")
        st.stop()
    le         = artifact["label_encoder"]
    feat_cols  = artifact["feature_cols"]
    all_models = artifact["all_models"]
    model_name = artifact["model_name"]
    model      = all_models[model_name]

    df = pd.DataFrame([sample])
    for col in feat_cols:
        if col not in df.columns:
            df[col] = np.nan

    pred_enc  = model.predict(df[feat_cols])[0]
    proba     = model.predict_proba(df[feat_cols])[0]
    label     = le.inverse_transform([pred_enc])[0]
    ckd_idx   = list(le.classes_).index("ckd")
    prob_ckd  = float(proba[ckd_idx])
    return label, prob_ckd, df[feat_cols]

def _get_shap(df_proc: pd.DataFrame, feat_cols: list[str]) -> dict[str, float]:
    all_models  = artifact["all_models"]
    xgb_name    = next((n for n in ("XGBoost (Tuned)", "XGBoost") if n in all_models), None)
    if xgb_name is None:
        return {}
    try:
        from ckd.explain.shap_explain import local_values
        return local_values(all_models[xgb_name], df_proc, feat_cols)
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Patient Clinical Input")
    st.info(
        "Enter patient values below and click **Run Prediction**. "
        "Empty fields default to training-set median/mode."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demographics & Vitals**")
        age  = st.number_input("Age (years)", 1,   100, 50)
        bp   = st.number_input("Blood Pressure (mm Hg)", 50, 200, 80)
        sg   = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025], index=2)
        al   = st.selectbox("Albumin (0–5)", [0,1,2,3,4,5])
        su   = st.selectbox("Sugar (0–5)",   [0,1,2,3,4,5])
    with c2:
        st.markdown("**Blood & Urine Tests**")
        bgr  = st.number_input("Blood Glucose Random (mgs/dl)", 22, 490, 120)
        bu   = st.number_input("Blood Urea (mgs/dl)", 1, 400, 36)
        sc   = st.number_input("Serum Creatinine (mgs/dl)", 0.4, 80.0, 1.2, step=0.1)
        sod  = st.number_input("Sodium (mEq/L)", 100, 165, 137)
        pot  = st.number_input("Potassium (mEq/L)", 2.5, 50.0, 4.5, step=0.1)
        hemo = st.number_input("Hemoglobin (gms)", 3.0, 18.0, 13.0, step=0.1)
        pcv  = st.number_input("Packed Cell Volume", 9, 55, 40)
        wc   = st.number_input("WBC Count (cells/cumm)", 2000, 20000, 7500)
        rc   = st.number_input("RBC Count (millions/cmm)", 2.0, 7.0, 4.8, step=0.1)
    with c3:
        st.markdown("**Categorical / Comorbidities**")
        rbc   = st.selectbox("Red Blood Cells",          ["normal","abnormal"])
        pc    = st.selectbox("Pus Cell",                 ["normal","abnormal"])
        pcc   = st.selectbox("Pus Cell Clumps",          ["notpresent","present"])
        ba    = st.selectbox("Bacteria",                  ["notpresent","present"])
        htn   = st.selectbox("Hypertension",              ["no","yes"])
        dm    = st.selectbox("Diabetes Mellitus",         ["no","yes"])
        cad   = st.selectbox("Coronary Artery Disease",   ["no","yes"])
        appet = st.selectbox("Appetite",                  ["good","poor"])
        pe    = st.selectbox("Pedal Edema",               ["no","yes"])
        ane   = st.selectbox("Anemia",                    ["no","yes"])

    run = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    if run:
        sample = dict(
            age=age, bp=bp, sg=sg, al=al, su=su,
            bgr=bgr, bu=bu, sc=sc, sod=sod, pot=pot, hemo=hemo,
            pcv=pcv, wc=wc, rc=rc,
            rbc=rbc, pc=pc, pcc=pcc, ba=ba,
            htn=htn, dm=dm, cad=cad, appet=appet, pe=pe, ane=ane,
        )
        with st.spinner("Running prediction…"):
            label, prob_ckd, df_proc = _preprocess_and_predict(sample)
            feat_cols = artifact["feature_cols"]

        st.divider()
        rc_col, gauge_col = st.columns([1.5, 1])

        with rc_col:
            if label == "ckd":
                st.error("### ⚠️  CKD Positive")
            else:
                st.success("### ✅  No CKD Detected")

            m1, m2, m3 = st.columns(3)
            m1.metric("CKD Probability",     f"{prob_ckd*100:.1f}%")
            m2.metric("Not-CKD Probability", f"{(1-prob_ckd)*100:.1f}%")
            m3.metric("Risk Level", _risk_label(prob_ckd))
            st.caption(f"Model: **{artifact['model_name']}**")

        with gauge_col:
            fig_g = _render_gauge(prob_ckd)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

        # ── SHAP ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔬 SHAP Feature Impact")
        with st.spinner("Computing SHAP values…"):
            shap_dict = _get_shap(df_proc, feat_cols)

        if shap_dict:
            fig_s = _render_waterfall(shap_dict)
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)
        else:
            st.info("SHAP not available for the selected model.")

        # ── Per-model comparison ───────────────────────────────────────────
        st.divider()
        st.subheader("🤖 Per-Model Comparison")
        le        = artifact["label_encoder"]
        ckd_idx   = list(le.classes_).index("ckd")
        rows = []
        for mname, mdl in artifact["all_models"].items():
            try:
                p_enc = mdl.predict(df_proc)[0]
                proba = mdl.predict_proba(df_proc)[0]
                lbl   = le.inverse_transform([p_enc])[0]
                rows.append({"Model": mname, "Prediction": lbl.upper(),
                              "CKD Prob %": f"{proba[ckd_idx]*100:.1f}"})
            except Exception:
                pass
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # ── Claude AI explanation ──────────────────────────────────────────
        st.divider()
        st.subheader("🧠 Claude AI Clinical Explanation")
        if not api_key:
            st.info("Add your Anthropic API key in the sidebar to enable AI explanations.")
        else:
            with st.spinner(f"Asking {ANTHROPIC_MODEL}…"):
                from ckd.ai.claude_explain import explain
                ai_text = explain(
                    prediction=label,
                    prob_ckd=prob_ckd,
                    shap_top=shap_dict,
                    patient_context={k: v for k, v in sample.items()},
                    api_key=api_key,
                )
            st.markdown(
                f"<div style='background:#f0f4ff;padding:1.1em 1.4em;border-radius:8px;"
                f"border-left:4px solid #4c78a8;font-size:0.97em'>{ai_text}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Generated by {ANTHROPIC_MODEL} · for educational use only")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICS
# ══════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.subheader("Model Performance")

    if not metrics:
        st.warning("No metrics.json found. Run training first.")
    else:
        st.markdown(f"**Best model:** `{metrics.get('best_model','—')}`")
        st.markdown(f"Random state: `{metrics.get('random_state')}` | CV folds: `{metrics.get('cv_folds')}`")

        st.markdown("#### Cross-Validation Results (training set)")
        cv_df = pd.DataFrame(metrics.get("cv_results", {})).T
        if not cv_df.empty:
            st.dataframe(cv_df.style.highlight_max(axis=0, color="#c8f7c5"), use_container_width=True)

        st.markdown("#### Hold-Out Test Results (20%)")
        test_df = pd.DataFrame(metrics.get("test_results", {})).T
        if not test_df.empty:
            st.dataframe(test_df.style.highlight_max(axis=0, color="#c8f7c5"), use_container_width=True)

        st.markdown("#### Best Hyperparameters")
        hp_col1, hp_col2, hp_col3 = st.columns(3)
        with hp_col1:
            st.json(metrics.get("rf_best_params", {}), expanded=False)
            st.caption("Random Forest")
        with hp_col2:
            st.json(metrics.get("xgb_best_params", {}), expanded=False)
            st.caption("XGBoost")
        with hp_col3:
            st.json(metrics.get("svm_best_params", {}), expanded=False)
            st.caption("SVM")

    st.divider()
    st.markdown("#### Saved Report Images")
    from ckd.config import REPORTS_DIR
    for img_name in ["roc_curves.png", "calibration.png"]:
        img_path = REPORTS_DIR / img_name
        if img_path.exists():
            st.image(str(img_path), caption=img_name, use_container_width=True)

    shap_img = _ROOT / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="SHAP Global Summary (XGBoost)", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("UCI CKD Dataset Explorer")

    if df_raw.empty:
        st.warning("Dataset not found.")
    else:
        st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown("#### Missing Values")
        import seaborn as sns
        missing = df_raw.isnull().mean().sort_values(ascending=False)
        fig_mv, ax = plt.subplots(figsize=(10, 3))
        missing.plot(kind="bar", ax=ax, color="#4c78a8")
        ax.set_ylabel("Missing fraction")
        ax.set_title("Missing Value Rate per Column")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_mv, use_container_width=True)
        plt.close(fig_mv)

        st.markdown("#### Target Class Distribution")
        if "classification" in df_raw.columns:
            counts = df_raw["classification"].str.strip().value_counts()
            fig_t, ax2 = plt.subplots(figsize=(4, 3))
            counts.plot(kind="bar", ax=ax2, color=["#d62728","#2ca02c"])
            ax2.set_ylabel("Count"); ax2.set_title("CKD vs Not-CKD")
            plt.tight_layout()
            st.pyplot(fig_t, use_container_width=False)
            plt.close(fig_t)

        st.markdown("#### Numeric Feature Distributions")
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()[:8]
        fig_dist, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax, col in zip(axes.flat, num_cols):
            df_raw[col].dropna().hist(ax=ax, bins=20, color="#4c78a8", edgecolor="white")
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## About This Project

| | |
|---|---|
| **Author** | Abhishek Ashok Deshmukh |
| **Course** | CSS 581 – Machine Learning · UW Bothell |
| **Dataset** | UCI Chronic Kidney Disease (400 patients, 25 features) |
| **GitHub** | *(add your repo link here)* |

---

### Software Architecture

```
ckd-predictor/
├── src/ckd/
│   ├── config.py            ← central config (paths, hyperparameters)
│   ├── data/loader.py       ← data loading + cleaning
│   ├── features/pipeline.py ← sklearn ColumnTransformer preprocessing
│   ├── models/
│   │   ├── train.py         ← multi-model training + GridSearchCV
│   │   └── evaluate.py      ← metrics, ROC, calibration plots
│   ├── explain/
│   │   └── shap_explain.py  ← SHAP global + local explanations
│   ├── ai/
│   │   └── claude_explain.py← Claude AI clinical narrative
│   ├── api/
│   │   └── server.py        ← FastAPI REST API (/predict, /explain)
│   └── app/
│       └── dashboard.py     ← this Streamlit dashboard
├── scripts/train.py          ← CLI training entry point
├── tests/                    ← pytest unit tests
├── Dockerfile
└── .github/workflows/ci.yml  ← GitHub Actions CI
```

### ML Pipeline

| Step | Detail |
|------|--------|
| Preprocessing | Median imputation (numeric) · Most-frequent (categorical) · StandardScaler |
| Leakage prevention | All transforms fit inside CV folds (no test-set contamination) |
| Models | Logistic Regression · Random Forest · SVM · XGBoost |
| Tuning | GridSearchCV with 5-fold StratifiedKFold |
| Final model | Soft Voting Ensemble (Tuned RF + Tuned XGBoost) |
| Explainability | SHAP TreeExplainer on XGBoost |
| AI layer | Claude claude-sonnet-4-6 clinical narrative via Anthropic API |
| API | FastAPI with Pydantic v2 validation |

> ⚠️ **Disclaimer:** This tool is for educational purposes only and must not replace clinical diagnosis.
""")
