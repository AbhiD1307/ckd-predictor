"""
CKD Prediction Dashboard — Abhishek Ashok Deshmukh | CSS 581 ML
Streamlit web app for Chronic Kidney Disease prediction using saved artifacts.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_PATH     = os.path.join(os.path.dirname(__file__), "Chronic_Kidney_Disease.csv")

st.set_page_config(
    page_title="CKD Predictor",
    page_icon="🩺",
    layout="wide",
)

# ──────────────────────────────────────────────
# Load artifacts (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    preproc   = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessing.joblib"))
    ensemble  = joblib.load(os.path.join(ARTIFACTS_DIR, "Ensemble_Voting.joblib"))
    xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, "XGBoost.joblib"))
    rf_model  = joblib.load(os.path.join(ARTIFACTS_DIR, "Random_Forest_Tuned.joblib"))
    metrics   = json.load(open(os.path.join(ARTIFACTS_DIR, "metrics.json")))
    df_raw    = pd.read_csv(DATA_PATH)
    return preproc, ensemble, xgb_model, rf_model, metrics, df_raw

preproc, ensemble, xgb_model, rf_model, metrics, df_raw = load_artifacts()
scaler          = preproc["scaler"]
label_encoders  = preproc["label_encoders"]
y_le            = preproc["y_le"]

NUM_COLS = metrics["num_cols"]
CAT_COLS = metrics["cat_cols"]
ALL_COLS = metrics["columns"]

# ──────────────────────────────────────────────
# Preprocessing helper
# ──────────────────────────────────────────────
def preprocess_input(sample: dict) -> pd.DataFrame:
    df_in = pd.DataFrame([sample])
    X_stats = df_raw.drop("classification", axis=1)

    for col in df_in.columns:
        if df_in[col].dtype == object:
            mode_val = X_stats[col].mode()[0] if col in X_stats.columns else df_in[col].mode()[0]
            df_in[col] = df_in[col].fillna(mode_val)
        else:
            med = X_stats[col].median() if col in X_stats.columns else df_in[col].median()
            df_in[col] = df_in[col].fillna(med)

    for col, le in label_encoders.items():
        if col in df_in.columns:
            val = str(df_in[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df_in[col] = le.transform([val])

    for col in ALL_COLS:
        if col not in df_in.columns:
            df_in[col] = 0

    df_in[NUM_COLS] = scaler.transform(df_in[NUM_COLS])
    return df_in[ALL_COLS]

# ──────────────────────────────────────────────
# Gauge chart
# ──────────────────────────────────────────────
def render_gauge(prob_ckd: float):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.1, 1.3)
    ax.axis("off")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=18, solid_capstyle="round")

    # Colored fill arc
    fill_theta = np.linspace(np.pi, np.pi - prob_ckd * np.pi, 200)
    color = "#d62728" if prob_ckd > 0.6 else "#ff7f0e" if prob_ckd > 0.3 else "#2ca02c"
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=18, solid_capstyle="round")

    # Needle
    angle = np.pi - prob_ckd * np.pi
    ax.annotate("", xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.add_patch(plt.Circle((0, 0), 0.08, color="black", zorder=5))

    ax.text(0, 0.15, f"{prob_ckd*100:.1f}%", ha="center", va="bottom", fontsize=18, fontweight="bold", color=color)
    ax.text(-1.1, -0.05, "0%", ha="center", fontsize=9, color="#555")
    ax.text( 1.1, -0.05, "100%", ha="center", fontsize=9, color="#555")
    ax.text(0, 1.15, "CKD Risk", ha="center", fontsize=11, color="#333")
    plt.tight_layout(pad=0.1)
    return fig

# ──────────────────────────────────────────────
# SHAP waterfall for one sample
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Computing SHAP…")
def get_xgb_explainer():
    return shap.TreeExplainer(xgb_model)

def shap_waterfall(X_proc: pd.DataFrame):
    explainer   = get_xgb_explainer()
    shap_vals   = explainer.shap_values(X_proc)
    base_val    = explainer.expected_value

    # For binary XGB the output can be 1-D
    if isinstance(shap_vals, list):
        sv = shap_vals[0][0]
        bv = base_val[0]
    else:
        sv = shap_vals[0]
        bv = base_val

    # Manual waterfall bar chart (shap.plots.waterfall needs display env)
    feature_names = X_proc.columns.tolist()
    pairs = sorted(zip(np.abs(sv), sv, feature_names), reverse=True)[:12]
    _, vals, names = zip(*pairs)

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("Feature Impact on This Prediction (XGBoost SHAP)", fontsize=12)
    red_p  = mpatches.Patch(color="#d62728", label="Pushes toward CKD")
    blue_p = mpatches.Patch(color="#1f77b4", label="Pushes toward Not-CKD")
    ax.legend(handles=[red_p, blue_p], fontsize=9)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────
# ── UI ──
# ──────────────────────────────────────────────
st.title("🩺 Chronic Kidney Disease (CKD) Prediction Dashboard")
st.caption("Abhishek Ashok Deshmukh · CSS 581 Machine Learning · UW Bothell")
st.markdown("---")

tab_predict, tab_metrics, tab_about = st.tabs(["🔍 Predict", "📊 Model Metrics", "ℹ️ About"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    st.subheader("Enter Patient Clinical Values")
    st.info("Fill in the patient data below and press **Run Prediction**. Fields left blank use training-set median/mode defaults.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Vitals**")
        age = st.number_input("Age (years)",         min_value=1,   max_value=100, value=50)
        bp  = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=180, value=80)
        sg  = st.selectbox("Specific Gravity",       [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
        al  = st.selectbox("Albumin (0–5)",          [0, 1, 2, 3, 4, 5], index=0)
        su  = st.selectbox("Sugar (0–5)",            [0, 1, 2, 3, 4, 5], index=0)

    with col2:
        st.markdown("**Blood / Urine Tests**")
        bgr = st.number_input("Blood Glucose Random (mgs/dl)", min_value=22,  max_value=490, value=120)
        bu  = st.number_input("Blood Urea (mgs/dl)",           min_value=1,   max_value=391, value=36)
        sc  = st.number_input("Serum Creatinine (mgs/dl)",     min_value=0.4, max_value=76.0, value=1.2, step=0.1)
        sod = st.number_input("Sodium (mEq/L)",                min_value=104, max_value=163, value=137)
        pot = st.number_input("Potassium (mEq/L)",             min_value=2.5, max_value=47.0, value=4.5, step=0.1)
        hemo= st.number_input("Hemoglobin (gms)",              min_value=3.1, max_value=17.8, value=13.0, step=0.1)

    with col3:
        st.markdown("**Categorical / Comorbidities**")
        rbc   = st.selectbox("Red Blood Cells",        ["normal", "abnormal"])
        pc    = st.selectbox("Pus Cell",               ["normal", "abnormal"])
        pcc   = st.selectbox("Pus Cell Clumps",        ["notpresent", "present"])
        ba    = st.selectbox("Bacteria",               ["notpresent", "present"])
        htn   = st.selectbox("Hypertension",           ["no", "yes"])
        dm    = st.selectbox("Diabetes Mellitus",      ["no", "yes"])
        cad   = st.selectbox("Coronary Artery Disease",["no", "yes"])
        appet = st.selectbox("Appetite",               ["good", "poor"])
        pe    = st.selectbox("Pedal Edema",            ["no", "yes"])
        ane   = st.selectbox("Anemia",                 ["no", "yes"])

    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    if run_btn:
        sample = {
            "id": 0,
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot, "hemo": hemo,
            "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
            "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane,
            "pcv": None, "wc": None, "rc": None,
        }

        X_proc = preprocess_input(sample)
        pred_enc = ensemble.predict(X_proc)[0]
        prob_arr  = ensemble.predict_proba(X_proc)[0]
        prob_ckd  = float(prob_arr[0])   # class 0 = ckd
        label     = y_le.inverse_transform([pred_enc])[0]

        st.markdown("---")
        res_col, gauge_col = st.columns([1.3, 1])

        with res_col:
            if label == "ckd":
                st.error(f"### ⚠️ Prediction: **CKD Positive**")
            else:
                st.success(f"### ✅ Prediction: **No CKD Detected**")

            p_col1, p_col2 = st.columns(2)
            p_col1.metric("CKD Probability",     f"{prob_ckd*100:.1f}%")
            p_col2.metric("Not-CKD Probability", f"{(1-prob_ckd)*100:.1f}%")

            risk = "High" if prob_ckd > 0.6 else "Medium" if prob_ckd > 0.3 else "Low"
            color_map = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
            st.markdown(f"**Risk Level:** {color_map[risk]} {risk}")

            st.markdown("**Model used:** Soft Voting Ensemble (Random Forest + SVM, tuned)")

        with gauge_col:
            fig_gauge = render_gauge(prob_ckd)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

        st.markdown("---")
        st.subheader("🔬 Feature Impact Explanation (XGBoost SHAP)")
        st.caption("Red bars push the prediction toward CKD; blue bars push away from CKD.")
        fig_shap = shap_waterfall(X_proc)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        # Per-model comparison
        st.markdown("---")
        st.subheader("🤖 Per-Model Comparison")
        model_preds = {
            "Ensemble (Voting)": (ensemble, prob_ckd),
            "XGBoost":           (xgb_model, float(xgb_model.predict_proba(X_proc)[0][0])),
            "Random Forest":     (rf_model,  float(rf_model.predict_proba(X_proc)[0][0])),
        }
        comp_rows = []
        for mname, (mdl, p_ckd) in model_preds.items():
            pred_l = y_le.inverse_transform([mdl.predict(X_proc)[0]])[0]
            comp_rows.append({"Model": mname, "Prediction": pred_l.upper(), "CKD Prob %": f"{p_ckd*100:.1f}"})
        st.dataframe(pd.DataFrame(comp_rows).set_index("Model"), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — METRICS
# ══════════════════════════════════════════════
with tab_metrics:
    st.subheader("Model Performance on Hold-Out Test Set (20%)")

    baseline = pd.DataFrame(metrics["baseline_results"]).T.drop(columns=["TrainTimeSec"], errors="ignore")
    tuned    = pd.DataFrame(metrics["tuned_results"]).T.drop(columns=["TrainTimeSec"], errors="ignore")

    st.markdown("#### Baseline Models")
    st.dataframe(baseline.style.highlight_max(axis=0, color="#c8f7c5"), use_container_width=True)

    st.markdown("#### Tuned + Ensemble Models")
    st.dataframe(tuned.style.highlight_max(axis=0, color="#c8f7c5"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### SHAP Global Feature Importance")
    shap_img = os.path.join(os.path.dirname(__file__), "shap_summary.png")
    if os.path.exists(shap_img):
        st.image(shap_img, caption="SHAP Summary Plot (XGBoost — Training Set)", use_container_width=True)

    st.markdown("#### XGBoost Confusion Matrix")
    cm_img = os.path.join(os.path.dirname(__file__), "confusion_matrix_xgb.png")
    if os.path.exists(cm_img):
        st.image(cm_img, caption="XGBoost Confusion Matrix", use_container_width=False)

# ══════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════
with tab_about:
    st.markdown("""
## About This Project

**Author:** Abhishek Ashok Deshmukh
**Course:** CSS 581 – Machine Learning | UW Bothell
**Dataset:** UCI Chronic Kidney Disease Dataset (400 patients, 25 features)

### Pipeline Summary
| Step | Detail |
|------|--------|
| Preprocessing | Median / mode imputation (fit on train only), StandardScaler, LabelEncoder |
| Data Split | 80 / 20 stratified split, `RANDOM_STATE = 42` |
| Baseline Models | Logistic Regression, Random Forest, SVM (RBF) |
| Boosting | XGBoost (300 estimators, lr=0.05, depth=4) |
| Hyperparameter Tuning | GridSearchCV with 5-fold StratifiedKFold |
| Final Model | Soft Voting Ensemble (Tuned RF + Tuned SVM) |
| Explainability | SHAP TreeExplainer on XGBoost |

### Key Results
- **Ensemble Accuracy:** 100% on test set
- **Ensemble F1 (CKD class):** 1.000
- **Top predictive features:** hemoglobin, specific gravity, serum creatinine, albumin, sodium

### Clinical Notes
> This dashboard is for **educational purposes only** and should not be used for actual clinical diagnosis.
""")
