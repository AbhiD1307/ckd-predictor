# CKD Predictor — Production ML Package

> **Chronic Kidney Disease Prediction using Machine Learning**

| | |
|---|---|
| **Author** | Abhishek Ashok Deshmukh |
| **Course** | CSS 581 Machine Learning · Autumn 2025 |
| **University** | University of Washington Bothell |
| **Dataset** | UCI Chronic Kidney Disease (400 patients · 25 features) |

[![CI](https://github.com/YOUR_USERNAME/ckd-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ckd-predictor/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen)]()

---

## What This Project Is

A **production-grade machine learning software package** for predicting Chronic Kidney Disease (CKD) from clinical lab values.

This is **not a Jupyter notebook**. It is a fully structured, deployable Python package built to industry standards — with a REST API, web dashboard, AI explanations, model registry, drift monitoring, and a full test suite.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **ML & Data** | scikit-learn · XGBoost · SHAP · pandas · numpy |
| **REST API** | FastAPI · Pydantic v2 · Uvicorn · API key auth |
| **Dashboard** | Streamlit (4 tabs: Predict · Metrics · Data Explorer · About) |
| **AI Explanations** | Claude `claude-sonnet-4-6` via Anthropic API |
| **Experiment Tracking** | MLflow |
| **Model Registry** | Custom versioned registry (save / promote / rollback) |
| **Prediction Database** | SQLite — logs every API call with SHAP values |
| **Drift Detection** | PSI-based feature drift monitoring |
| **Input Validation** | Pydantic v2 strict + lenient clinical range validation |
| **Testing** | pytest · 57 unit tests · httpx TestClient |
| **CI/CD** | GitHub Actions (test · lint · train smoke test) |
| **Containerisation** | Docker multi-stage · docker-compose |
| **Dev Tooling** | Makefile · pre-commit · ruff |

---

## Project Structure

```
ckd-predictor/
│
├── src/ckd/                          ← Python package
│   ├── config.py                     ← paths, hyperparameters, feature schemas
│   ├── data/
│   │   ├── loader.py                 ← load + clean UCI CKD CSV
│   │   └── validator.py              ← Pydantic v2 strict/lenient validation
│   ├── features/
│   │   └── pipeline.py               ← sklearn ColumnTransformer (no data leakage)
│   ├── models/
│   │   ├── train.py                  ← LR, RF, SVM, XGBoost + GridSearchCV + Ensemble
│   │   ├── evaluate.py               ← metrics, ROC, confusion matrix, calibration
│   │   └── registry.py               ← versioned model registry (save/load/promote)
│   ├── explain/
│   │   └── shap_explain.py           ← SHAP global summary + local waterfall
│   ├── ai/
│   │   └── claude_explain.py         ← Claude API → clinical narrative explanation
│   ├── api/
│   │   └── server.py                 ← FastAPI REST API (auth, logging, 6 endpoints)
│   ├── app/
│   │   └── dashboard.py              ← Streamlit 4-tab dashboard
│   ├── db/
│   │   └── database.py               ← SQLite prediction logger
│   └── monitoring/
│       ├── drift.py                  ← PSI data drift detection
│       └── mlflow_tracking.py        ← MLflow experiment tracking wrapper
│
├── scripts/
│   ├── train.py                      ← CLI: train + registry + MLflow
│   ├── predict.py                    ← CLI: batch inference from CSV
│   └── check_drift.py                ← CLI: drift report vs. training baseline
│
├── tests/                            ← pytest · 57 tests
│   ├── conftest.py                   ← shared synthetic data fixtures
│   ├── test_data.py                  ← data loading + cleaning (12 tests)
│   ├── test_features.py              ← preprocessing pipeline (6 tests)
│   ├── test_models.py                ← training + evaluation (5 tests)
│   ├── test_db.py                    ← SQLite database (12 tests)
│   ├── test_monitoring.py            ← drift detection + validator (19 tests)
│   └── test_api.py                   ← FastAPI endpoints (7 tests)
│
├── data/
│   └── sample_patients.csv           ← 10 demo patients for batch predict
│
├── artifacts/                        ← saved model artifacts + predictions.db
│   └── registry/                     ← versioned model registry
│
├── reports/                          ← evaluation plots (auto-generated)
│
├── Makefile                          ← all project commands
├── Dockerfile                        ← multi-stage production build
├── docker-compose.yml                ← run API + Dashboard together
├── pyproject.toml                    ← PEP 517 packaging
├── requirements.txt
├── .env.example                      ← environment variable template
├── .gitignore
├── .pre-commit-config.yaml           ← ruff lint + format on every commit
└── .github/workflows/ci.yml          ← GitHub Actions CI pipeline
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/ckd-predictor.git
cd ckd-predictor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and optional CKD_API_KEY
```

### 2. Train Models
```bash
make train
```
This trains Logistic Regression, Random Forest, SVM, XGBoost, and a Soft Voting Ensemble.  
Saves artifacts to `artifacts/` and registers the model in the version registry.

### 3. Run the Dashboard
```bash
make dashboard
# → http://localhost:8501
```

### 4. Run the REST API
```bash
make api
# → http://localhost:8000/docs   (interactive Swagger UI)
```

### 5. Run Tests
```bash
make test          # all 57 tests
make test-unit     # fast unit tests only (no artifact needed)
make test-cov      # with HTML coverage report → htmlcov/index.html
```

### 6. Batch Inference
```bash
make predict INPUT=data/sample_patients.csv OUTPUT=data/results.csv
```

### 7. Docker (one command for everything)
```bash
make docker-build
docker-compose up          # API on :8000 · Dashboard on :8501
```

### 8. MLflow Experiment Tracking
```bash
# Train with tracking enabled
PYTHONPATH=src .venv/bin/python3.13 scripts/train.py --track --run-name "baseline_v1"

# Open MLflow UI
.venv/bin/python3.13 -m mlflow ui --port 5000
# → http://localhost:5000
```

### 9. Drift Detection
```bash
make drift-report          # compare recent predictions to training baseline
```

---

## API Reference

**Base URL:** `http://localhost:8000`  
**Docs:** `http://localhost:8000/docs`  
**Auth:** Pass `X-API-Key: <your-key>` header (set `CKD_API_KEY` in `.env`; leave blank to disable in dev).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check — model loaded, DB ready |
| `GET`  | `/models` | All trained models + test-set metrics |
| `POST` | `/predict` | Single patient prediction |
| `POST` | `/predict/batch` | Up to 500 patients at once |
| `POST` | `/predict/explain` | Prediction + SHAP values + Claude AI narrative |
| `GET`  | `/stats` | Aggregate stats from prediction database |
| `GET`  | `/predictions/history` | Recent prediction log |
| `POST` | `/validate` | Validate patient data without predicting |

### Example: `POST /predict`
**Request:**
```json
{
  "age": 65,
  "bp": 90,
  "sg": 1.005,
  "al": 4,
  "hemo": 7.0,
  "sc": 4.5,
  "htn": "yes",
  "dm": "yes"
}
```
**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction": "ckd",
  "probability_ckd": 0.9742,
  "probability_not_ckd": 0.0258,
  "risk_level": "High",
  "model_used": "Ensemble (Voting)",
  "response_time_ms": 12.4
}
```

### Example: `POST /predict/explain`
Returns the above **plus** `shap_values` (dict of feature impacts) and `ai_explanation` (Claude-generated clinical narrative in plain English).

---

## ML Pipeline

### Models Trained

| Model | Type | Tuning |
|-------|------|--------|
| Logistic Regression | Linear | — |
| Random Forest | Tree Ensemble | GridSearchCV |
| SVM (RBF kernel) | Kernel | GridSearchCV |
| XGBoost | Gradient Boosting | GridSearchCV |
| **Ensemble (Voting)** | **Soft Voting (RF + XGB)** | **Final production model** |

### Preprocessing (no data leakage)

```
Raw CSV
  ↓  load + clean (loader.py)         ← strip noisy labels, fix mixed-type cols
  ↓  train/test split (80/20)         ← stratified, random_state=42
  ↓  ColumnTransformer (pipeline.py)  ← fit ONLY on train set
       ├─ Numeric  → median impute → StandardScaler
       └─ Categorical → mode impute → OrdinalEncoder
  ↓  GridSearchCV (5-fold StratifiedKFold)
  ↓  Soft Voting Ensemble
  ↓  SHAP TreeExplainer
  ↓  Claude AI narrative
```

### Results (hold-out test set · 80/20 split · random_state=42)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| **Ensemble (Voting)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Random Forest (Tuned) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost (Tuned) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression | 0.988 | 1.000 | 0.980 | 0.990 | 0.998 |
| SVM (Tuned) | 0.913 | 0.877 | 1.000 | 0.935 | 0.968 |

---

## Model Registry

```python
from ckd.models.registry import save, load, list_versions, promote, compare, rollback

# Save a new version after training
tag = save(artifact, metrics, notes="XGB depth=6, RF 400 trees")

# List all registered versions
for v in list_versions():
    print(v["version"], v["test_f1"])

# Compare two versions side-by-side
report = compare("v20240101_120000", "v20240102_090000")

# Promote best version to production
promote(tag, stage="production")

# Load the production model
artifact = load("production")

# Roll back if something breaks
rollback("production")
```

---

## Data Drift Detection

```python
from ckd.monitoring.drift import DriftDetector

# Build baseline from training data (run once after training)
detector = DriftDetector.from_training_data(X_train, feature_names)
detector.save()

# Check incoming data against baseline
report = detector.check(X_new_patients)
print(report.summary())

if report.needs_retraining:
    # PSI >= 0.25 on one or more features
    trigger_retrain()
```

PSI interpretation: `< 0.10` stable · `0.10–0.25` moderate · `≥ 0.25` significant shift.

---

## Input Validation

```python
from ckd.data.validator import StrictPatientInput, LenientPatientInput, validate_batch

# Strict — raises ValidationError on bad values
p = StrictPatientInput(age=65, htn="yes", sc=4.5)

# Lenient — coerces out-of-range to None, logs a warning
p = LenientPatientInput(age=999, htn="unknown")   # age → None, htn → None

# Batch validate a list of dicts
valid_records, report = validate_batch(raw_list, strict=True)
print(report.summary())   # "Validation: 48/50 valid (2 rejected)"
```

---

## Makefile Commands

```
make install         Install all dependencies
make install-dev     Install + dev tools (ruff, pre-commit)
make train           Train all models and save artifacts
make train-quick     Train without evaluation plots
make dashboard       Launch Streamlit dashboard  → :8501
make api             Launch FastAPI server        → :8000
make predict         Batch predict from CSV
make test            Run all 57 tests
make test-unit       Fast unit tests only
make test-cov        Tests with HTML coverage report
make lint            Run ruff linter
make format          Auto-format with ruff
make docker-build    Build Docker image
make docker-api      Run API in Docker
make docker-dashboard Run dashboard in Docker
make db-init         Initialise prediction database
make db-stats        Print database statistics
make drift-report    Generate data drift report
make clean           Remove cache files
make clean-artifacts Remove saved models (forces retrain)
make help            Show all commands
```

---

> ⚠️ **Disclaimer:** This software is for educational and research purposes only.
> It is not a medical device and must not be used for actual clinical diagnosis.
