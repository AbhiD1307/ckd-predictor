# ── Stage 1: base ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for numpy/sklearn/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: deps ──────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 3: app ───────────────────────────────────────────────────────────
FROM deps AS app

COPY src/       /app/src/
COPY scripts/   /app/scripts/
COPY artifacts/ /app/artifacts/
COPY Chronic_Kidney_Disease.csv /app/

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default: run FastAPI server
EXPOSE 8000
CMD ["uvicorn", "ckd.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: Streamlit dashboard
# CMD ["streamlit", "run", "src/ckd/app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
