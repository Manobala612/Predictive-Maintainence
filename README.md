# AI-Based Predictive Maintenance System (MLOps)

A production-ready, modular MLOps project for predictive maintenance diagnostics. The system uses a pre-trained Scikit-learn **Random Forest Classifier** trained on the **AI4I 2020 Predictive Maintenance Dataset** to predict machine failure events from live sensor telemetry.

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                       │
│                                                                   │
│  ┌──────────────┐     HTTP/JSON      ┌───────────────────────┐  │
│  │  React SPA   │ ─────────────────► │  FastAPI Backend      │  │
│  │  :5173       │ ◄───────────────── │  :8000                │  │
│  └──────────────┘                    │  /api/v1/predict      │  │
│                                      │  /api/v1/health       │  │
│                                      └──────────┬────────────┘  │
│                                                 │               │
│                                    MLflow SDK   │               │
│                                    (per-pred)   ▼               │
│                                      ┌───────────────────────┐  │
│                                      │  MLflow Server  :5000 │  │
│                                      │  SQLite backend       │  │
│                                      └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
predictive_maintenance/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI (lint + test + build)
├── backend/
│   ├── api/
│   │   └── endpoints.py        # FastAPI routes (/predict, /health)
│   ├── core/
│   │   ├── config.py           # Pydantic settings with .env support
│   │   └── logging_config.py   # Structured stdout logging
│   ├── schemas/
│   │   └── telemetry.py        # Pydantic request/response schemas
│   ├── services/
│   │   └── prediction.py       # RF inference + per-prediction MLflow logging
│   ├── main.py                 # FastAPI app with lifespan handler
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Navigation.jsx  # Header + live API health badge
│   │   ├── pages/
│   │   │   ├── Home.jsx        # Landing page
│   │   │   ├── Prediction.jsx  # Sensor form + results + history
│   │   │   └── About.jsx       # Project info
│   │   ├── services/
│   │   │   └── api.js          # Axios client wrapper
│   │   ├── App.jsx             # Page router + health polling
│   │   └── main.jsx            # Vite entry point
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── model/
│   ├── train_model.py          # Training script with full MLflow tracking
│   ├── requirements.txt        # Training-only dependencies
│   ├── machine_failure_model.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
├── dataset/
│   └── ai4i2020.csv            # Raw dataset (DVC tracked)
├── mlflow/                     # MLflow SQLite DB + run artifacts
├── mlflow_artifacts/           # Local plots generated during training
├── tests/
│   └── test_backend.py         # Pytest unit tests
├── docker/
│   ├── backend.Dockerfile
│   └── frontend.Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | Scikit-learn Random Forest | Binary classification of machine failure |
| Backend API | FastAPI + Pydantic | REST inference service with input validation |
| Frontend UI | React + Vite + Tailwind CSS | Dashboard SPA |
| HTTP Client | Axios | Frontend → Backend communication |
| Experiment Tracking | MLflow | Training metrics + inference audit logging |
| Data Versioning | DVC | Dataset and model artifact versioning |
| Containerisation | Docker + Docker Compose | Unified environment |
| CI/CD | GitHub Actions | Lint, test, and build verification |

---

## Quick Start (Docker Compose)

Requires: **Docker Desktop** installed and running.

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| React Dashboard | http://localhost:5173 |
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## Local Development (Manual)

### Backend

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### MLflow Server (local)

```bash
cd mlflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ENV` | `development` | Application environment mode |
| `BACKEND_PORT` | `8000` | FastAPI exposed port |
| `FRONTEND_PORT` | `5173` | React Vite dev server port |
| `MLFLOW_PORT` | `5000` | MLflow tracking server port |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `VITE_API_URL` | `http://localhost:8000` | Frontend → Backend API base URL |

---

## MLflow Usage

### Two Distinct Experiment Tracks

#### 1. Training Experiment (`Predictive_Maintenance_Training`)

Run the training script to log a full model training run:

```bash
# Start the MLflow server first (in background or separate terminal)
cd mlflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Then run training from project root
pip install -r model/requirements.txt
python model/train_model.py
```

**What gets logged:**

| Type | Items |
|---|---|
| Parameters | `n_estimators`, `random_state`, `class_weight`, `test_size`, dataset sizes |
| Metrics | `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc` |
| Artifacts | `confusion_matrix.png`, `feature_importance.png`, `roc_curve.png`, all `.pkl` files |
| Model | Registered as `RandomForest_MachineFail` in the MLflow Model Registry |

#### 2. Inference Experiment (`Predictive_Maintenance_Inference`)

Logged automatically by the backend on every `/api/v1/predict` call.

**What gets logged per prediction:**

| Type | Items |
|---|---|
| Parameters | `Type`, `Air_temperature_K`, `Process_temperature_K`, `Rotational_speed_rpm`, `Torque_Nm`, `Tool_wear_min` |
| Metrics | `prediction`, `failure_probability`, `confidence_score`, `inference_time_ms` |
| Tags | `model_name=RandomForest`, `model_version=1.0`, `api_version=1.0` |
| Run Name | `prediction_YYYYMMDD_HHMMSS` (unique per request) |

> **Design decision**: MLflow logging is isolated per prediction. If the MLflow server is offline, the API continues serving predictions and logs a `WARNING` instead of raising an error.

#### 3. Model Comparison / AutoML (`Predictive_Maintenance_ModelComparison`)

Compare the production Random Forest against 4 other algorithms (Gradient Boosting, Logistic Regression, Decision Tree, KNN) on the same dataset split.

```bash
python model/compare_models.py
```
Logs metrics and a comparison bar chart to MLflow. The production application continues to use the existing Random Forest.

#### 4. SHAP Interpretability

Generate global and local SHAP explanations for the Random Forest model.

```bash
python model/shap_analysis.py
```
Logs global feature importance (bar and beeswarm plots) and local single-prediction (waterfall plot) artifacts to the `Predictive_Maintenance_Training` experiment.

---

## DVC Usage

Dataset and model files are tracked by DVC, not Git:

```bash
# Check tracking status
dvc status

# To configure a remote and push large files
dvc remote add -d storage s3://your-bucket/path
dvc push

# To restore files after a fresh clone
dvc pull
```

---

## API Documentation

### `POST /api/v1/predict`

**Request body:**

```json
{
  "Type": "M",
  "Air temperature [K]": 300.0,
  "Process temperature [K]": 310.0,
  "Rotational speed [rpm]": 1500.0,
  "Torque [Nm]": 40.0,
  "Tool wear [min]": 50.0
}
```

**Validation constraints:** `Type` ∈ {L, M, H} · Temperature ∈ [280, 320] K · Speed ∈ [500, 4000] rpm · Torque ∈ [0, 100] Nm · Wear ∈ [0, 300] min

**Response:**

```json
{
  "machine_failure": 0,
  "failure_probability": 0.04,
  "confidence_score": 0.96,
  "recommendation": "Machine is operating within normal performance bounds."
}
```

### `GET /api/v1/health`

Returns API status and model loading state.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

**Expected output:**

```
============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-9.1.1, pluggy-1.6.0
...
collected 27 items

tests/test_backend.py::TestHealth::test_health_status_200 PASSED         [  3%]
...
tests/test_backend.py::TestPredictValidation::test_string_value_for_numeric_field_returns_422 PASSED [100%]

============================= 27 passed in 9.47s ==============================
```

---

## Demo Instructions

1. `docker compose up --build`
2. Open **http://localhost:5173**
3. Select the **Inference Panel** tab
4. Click **"Tool Failure"** preset → click **"Evaluate Operational Risk"**
5. Observe: red **CRITICAL DANGER** alert + confidence bar + checklist
6. Open **http://localhost:5000** → confirm run appears in `Predictive_Maintenance_Inference`
7. Click **"Normal Run"** preset → run again → observe: green **SAFE OPERATION** result
8. Scroll down to see **Prediction History Log** table in the UI

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `API: Offline` badge in header | Backend container not running | Check `docker compose logs backend` |
| `422 Validation Error` from API | Input value outside allowed range | Check temperature 280–320 K, speed 500–4000 rpm |
| MLflow runs not appearing | MLflow container still starting | Wait 30 s and refresh; MLflow installs pip on first boot |
| `ModuleNotFoundError: backend` | Running pytest from wrong directory | Run from project root: `python -m pytest tests/` |
| Frontend shows blank page | Vite not started | `cd frontend && npm install && npm run dev` |
| `pydantic_core.ValidationError` on Settings | `.env` has keys not in Pydantic model | Already handled: `extra="ignore"` in `config.py` |
