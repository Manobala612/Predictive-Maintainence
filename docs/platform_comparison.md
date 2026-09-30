# Platform and Tool Comparison
## Technologies Used in the Predictive Maintenance MLOps Project

This document explains the technology choices made in this project and how each tool compares to its alternatives. Only tools actually used or evaluated in this project are covered.

---

## 1. Machine Learning Framework

### Choice: scikit-learn (Random Forest)

| Library | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **scikit-learn** ✅ | Simple API, well-documented, fast training for tabular data, consistent pipeline interface | No GPU acceleration, limited deep learning support | Chosen: AI4I dataset is tabular with 10k rows — no deep learning needed. RF with scikit-learn is fast and interpretable. |
| XGBoost | Usually higher accuracy on tabular data, handles imbalanced classes well | More hyperparameters to tune, harder to interpret without SHAP | Not chosen as primary; tested in compare_models.py as GradientBoosting |
| LightGBM | Fastest gradient boosting implementation, low memory | Less common in entry-level MLOps pipelines | Available as comparison candidate |
| TensorFlow / PyTorch | Excellent for sequential, image, and NLP data | Overkill for 6-feature tabular classification | Not appropriate for this problem |

**Verdict:** scikit-learn Random Forest is the right choice for this dataset scale and feature type. Its `predict_proba()` method provides calibrated probabilities suitable for the failure probability score.

---

## 2. Model Serving / API Framework

### Choice: FastAPI

| Framework | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **FastAPI** ✅ | Automatic OpenAPI docs, Pydantic validation, async support, fast (Starlette/uvicorn), type hints | Newer than Flask so smaller legacy ecosystem | Chosen: Pydantic schemas provide automatic input validation with clear error messages. OpenAPI docs at /docs are available without extra work. |
| Flask | Mature, large ecosystem, minimal boilerplate | No built-in validation, no async, manual schema definition | Not chosen: validation would need marshmallow or similar manually |
| Django REST Framework | Feature-rich, ORM integration | Heavy for a single-model API, complex setup | Not appropriate for this scope |
| BentoML / Seldon | MLOps-native model serving, auto-scaling | Additional infrastructure complexity | Valid alternative for larger deployments |

**Verdict:** FastAPI is the best choice for a single-model inference API that needs documented endpoints, input validation, and an interactive test UI.

---

## 3. Experiment Tracking

### Choice: MLflow

| Tool | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **MLflow** ✅ | Open-source, self-hosted, model registry, UI, language agnostic, scikit-learn autologging | UI is functional but not polished, self-hosting adds ops burden | Chosen: integrates directly with scikit-learn, free, can run locally with SQLite |
| Weights & Biases | Best-in-class UI, team collaboration, sweep hyperparameter tuning | SaaS subscription required for teams, data leaves your infrastructure | Good alternative for teams; not chosen to keep project self-contained |
| Neptune.ai | Similar to W&B, strong comparison views | SaaS, cost | Not chosen |
| TensorBoard | Excellent for deep learning metrics and model graphs | Weak support for scikit-learn tabular experiments | Not appropriate here |
| DVC Studio | Integrates with DVC tracking | Requires DVC pipeline setup | Partial overlap with DVC already in project |

**Two MLflow experiments in this project:**

| Experiment | Purpose | Script |
|---|---|---|
| `Predictive_Maintenance_Training` | Log hyperparameters, metrics, plots, register model | `model/train_model.py` |
| `Predictive_Maintenance_Inference` | Log every API prediction for audit/drift detection | `backend/services/prediction.py` |
| `Predictive_Maintenance_ModelComparison` | Compare 5 candidate models | `model/compare_models.py` |

---

## 4. Data Versioning

### Choice: DVC (Data Version Control)

| Tool | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **DVC** ✅ | Git-like interface for data/models, storage-agnostic (S3/GCS/local), pipeline stages, reproducibility | No web UI, requires remote storage for team sharing | Chosen: industry standard for ML data versioning, works with any git repo |
| Git LFS | Simple, built into GitHub/GitLab | Storage limits, not ML-aware, no pipeline support | Not suitable for 3MB+ model files at scale |
| Pachyderm | Strong pipeline and versioning | Complex Kubernetes dependency | Overkill |
| Delta Lake / Iceberg | Excellent for large analytical datasets | Overkill for a 10k-row CSV | Not appropriate |

**Current DVC tracking in this project:**

```
dataset/ai4i2020.csv         ← tracked by dataset/ai4i2020.csv.dvc
model/machine_failure_model.pkl  ← tracked by model/machine_failure_model.pkl.dvc
model/label_encoders.pkl         ← tracked by model/label_encoders.pkl.dvc
model/feature_names.pkl          ← tracked by model/feature_names.pkl.dvc
```

> [!NOTE]
> No DVC remote is configured. Files are tracked locally. To enable team sharing: `dvc remote add -d myremote s3://your-bucket && dvc push`

---

## 5. Frontend Framework

### Choice: React + Vite + Tailwind CSS

| Stack | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **React + Vite + Tailwind** ✅ | Fast build, component reuse, Tailwind for rapid styling, large ecosystem | More complex than plain HTML for simple dashboards | Chosen: suitable for a multi-page SPA with state (prediction history, health polling) |
| Vue.js + Vite | Simpler learning curve, good tooling | Smaller ecosystem | Valid alternative |
| Streamlit | ML-native, Python-based, minimal code | Limited customisation, Python runtime required on frontend server | Good for quick prototypes, not for a standalone web app |
| Gradio | Fast ML demo UIs | Very limited styling, not suitable for production | Prototype only |

---

## 6. Containerisation

### Choice: Docker + Docker Compose

| Tool | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **Docker + Docker Compose** ✅ | Standard, portable, defines all three services in one file | No orchestration (no auto-restart beyond what Docker provides) | Chosen: sufficient for single-machine deployment and development |
| Kubernetes (k8s) | Production-grade orchestration, auto-scaling, rolling updates | Steep learning curve, requires cluster setup | Appropriate next step for production scaling |
| Docker Swarm | Simpler than k8s, native Docker | Less adoption than k8s | Valid stepping stone |
| Podman Compose | Rootless containers, no daemon | Less tooling ecosystem | Valid alternative to Docker |

**Docker Compose service map:**

```
mlflow  (ghcr.io/mlflow/mlflow:v2.11.1) :5000  ← healthcheck configured
  ↓ (service_healthy)
backend (python:3.11-slim + uvicorn)     :8000
  ↓
frontend (node:18-alpine + vite)         :5173
```

---

## 7. Model Interpretability

### Choice: SHAP (SHapley Additive exPlanations)

| Tool | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **SHAP** ✅ | Mathematically principled (Shapley values), model-agnostic, tree-optimised path for RF, global + local explanations | Slow for large datasets without sampling | Chosen: industry standard for ML interpretability, TreeExplainer is exact for RF |
| LIME | Fast, intuitive local explanations | Approximation-based, can be unstable | Valid alternative for local explanations |
| ELI5 | Simple permutation importance | Less informative than SHAP | Too basic |
| Captum | PyTorch-native attribution | Requires PyTorch model | Not applicable |

**SHAP outputs generated by `model/shap_analysis.py`:**

| Output | Description |
|---|---|
| `shap_summary_bar.png` | Global feature importance (mean absolute SHAP per feature) |
| `shap_summary_dot.png` | Beeswarm: distribution of SHAP values, coloured by feature value |
| `shap_waterfall.png` | Single-prediction explanation: each feature's contribution shown |

---

## 8. CI/CD

### Choice: GitHub Actions

| Tool | Strengths | Weaknesses | Why chosen / not chosen |
|---|---|---|---|
| **GitHub Actions** ✅ | Free for public repos, tight GitHub integration, matrix builds, community actions | Vendor lock-in, minutes limits on private repos | Chosen: project is hosted on GitHub; Actions requires no additional infrastructure |
| GitLab CI | Self-hosted runners, Docker-native | Requires GitLab | Not chosen (project is on GitHub) |
| Jenkins | Highly configurable, self-hosted | Complex setup, maintenance burden | Overkill |
| CircleCI | Fast, good caching | SaaS cost | Not chosen |

**GitHub Actions pipeline in this project (`.github/workflows/ci.yml`):**

```
Push / PR to main
  ├── backend-test job (ubuntu-latest)
  │   ├── Python 3.11 setup
  │   ├── pip install backend/requirements.txt + flake8 + pytest + httpx
  │   ├── flake8 hard fail (syntax errors, undefined names)
  │   ├── flake8 soft fail (style warnings)
  │   └── pytest tests/ -v (with MLFLOW_TRACKING_URI=sqlite:///mlflow_test.db)
  │
  └── frontend-build job (ubuntu-latest)
      ├── Node.js 18 setup
      ├── npm install
      └── npm run build (Vite production build)
```

---

## 9. Summary Recommendations

| Decision Point | Current Choice | Consider Upgrading When |
|---|---|---|
| Model serving | FastAPI | Traffic exceeds 1000 req/s → consider BentoML or Triton |
| Experiment tracking | MLflow (local SQLite) | Team of 3+ → switch to MLflow with PostgreSQL backend or W&B |
| Data versioning | DVC (no remote) | Multiple contributors → add S3/GCS remote |
| Containerisation | Docker Compose | Moving to cloud/Kubernetes cluster → write Helm chart |
| ML model | Random Forest | After sufficient real-world data → benchmark XGBoost/LightGBM |
| Interpretability | SHAP (batch script) | Regulatory requirement → expose SHAP values via API endpoint |
