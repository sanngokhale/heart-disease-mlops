# MLOps Assignment Report
## Heart Disease Prediction - End-to-End MLOps Pipeline

**Course:** MLOps (S2-25_AMLCSZG523)  
**Assignment:** Assignment 1  
**Date:** May 2026  
**Repository:** [GitHub Link - Add Your URL Here]

---

## Table of Contents
1. [Setup & Installation Instructions](#1-setup--installation-instructions)
2. [EDA and Modelling Choices](#2-eda-and-modelling-choices)
3. [Experiment Tracking Summary](#3-experiment-tracking-summary)
4. [Architecture Diagram](#4-architecture-diagram)
5. [CI/CD and Deployment Workflow](#5-cicd-and-deployment-workflow)
6. [Monitoring & Logging](#6-monitoring--logging)
7. [Screenshots](#7-screenshots)

---

## 1. Setup & Installation Instructions

### Prerequisites
- Python 3.11+
- Docker Desktop
- Kubernetes (Docker Desktop with Kubernetes enabled)
- Git

### Installation Steps

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd heart-disease-mlops

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and preprocess data
python src/data/download_data.py
python src/data/preprocess.py

# 5. Train model with MLflow tracking
python src/models/train_with_mlflow.py

# 6. Run tests
pytest tests/ -v

# 7. Start API locally
uvicorn src.api.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Build Docker image
docker build -t heart-disease-api .

# Run container
docker run -d -p 8000:8000 --name heart-api heart-disease-api

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

### Kubernetes Deployment

```bash
# Enable Kubernetes in Docker Desktop, then:
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Port forward to access locally
kubectl port-forward svc/heart-disease-api-service 8080:80 -n mlops-heart-disease
```

---

## 2. EDA and Modelling Choices

### Dataset Overview
- **Source:** UCI Heart Disease Dataset (Cleveland)
- **Records:** 303 samples
- **Features:** 13 clinical attributes + 1 target
- **Target:** Binary classification (0 = No Disease, 1 = Heart Disease)

### Features

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Gender (1=male, 0=female) | Categorical |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl | Categorical |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina | Categorical |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-4) | Categorical |
| thal | Thalassemia (0-3) | Categorical |

### EDA Key Findings

1. **Class Distribution:** Balanced dataset (~54% disease, ~46% no disease)
2. **Age:** Higher heart disease prevalence in ages 50-65
3. **Sex:** Males show higher heart disease rates
4. **Chest Pain (cp):** Type 2 (non-anginal pain) strongly correlates with heart disease
5. **Max Heart Rate (thalach):** Lower max heart rate associated with disease
6. **Oldpeak:** Higher ST depression correlates with heart disease

### Data Preprocessing
- Handled missing values using median imputation
- Converted target to binary (0 vs 1+)
- No feature scaling in raw data (handled in pipeline)

### Modelling Choices

**Models Evaluated:**
1. **Logistic Regression** - Baseline, interpretable
2. **Random Forest** - Ensemble method, handles non-linearity
3. **Gradient Boosting (XGBoost)** - State-of-the-art for tabular data

**Feature Engineering Pipeline:**
- StandardScaler for numeric features (age, trestbps, chol, thalach, oldpeak)
- OneHotEncoder for categorical features (sex, cp, fbs, restecg, exang, slope, ca, thal)
- ColumnTransformer to combine both transformations

**Why These Models?**
- Small dataset (303 samples) - tree-based models handle well
- Mixed feature types - pipeline handles scaling/encoding
- Need interpretability for medical domain - Logistic Regression provides coefficients
- Cross-validation to prevent overfitting

---

## 3. Experiment Tracking Summary

### MLflow Experiment Setup
- **Tracking URI:** Local file store (`./mlruns`)
- **Experiment Name:** heart-disease-classification
- **Metrics Logged:** accuracy, precision, recall, f1_score, roc_auc, cv_roc_auc_mean

### Experiment Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV ROC-AUC |
|-------|----------|-----------|--------|----------|---------|------------|
| Logistic Regression | 0.87 | 0.86 | 0.91 | 0.88 | 0.92 | 0.90 |
| Random Forest | 0.85 | 0.84 | 0.88 | 0.86 | 0.91 | 0.88 |
| Gradient Boosting | 0.84 | 0.83 | 0.88 | 0.85 | 0.90 | 0.87 |

### Best Model Selection
- **Selected Model:** Logistic Regression
- **Selection Criteria:** Highest ROC-AUC score
- **Model Artifact:** `models/best_model.joblib`
- **Preprocessor Artifact:** `models/preprocessor.joblib`

### MLflow Artifacts Logged
- Trained model (joblib format)
- Confusion matrix plots
- Feature importance plots
- Model parameters
- Training/test metrics

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   UCI ML     │     │    Data      │     │   Feature    │     │    Model     │
│  Repository  │────▶│ Preprocessing│────▶│ Engineering  │────▶│   Training   │
│              │     │              │     │              │     │              │
│ (download_   │     │ (preprocess. │     │ (feature_    │     │ (train_with_ │
│  data.py)    │     │    py)       │     │  engineering │     │  mlflow.py)  │
└──────────────┘     └──────────────┘     │    .py)      │     └──────┬───────┘
                                          └──────────────┘            │
                                                                      │
                     ┌────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   MLflow     │     │    Model     │     │   GitHub     │
│  Tracking    │◀────│   Artifacts  │────▶│   Actions    │
│   Server     │     │              │     │    CI/CD     │
│              │     │ best_model.  │     │              │
│ (Port 5001)  │     │ joblib       │     │ (lint/test/  │
└──────────────┘     └──────┬───────┘     │  build)      │
                           │              └──────┬───────┘
                           │                     │
                           ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   FastAPI    │     │   Docker     │
                     │     API      │◀────│    Image     │
                     │              │     │              │
                     │ /predict     │     │ heart-       │
                     │ /health      │     │ disease-api  │
                     │ /metrics     │     └──────┬───────┘
                     └──────┬───────┘            │
                           │                     │
                           ▼                     ▼
                     ┌─────────────────────────────────────┐
                     │         Kubernetes Cluster          │
                     │                                     │
                     │  ┌─────────────┐ ┌─────────────┐   │
                     │  │   Pod 1     │ │   Pod 2     │   │
                     │  │ (replica)   │ │ (replica)   │   │
                     │  └─────────────┘ └─────────────┘   │
                     │                                     │
                     │  ┌─────────────────────────────┐   │
                     │  │    LoadBalancer Service     │   │
                     │  │       (Port 80)             │   │
                     │  └─────────────────────────────┘   │
                     │                                     │
                     │  ┌─────────────────────────────┐   │
                     │  │  HorizontalPodAutoscaler    │   │
                     │  │     (2-5 replicas)          │   │
                     │  └─────────────────────────────┘   │
                     └─────────────────────────────────────┘
                                      │
                                      ▼
                     ┌─────────────────────────────────────┐
                     │           Monitoring                │
                     │                                     │
                     │  • Prometheus Metrics (/metrics)    │
                     │  • Request Logging (middleware)     │
                     │  • Health Checks (/health)          │
                     └─────────────────────────────────────┘
```

### Component Descriptions

| Component | Purpose | Technology |
|-----------|---------|------------|
| Data Ingestion | Download UCI dataset | Python requests |
| Preprocessing | Clean, handle missing values | Pandas |
| Feature Engineering | Scale numeric, encode categorical | scikit-learn Pipeline |
| Model Training | Train & compare models | scikit-learn, XGBoost |
| Experiment Tracking | Log metrics, parameters, artifacts | MLflow |
| API | Serve predictions | FastAPI |
| Containerization | Package application | Docker |
| Orchestration | Deploy & scale | Kubernetes |
| CI/CD | Automate testing & deployment | GitHub Actions |
| Monitoring | Track API performance | Prometheus metrics |

---

## 5. CI/CD and Deployment Workflow

### GitHub Actions Pipeline (`.github/workflows/ci-cd.yml`)

```yaml
Jobs:
1. lint      → Code quality (flake8)
2. test      → Unit tests (pytest)
3. train     → Model training
4. build     → Docker image
5. deploy    → Kubernetes deployment
```

### Pipeline Stages

#### Stage 1: Lint
- Tool: flake8
- Checks: Code style, unused imports, syntax errors
- Config: `--max-line-length=88 --ignore=E402`

#### Stage 2: Test
- Tool: pytest
- Tests: 27 unit tests
- Coverage: Data processing, model training, API endpoints

#### Stage 3: Train
- Script: `src/models/train_with_mlflow.py`
- Output: `models/best_model.joblib`

#### Stage 4: Build
- Dockerfile: Multi-stage build
- Image: `heart-disease-api:latest`
- Security: Non-root user

#### Stage 5: Deploy
- Platform: Kubernetes (Docker Desktop)
- Manifests: namespace, deployment, service, hpa
- Replicas: 2 (auto-scales to 5)

### Kubernetes Resources

| Resource | File | Purpose |
|----------|------|---------|
| Namespace | `kubernetes/namespace.yaml` | Isolate resources |
| Deployment | `kubernetes/deployment.yaml` | Run 2 pod replicas |
| Service | `kubernetes/service.yaml` | LoadBalancer on port 80 |
| HPA | `kubernetes/hpa.yaml` | Auto-scale 2-5 pods |

---

## 6. Monitoring & Logging

### API Request Logging
- **Implementation:** FastAPI middleware
- **Logged Info:** Method, path, status code, duration
- **Format:** `POST /predict - Status: 200 - Duration: 0.045s`

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `heart_disease_predictions_total` | Counter | Total predictions by risk level |
| `heart_disease_prediction_latency_seconds` | Histogram | Prediction latency |
| `heart_disease_api_requests_total` | Counter | All API requests |

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Prometheus metrics |
| `/model-info` | GET | Model metadata |
| `/docs` | GET | Swagger UI |

---

## 7. Screenshots

### Part 1: EDA
- [ ] Jupyter notebook with visualizations
- [ ] Distribution plots
- [ ] Correlation heatmap

### Part 2: Feature Engineering & Model Development
- [ ] Model training output
- [ ] Model comparison metrics

### Part 3: Experiment Tracking
- [ ] MLflow UI - Experiments list
- [ ] MLflow UI - Run comparison
- [ ] MLflow UI - Metrics charts

### Part 4: Model Packaging
- [ ] Model files in `models/` directory
- [ ] `model_info.json` contents

### Part 5: CI/CD Pipeline
- [ ] GitHub Actions workflow
- [ ] Lint passing
- [ ] Tests passing (27/27)

### Part 6: Docker Containerization
- [ ] `docker build` output
- [ ] `docker run` and container running
- [ ] `curl /predict` response

### Part 7: Kubernetes Deployment
- [ ] `kubectl get nodes`
- [ ] `kubectl get pods -n mlops-heart-disease`
- [ ] `kubectl get svc -n mlops-heart-disease`
- [ ] `kubectl get deployment -n mlops-heart-disease`
- [ ] Health endpoint response
- [ ] Prediction endpoint response

### Part 8: Monitoring & Logging
- [ ] `kubectl logs` showing request logs
- [ ] `/metrics` endpoint output
- [ ] FastAPI Swagger UI (`/docs`)

---

## Repository Structure

```
heart-disease-mlops/
├── .github/workflows/ci-cd.yml    # CI/CD pipeline
├── data/
│   ├── raw/heart_disease.csv      # Original dataset
│   └── processed/                 # Preprocessed data
├── docs/
│   └── MLOps_Assignment_Report.md # This report
├── kubernetes/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── models/
│   ├── best_model.joblib          # Trained model
│   ├── preprocessor.joblib        # Feature transformer
│   └── model_info.json            # Model metadata
├── notebooks/
│   └── 01_EDA.ipynb               # Exploratory analysis
├── src/
│   ├── api/main.py                # FastAPI application
│   ├── data/
│   │   ├── download_data.py
│   │   └── preprocess.py
│   └── models/
│       ├── feature_engineering.py
│       └── train_with_mlflow.py
├── tests/                         # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Conclusion

This project demonstrates a complete MLOps pipeline for heart disease prediction, implementing:

✅ **Data Pipeline:** Automated data acquisition and preprocessing  
✅ **ML Pipeline:** Feature engineering, model training, experiment tracking  
✅ **CI/CD Pipeline:** Automated testing, building, and deployment  
✅ **Production Deployment:** Containerized API on Kubernetes  
✅ **Monitoring:** Prometheus metrics and request logging  

The system is designed for reproducibility, scalability, and maintainability following MLOps best practices.

---

**Repository Link:** [Add Your GitHub URL Here]
