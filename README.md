# Heart Disease MLOps Project

## MLOps End-to-End ML Model Development, CI/CD, and Production Deployment

### Course: MLOps (S2-25_AMLCSZG523)
### Assignment 1 

---

## 🎯 Project Overview

This project implements a complete MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. It demonstrates modern MLOps best practices including:

- Data acquisition and preprocessing
- Feature engineering and model development  
- Experiment tracking with MLflow
- CI/CD pipelines with GitHub Actions
- Docker containerization
- Kubernetes deployment
- Monitoring and logging

## 📁 Project Structure

```
heart-disease-mlops/
├── .github/workflows/      # CI/CD pipeline configurations
├── configs/                # Configuration files
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Preprocessed data
├── docker/                # Docker-related files
├── docs/                  # Documentation
├── kubernetes/            # K8s deployment manifests
├── models/                # Saved models and artifacts
├── notebooks/             # Jupyter notebooks for EDA
├── screenshots/           # Screenshots for documentation
├── src/
│   ├── api/              # FastAPI application
│   ├── data/             # Data processing scripts
│   ├── models/           # Model training scripts
│   └── utils/            # Utility functions
├── tests/                 # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker
- Minikube (for local Kubernetes)
- Git

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd heart-disease-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Download and preprocess data
python src/data/download_data.py
python src/data/preprocess.py

# 2. Train model with MLflow tracking
python src/models/train_with_mlflow.py

# 3. Run tests
pytest tests/ -v

# 4. Start API locally
uvicorn src.api.main:app --reload

# 5. Build Docker image
docker build -t heart-disease-api:latest .

# 6. Run Docker container
docker run -p 8000:8000 heart-disease-api:latest
```

## 📊 Dataset

**Heart Disease UCI Dataset**
- Source: UCI Machine Learning Repository
- Features: 14 attributes (age, sex, blood pressure, cholesterol, etc.)
- Target: Binary classification (presence/absence of heart disease)

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Prometheus metrics |

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 🐳 Docker

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

# Check logs
docker logs heart-disease-api
```

## ☸️ Kubernetes Deployment

```bash
# Start Minikube
minikube start

# Build image in Minikube
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .

# Deploy
kubectl apply -f kubernetes/

# Get service URL
minikube service heart-disease-api-service -n mlops-heart-disease --url
```

## 📈 MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser: http://localhost:5000
```

## 📝 Marking Scheme

| Task | Marks |
|------|-------|
| Data Acquisition & EDA | 5 |
| Feature Engineering & Model Development | 8 |
| Experiment Tracking | 5 |
| Model Packaging & Reproducibility | 7 |
| CI/CD Pipeline & Automated Testing | 8 |
| Model Containerization | 5 |
| Production Deployment | 7 |
| Monitoring & Logging | 3 |
| Documentation & Reporting | 2 |
| **Total** | **50** |

## 👤 Author

BITS Pilani - MLOps Course Assignment

## 📄 License

This project is for educational purposes only.
