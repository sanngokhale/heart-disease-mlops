# End-to-End MLOps Pipeline - Video Script (3 Minutes)

## Total Duration: 3 minutes

---

## SCENE 1: Introduction (15 seconds)

**[Show: VS Code with project open]**

**SAY:**
> "Hi, I'm Sannidhi, Student ID CS202505064. This is my MLOps pipeline for Heart Disease Prediction using the UCI dataset. I'll quickly walk through the end-to-end pipeline."

---

## SCENE 2: Data & Model Training (45 seconds)

**[Show: MLflow UI in browser - http://127.0.0.1:5001]**

**SAY:**
> "The pipeline starts with data preprocessing - handling missing values and feature engineering using scikit-learn's ColumnTransformer."

> "I trained three models - Logistic Regression, Random Forest, and Gradient Boosting - all tracked in MLflow. Logistic Regression achieved the best ROC-AUC of 0.92 and was selected as the production model."

**[Show: models folder in VS Code]**

**SAY:**
> "The model and preprocessor are saved as joblib files for reproducibility."

---

## SCENE 3: Testing & Docker (45 seconds)

**[Show: Terminal with pytest output or screenshot]**

**SAY:**
> "The code includes 27 unit tests covering the API, data processing, and model - all passing."

**[Show: Terminal - run docker ps]**

```bash
docker ps
```

**SAY:**
> "The application is containerized using Docker with a multi-stage build. The container runs a FastAPI server on port 8000."

**[Show: curl health check]**

```bash
curl http://localhost:8000/health
```

**SAY:**
> "The health endpoint confirms the model is loaded and ready."

---

## SCENE 4: Kubernetes Deployment (45 seconds)

**[Show: Terminal]**

```bash
kubectl get pods -n mlops-heart-disease
kubectl get svc -n mlops-heart-disease
```

**SAY:**
> "The API is deployed to Kubernetes with 2 replicas for high availability, a LoadBalancer service, and auto-scaling configured from 2 to 5 pods."

**[Show: Prediction curl command]**

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

**SAY:**
> "The prediction endpoint returns the result - Low Risk with 87% confidence."

---

## SCENE 5: Monitoring & Conclusion (30 seconds)

**[Show: Terminal]**

```bash
curl http://localhost:8080/metrics | grep heart_disease
```

**SAY:**
> "The API exposes Prometheus metrics for monitoring predictions and latency. Request logging is also enabled for debugging."

> "That's the complete MLOps pipeline - from data to production with experiment tracking, CI/CD, containerization, Kubernetes deployment, and monitoring. Thank you!"

---

## Pre-Recording Setup

Run these BEFORE you start recording:

```bash
# Make sure everything is running
docker ps
kubectl get pods -n mlops-heart-disease

# Port forward 
kubectl port-forward svc/heart-disease-api-service 8080:80 -n mlops-heart-disease &

# Start MLflow UI
./venv/bin/mlflow ui --port 5001 &
```

---

## 3-Minute Checklist

| Scene | Time | What to Show |
|-------|------|--------------|
| ☐ Intro | 0:00-0:15 | VS Code project |
| ☐ Data & Training | 0:15-1:00 | MLflow UI, models folder |
| ☐ Testing & Docker | 1:00-1:45 | pytest, docker ps, health |
| ☐ Kubernetes | 1:45-2:30 | pods, svc, predict |
| ☐ Conclusion | 2:30-3:00 | metrics, wrap up |

---

## Tips

1. **Pre-run everything** before recording
2. **Have browser tabs ready** - MLflow UI, Swagger docs
3. **Clear terminal** before each command
4. **Speak naturally** - don't rush

*Good luck, Sannidhi!*
