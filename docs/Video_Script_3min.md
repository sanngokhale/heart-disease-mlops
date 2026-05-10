# End-to-End MLOps Pipeline - Video Script (3 Minutes)

## Total Duration: 3 minutes

---

## SCENE 1: Introduction (15 seconds)

**[Show: VS Code with project open]**

**SAY:**
> "Hi, I'm Sannidhi, Student ID CS202505064. This is my MLOps pipeline for Heart Disease Prediction using the UCI dataset. I'll quickly walk through the end-to-end pipeline."

---

## SCENE 2: Data & Model Training (30 seconds)

**[Show: MLflow UI in browser - http://127.0.0.1:5001]**

**SAY:**
> "The pipeline starts with data preprocessing - handling missing values and feature engineering using scikit-learn's ColumnTransformer."

> "I trained three models - Logistic Regression, Random Forest, and Gradient Boosting - all tracked in MLflow. Logistic Regression achieved the best ROC-AUC of 0.92 and was selected as the production model."

---

## SCENE 3: Model Packaging (30 seconds)

**[Show: models/ folder in VS Code - expand to show best_model.joblib, preprocessor.joblib, model_info.json]**

**SAY:**
> "For model packaging, the best model and its preprocessor are serialized using joblib. The model_info.json file tracks metadata like model type, training date, and performance metrics."

> "This packaged model is what gets loaded into the Docker container for serving - ensuring consistency between training and production."

---

## SCENE 4: Testing & Docker (30 seconds)

**[Show: Terminal with pytest output or screenshot]**

**SAY:**
> "The code includes 27 unit tests covering the API, data processing, and model - all passing."

**[Show: Terminal - run docker ps]**

```bash
docker ps
```

**SAY:**
> "The application is containerized using Docker with a multi-stage build. The container runs a FastAPI server on port 8000."

---

## SCENE 5: Kubernetes Deployment (45 seconds)

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

## SCENE 6: Monitoring & Conclusion (30 seconds)

**[Show: Terminal]**

```bash
curl http://localhost:8080/metrics | grep heart_disease
```

**SAY:**
> "The API exposes Prometheus metrics for monitoring predictions and latency."

> "That's the complete MLOps pipeline - from data to production with experiment tracking, model packaging, containerization, Kubernetes deployment, and monitoring. Thank you!"

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
| ☐ Data & Training | 0:15-0:45 | MLflow UI |
| ☐ Model Packaging | 0:45-1:15 | models/ folder (joblib, json) |
| ☐ Testing & Docker | 1:15-1:45 | pytest, docker ps |
| ☐ Kubernetes | 1:45-2:30 | pods, svc, predict |
| ☐ Conclusion | 2:30-3:00 | metrics, wrap up |

---

## Tips

1. **Pre-run everything** before recording
2. **Have browser tabs ready** - MLflow UI, Swagger docs
3. **Clear terminal** before each command
4. **Speak naturally** - don't rush

*Good luck, Sannidhi!*
