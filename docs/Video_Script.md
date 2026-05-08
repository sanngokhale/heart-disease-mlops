# End-to-End MLOps Pipeline - Video Script

## Total Duration: ~8-10 minutes

---

## SCENE 1: Introduction (30 seconds)

**[Show: Desktop/Title slide or webcam]**

**SAY:**
> "Hello, my name is Sannidhi, Student ID CS202505064. This video demonstrates an end-to-end MLOps pipeline for Heart Disease Prediction, completed as part of the MLOps course assignment.
>
> In this video, I will walk you through all 8 parts of the pipeline - from data exploration to production deployment with monitoring."

---

## SCENE 2: Project Overview (1 minute)

**[Show: VS Code with project folder expanded]**

**SAY:**
> "Let me start by showing you the project structure. This is the heart-disease-mlops project."

**[Expand folders as you mention them]**

**SAY:**
> "The project is organized into several directories:
> - The `src` folder contains all source code - including the API, data processing, and model training scripts
> - The `data` folder has raw and processed datasets
> - The `models` folder stores our trained model artifacts
> - The `kubernetes` folder contains deployment manifests
> - The `tests` folder has our unit tests
> - And we have a Dockerfile for containerization"

**SAY:**
> "The tech stack includes Python, scikit-learn for machine learning, FastAPI for the REST API, MLflow for experiment tracking, Docker for containerization, and Kubernetes for orchestration."

---

## SCENE 3: Part 1 - Exploratory Data Analysis (1 minute)

**[Show: Jupyter Notebook - 01_EDA.ipynb]**

**SAY:**
> "Part 1 covers Exploratory Data Analysis. Let me open the EDA notebook."

**[Scroll through the notebook showing visualizations]**

**SAY:**
> "The dataset is the UCI Heart Disease dataset with 303 patient records and 13 clinical features like age, blood pressure, cholesterol, and chest pain type."

**[Show distribution plot]**

**SAY:**
> "The target variable is relatively balanced - about 54% have heart disease and 46% do not. This is good because we don't need special handling for class imbalance."

**[Show correlation heatmap if visible]**

**SAY:**
> "Key findings from EDA include:
> - Higher heart disease prevalence in ages 50 to 65
> - Maximum heart rate shows negative correlation with disease
> - Chest pain type is a strong predictor"

---

## SCENE 4: Part 2 - Data Pipeline & Feature Engineering (1.5 minutes)

**[Show: Terminal]**

**SAY:**
> "Part 2 covers feature engineering and model development. First, let me run the data pipeline."

**[Type and run:]**
```bash
python src/data/download_data.py
```

**SAY:**
> "This script downloads the heart disease dataset from the UCI repository."

**[Wait for completion, then run:]**
```bash
python src/data/preprocess.py
```

**SAY:**
> "The preprocessing script handles missing values using median imputation, validates data types, and encodes the target variable to binary format."

**[Show: VS Code - feature_engineering.py]**

**SAY:**
> "For feature engineering, I'm using a scikit-learn ColumnTransformer. Numeric features like age, blood pressure, and cholesterol are standardized using StandardScaler. Categorical features like chest pain type and gender are one-hot encoded. This pipeline ensures consistent preprocessing for both training and inference."

---

## SCENE 5: Part 3 - Model Training with MLflow (1.5 minutes)

**[Show: Terminal]**

**SAY:**
> "Now let's train the models with MLflow experiment tracking."

**[Type and run:]**
```bash
python src/models/train_with_mlflow.py
```

**SAY:**
> "This script trains three different models: Logistic Regression, Random Forest, and Gradient Boosting. Each model is evaluated using 5-fold cross-validation."

**[Wait for output to appear]**

**SAY:**
> "As you can see, each model's metrics are printed - accuracy, precision, recall, F1-score, and ROC-AUC. The best model based on ROC-AUC score is selected automatically."

**[Point to the output]**

**SAY:**
> "Logistic Regression achieved the highest ROC-AUC of 0.92, so it was selected as the best model and saved to the models folder."

---

## SCENE 6: Part 4 - MLflow Experiment Tracking (1 minute)

**[Show: Terminal]**

**SAY:**
> "Let me show you the MLflow tracking interface."

**[Type and run:]**
```bash
./venv/bin/mlflow ui --port 5001
```

**[Open browser: http://127.0.0.1:5001]**

**SAY:**
> "This is the MLflow UI. It shows all our experiment runs."

**[Click on the experiment]**

**SAY:**
> "Here you can see all three model runs with their metrics. I can compare them side by side."

**[Click on a run to show details]**

**SAY:**
> "For each run, MLflow tracks:
> - Parameters like model type and hyperparameters
> - Metrics like accuracy, precision, recall, and ROC-AUC
> - Artifacts including the trained model and visualizations
>
> This ensures complete reproducibility - anyone can recreate these exact results."

---

## SCENE 7: Part 5 - Model Packaging (30 seconds)

**[Show: VS Code - models folder]**

**SAY:**
> "Part 4 covers model packaging. The trained artifacts are saved in the models folder."

**[Open/show each file]**

**SAY:**
> "We have three files:
> - `best_model.joblib` - the trained Logistic Regression model
> - `preprocessor.joblib` - the feature transformation pipeline
> - `model_info.json` - metadata including model name, training date, and performance metrics
>
> These artifacts are versioned and can be loaded for inference."

---

## SCENE 8: Part 6 - CI/CD and Testing (1 minute)

**[Show: VS Code - .github/workflows/ci-cd.yml]**

**SAY:**
> "Part 5 covers CI/CD pipeline and automated testing. Here's our GitHub Actions workflow."

**[Scroll through the YAML file]**

**SAY:**
> "The pipeline has multiple jobs:
> - Lint: runs flake8 for code quality
> - Test: runs pytest for unit tests
> - Build: creates the Docker image
> - Deploy: deploys to Kubernetes
>
> Each job depends on the previous one, ensuring code quality before deployment."

**[Show: Terminal]**

**SAY:**
> "Let me run the tests locally."

**[Type and run:]**
```bash
pytest tests/ -v
```

**[Wait for output]**

**SAY:**
> "All 27 tests pass. We have tests covering the API endpoints, data processing functions, and model training pipeline."

---

## SCENE 9: Part 7 - Docker Containerization (1 minute)

**[Show: VS Code - Dockerfile]**

**SAY:**
> "Part 6 covers Docker containerization. Here's our Dockerfile."

**[Scroll through Dockerfile]**

**SAY:**
> "This is a multi-stage build for optimized image size. The first stage installs dependencies, and the second stage creates a minimal production image. It also uses a non-root user for security."

**[Show: Terminal]**

**SAY:**
> "Let me build and run the container."

**[Type and run:]**
```bash
docker build -t heart-disease-api .
```

**[Wait for build - can fast-forward this part in editing]**

**SAY:**
> "Build complete. Now let's run the container."

**[Type and run:]**
```bash
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

**[Type and run:]**
```bash
docker ps
```

**SAY:**
> "The container is running. Let me test the health endpoint."

**[Type and run:]**
```bash
curl http://localhost:8000/health
```

**SAY:**
> "The API returns healthy status, confirming the model is loaded and ready for predictions."

---

## SCENE 10: Part 8 - Kubernetes Deployment (1.5 minutes)

**[Show: Docker Desktop - Kubernetes tab with green indicator]**

**SAY:**
> "Part 7 covers production deployment on Kubernetes. I'm using Docker Desktop's built-in Kubernetes cluster."

**[Show: VS Code - kubernetes folder]**

**SAY:**
> "We have four Kubernetes manifests:
> - namespace.yaml creates an isolated namespace
> - deployment.yaml runs 2 pod replicas
> - service.yaml creates a LoadBalancer
> - hpa.yaml enables auto-scaling from 2 to 5 pods"

**[Show: Terminal]**

**SAY:**
> "Let me deploy to Kubernetes."

**[Type and run:]**
```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
```

**SAY:**
> "All resources created. Let me verify the deployment."

**[Type and run:]**
```bash
kubectl get pods -n mlops-heart-disease
```

**SAY:**
> "Both pods are running. This provides high availability - if one pod fails, the other continues serving requests."

**[Type and run:]**
```bash
kubectl get svc -n mlops-heart-disease
```

**SAY:**
> "The LoadBalancer service is exposing our API on port 80."

---

## SCENE 11: Live API Demo (1 minute)

**[Show: Terminal]**

**SAY:**
> "Now let me demonstrate the API with a live prediction."

**[Type and run:]**
```bash
kubectl port-forward svc/heart-disease-api-service 8080:80 -n mlops-heart-disease &
```

**SAY:**
> "I'm port-forwarding to access the service locally."

**[Type and run:]**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

**[Wait for response]**

**SAY:**
> "The API returns a JSON response with:
> - Prediction: 0, meaning no heart disease
> - Probability: about 0.13 or 13% chance of disease
> - Risk level: Low Risk
> - Confidence: about 87%
>
> This is a production-ready prediction service."

---

## SCENE 12: Part 9 - Monitoring and Logging (1 minute)

**[Show: Terminal]**

**SAY:**
> "Part 8 covers monitoring and logging."

**[Type and run:]**
```bash
curl http://localhost:8080/metrics | grep heart_disease
```

**SAY:**
> "The API exposes Prometheus metrics. You can see:
> - Total predictions made, broken down by risk level
> - Prediction latency histogram
> - Total API requests by endpoint and status
>
> These metrics can be scraped by Prometheus and visualized in Grafana."

**[Type and run:]**
```bash
kubectl logs deployment/heart-disease-api -n mlops-heart-disease --tail=10
```

**SAY:**
> "The application also logs every request with timestamp, HTTP method, path, status code, and response time. This is essential for debugging and auditing."

---

## SCENE 13: Swagger UI (30 seconds)

**[Show: Browser - http://localhost:8080/docs]**

**SAY:**
> "Finally, the API includes interactive Swagger documentation."

**[Scroll through the page]**

**SAY:**
> "Developers can see all available endpoints:
> - /health for health checks
> - /predict for making predictions
> - /model-info for model metadata
> - /metrics for Prometheus metrics
>
> They can even test the API directly from this interface."

---

## SCENE 14: Conclusion (30 seconds)

**[Show: Desktop or webcam]**

**SAY:**
> "This completes the demonstration of the end-to-end MLOps pipeline.
>
> To summarize, we covered:
> - Data exploration and preprocessing
> - Feature engineering and model training
> - Experiment tracking with MLflow
> - Model packaging and versioning
> - CI/CD with automated testing
> - Docker containerization
> - Kubernetes deployment with auto-scaling
> - And production monitoring with Prometheus metrics
>
> The system is scalable, reproducible, and production-ready.
>
> Thank you for watching."

---

## Commands Quick Reference

```bash
# Data Pipeline
python src/data/download_data.py
python src/data/preprocess.py

# Training
python src/models/train_with_mlflow.py

# MLflow UI
./venv/bin/mlflow ui --port 5001

# Testing
pytest tests/ -v

# Docker
docker build -t heart-disease-api .
docker run -d -p 8000:8000 --name heart-api heart-disease-api
docker ps
curl http://localhost:8000/health

# Kubernetes
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl get pods -n mlops-heart-disease
kubectl get svc -n mlops-heart-disease

# Port Forward & Test
kubectl port-forward svc/heart-disease-api-service 8080:80 -n mlops-heart-disease &

curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'

# Monitoring
curl http://localhost:8080/metrics | grep heart_disease
kubectl logs deployment/heart-disease-api -n mlops-heart-disease --tail=10
```

---

## Recording Tips

1. **Before Recording:**
   - Clear all terminals: `clear`
   - Close unnecessary applications
   - Increase terminal font size (Cmd + Plus)
   - Ensure Docker Desktop and Kubernetes are running
   - Stop any existing containers: `docker stop heart-api && docker rm heart-api`

2. **During Recording:**
   - Speak slowly and clearly
   - Pause after each command to show output
   - Move mouse pointer to highlight important parts
   - Keep energy in your voice

3. **Screen Recording (Mac):**
   - QuickTime Player → File → New Screen Recording
   - Or use OBS Studio for more control

4. **Post-Recording:**
   - Can speed up long build/install sections
   - Add captions if needed
   - Export as MP4

---

*Good luck with your video!*
