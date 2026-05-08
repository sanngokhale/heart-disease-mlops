"""
FastAPI Application for Heart Disease Prediction
Production-ready API with monitoring and logging
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import logging
from datetime import datetime
import os
import json
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    API for predicting heart disease risk based on patient health data.

    ## Features
    - Real-time predictions using trained ML model
    - Health checks for monitoring
    - Prometheus metrics endpoint
    - Comprehensive input validation

    ## Model Information
    - Trained on UCI Heart Disease dataset
    - Binary classification: Low Risk (0) vs High Risk (1)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
PREDICTIONS_TOTAL = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made",
    ["risk_level"],
)

PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")
MODEL_INFO_PATH = os.getenv("MODEL_INFO_PATH", "models/model_info.json")

# Global model and preprocessor
model = None
preprocessor = None
model_info = None


def load_artifacts():
    """Load model and preprocessor artifacts"""
    global model, preprocessor, model_info

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")

        logger.info(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Preprocessor loaded successfully")

        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, "r") as f:
                model_info = json.load(f)
            logger.info(f"Model info loaded: {model_info.get('model_name', 'Unknown')}")

    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        logger.warning("Running without model - predictions will fail")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")


# Load artifacts on startup
@app.on_event("startup")
async def startup_event():
    """Load model artifacts on application startup"""
    load_artifacts()
    logger.info("Application startup complete")


# Feature names in expected order
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# Request/Response Models
class PatientData(BaseModel):
    """Patient health data for prediction"""

    age: int = Field(..., ge=0, le=120, description="Age in years (0-120)")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(
        ..., ge=0, le=300, description="Resting blood pressure (mm Hg)"
    )
    chol: float = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)"
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise induced angina (0=no, 1=yes)"
    )
    oldpeak: float = Field(
        ..., ge=0, le=10, description="ST depression induced by exercise"
    )
    slope: int = Field(
        ..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)"
    )
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    class Config:
        json_schema_extra = {
            "example": {
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
                "thal": 2,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model"""

    prediction: int = Field(
        ..., description="Predicted class (0=No Disease, 1=Disease)"
    )
    probability: float = Field(..., description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level interpretation")
    confidence: float = Field(..., description="Model confidence")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    model_version: Optional[str] = Field(None, description="Model version used")


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: Optional[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_name: Optional[str]
    model_type: Optional[str]
    training_date: Optional[str]
    metrics: Optional[dict]


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="200").inc()

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        model_name=model_info.get("model_name") if model_info else None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    REQUEST_COUNT.labels(endpoint="/model-info", method="GET", status="200").inc()

    if model_info is None:
        return ModelInfoResponse(
            model_name=None,
            model_type=type(model).__name__ if model else None,
            training_date=None,
            metrics=None,
        )

    return ModelInfoResponse(
        model_name=model_info.get("model_name"),
        model_type=model_info.get("model_type"),
        training_date=model_info.get("training_date"),
        metrics=model_info.get("metrics"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Predict heart disease risk for a patient.

    Returns prediction (0 or 1), probability, and risk level.
    """
    start_time = time.time()

    if model is None or preprocessor is None:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="503").inc()
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert input to DataFrame
        input_dict = patient.model_dump()
        logger.info(f"Received prediction request for patient age={input_dict['age']}")

        # Create DataFrame with correct column order
        df = pd.DataFrame([input_dict])
        df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        # Preprocess
        X = preprocessor.transform(df)

        # Predict
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        # Calculate confidence
        confidence = probability if prediction == 1 else 1 - probability

        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS_TOTAL.labels(risk_level=risk_level.replace(" ", "_").lower()).inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()

        response = PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            timestamp=datetime.utcnow().isoformat(),
            model_version=model_info.get("training_date") if model_info else None,
        )

        logger.info(
            f"Prediction: {prediction}, Probability: {probability:.4f}, Risk: {risk_level}"
        )

        return response

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Duration: {process_time:.3f}s"
    )

    return response


if __name__ == "__main__":
    import uvicorn

    # For local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
