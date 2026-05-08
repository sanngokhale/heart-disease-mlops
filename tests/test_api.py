"""
Unit Tests for API Module
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestAPIEndpoints:
    """Tests for API endpoints using mocked model"""

    @pytest.fixture
    def mock_model_and_preprocessor(self):
        """Create mock model and preprocessor"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[1.0] * 20])

        return mock_model, mock_preprocessor

    @pytest.fixture
    def client(self, mock_model_and_preprocessor):
        """Create test client with mocked dependencies"""
        mock_model, mock_preprocessor = mock_model_and_preprocessor

        with patch.dict("sys.modules", {"joblib": MagicMock()}):
            from src.api.main import app

            # Mock the global model and preprocessor
            import src.api.main as api_module

            api_module.model = mock_model
            api_module.preprocessor = mock_preprocessor
            api_module.model_info = {"model_name": "Test Model"}

            return TestClient(app)

    @pytest.fixture
    def sample_patient_data(self):
        """Sample valid patient data"""
        return {
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

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_predict_endpoint_valid_input(self, client, sample_patient_data):
        """Test prediction with valid input"""
        response = client.post("/predict", json=sample_patient_data)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "timestamp" in data

    def test_predict_endpoint_invalid_age(self, client, sample_patient_data):
        """Test prediction with invalid age"""
        sample_patient_data["age"] = 150  # Invalid age
        response = client.post("/predict", json=sample_patient_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_field(self, client, sample_patient_data):
        """Test prediction with missing field"""
        del sample_patient_data["age"]
        response = client.post("/predict", json=sample_patient_data)

        assert response.status_code == 422  # Validation error

    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model-info")

        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data


class TestInputValidation:
    """Tests for input validation"""

    def test_age_validation(self):
        """Test age field validation"""
        from pydantic import ValidationError
        from src.api.main import PatientData

        # Valid age
        data = PatientData(
            age=55,
            sex=1,
            cp=2,
            trestbps=130,
            chol=250,
            fbs=0,
            restecg=1,
            thalach=150,
            exang=0,
            oldpeak=1.5,
            slope=2,
            ca=0,
            thal=2,
        )
        assert data.age == 55

        # Invalid age (negative) should raise error
        with pytest.raises(ValidationError):
            PatientData(
                age=-1,
                sex=1,
                cp=2,
                trestbps=130,
                chol=250,
                fbs=0,
                restecg=1,
                thalach=150,
                exang=0,
                oldpeak=1.5,
                slope=2,
                ca=0,
                thal=2,
            )

    def test_sex_validation(self):
        """Test sex field validation"""
        from pydantic import ValidationError
        from src.api.main import PatientData

        # Invalid sex value
        with pytest.raises(ValidationError):
            PatientData(
                age=55,
                sex=2,
                cp=2,
                trestbps=130,
                chol=250,
                fbs=0,
                restecg=1,
                thalach=150,
                exang=0,
                oldpeak=1.5,
                slope=2,
                ca=0,
                thal=2,
            )

    def test_all_fields_required(self):
        """Test that all fields are required"""
        from pydantic import ValidationError
        from src.api.main import PatientData

        with pytest.raises(ValidationError):
            PatientData(age=55)  # Missing other required fields
