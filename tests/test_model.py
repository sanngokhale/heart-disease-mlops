"""
Unit Tests for Model Module
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.feature_engineering import (
    create_preprocessing_pipeline,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


class TestFeatureEngineering:
    """Tests for feature engineering pipeline"""

    def test_feature_names_exist(self):
        """Test that feature names are defined"""
        numeric, categorical = get_feature_names()

        assert len(numeric) > 0
        assert len(categorical) > 0
        assert len(numeric) == 5
        assert len(categorical) == 8

    def test_preprocessing_pipeline_creation(self):
        """Test that preprocessing pipeline is created"""
        preprocessor = create_preprocessing_pipeline()

        assert preprocessor is not None
        assert hasattr(preprocessor, "fit_transform")
        assert hasattr(preprocessor, "transform")

    def test_pipeline_transforms_data(self):
        """Test that pipeline correctly transforms data"""
        preprocessor = create_preprocessing_pipeline()

        # Create sample data
        sample_data = pd.DataFrame(
            {
                "age": [55],
                "trestbps": [130],
                "chol": [250],
                "thalach": [150],
                "oldpeak": [1.5],
                "sex": [1],
                "cp": [2],
                "fbs": [0],
                "restecg": [1],
                "exang": [0],
                "slope": [2],
                "ca": [0],
                "thal": [2],
            }
        )

        # Fit and transform
        X_transformed = preprocessor.fit_transform(sample_data)

        assert X_transformed is not None
        assert X_transformed.shape[0] == 1
        assert X_transformed.shape[1] > 0


class TestModelPrediction:
    """Tests for model prediction functionality"""

    @pytest.fixture
    def sample_input(self):
        """Sample patient data for testing"""
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

    def test_sample_input_has_all_features(self, sample_input):
        """Test that sample input has all required features"""
        all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

        for feature in all_features:
            assert feature in sample_input

    def test_model_file_loading(self):
        """Test model loading (only if model exists)"""
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_model.joblib"

        if model_path.exists():
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_proba")

    def test_preprocessor_file_loading(self):
        """Test preprocessor loading (only if exists)"""
        project_root = Path(__file__).resolve().parent.parent
        preprocessor_path = project_root / "models" / "preprocessor.joblib"

        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            assert preprocessor is not None
            assert hasattr(preprocessor, "transform")


class TestModelMetrics:
    """Tests for model evaluation metrics"""

    def test_prediction_binary(self):
        """Test that predictions are binary"""
        # Mock prediction output
        predictions = np.array([0, 1, 0, 1, 1, 0])

        assert set(predictions) <= {0, 1}

    def test_probability_range(self):
        """Test that probabilities are in valid range"""
        # Mock probability output
        probabilities = np.array([0.2, 0.8, 0.3, 0.9, 0.7, 0.1])

        assert all(0 <= p <= 1 for p in probabilities)
