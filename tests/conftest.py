"""
Pytest Configuration and Fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["MODEL_PATH"] = str(project_root / "models" / "best_model.joblib")
os.environ["PREPROCESSOR_PATH"] = str(project_root / "models" / "preprocessor.joblib")


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_patient_data():
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


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    import pandas as pd

    return pd.DataFrame(
        {
            "age": [45, 50, 55, 60, 65],
            "sex": [0, 1, 1, 0, 1],
            "cp": [0, 1, 2, 3, 2],
            "trestbps": [120, 130, 140, 135, 125],
            "chol": [200, 250, 300, 275, 225],
            "fbs": [0, 1, 0, 1, 0],
            "restecg": [0, 1, 2, 1, 0],
            "thalach": [150, 140, 130, 145, 155],
            "exang": [0, 0, 1, 1, 0],
            "oldpeak": [0.5, 1.0, 2.0, 1.5, 0.8],
            "slope": [0, 1, 2, 1, 0],
            "ca": [0, 1, 2, 1, 0],
            "thal": [1, 2, 3, 2, 1],
            "target": [0, 1, 1, 1, 0],
        }
    )
