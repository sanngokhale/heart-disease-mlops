"""
Feature Engineering Pipeline for Heart Disease Prediction
Handles feature scaling, encoding, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import logging
from typing import Tuple, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define feature types
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"


def get_feature_names() -> Tuple[List[str], List[str]]:
    """
    Get the numeric and categorical feature names

    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    return NUMERIC_FEATURES.copy(), CATEGORICAL_FEATURES.copy()


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create sklearn preprocessing pipeline for feature transformation

    Returns:
        ColumnTransformer object for preprocessing
    """
    # Numeric features pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical features pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
            ),
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    logger.info("Created preprocessing pipeline")
    logger.info(f"  Numeric features ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    logger.info(
        f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}"
    )

    return preprocessor


def prepare_features(
    df: pd.DataFrame, preprocessor: Optional[ColumnTransformer] = None, fit: bool = True
) -> Tuple[np.ndarray, pd.Series, ColumnTransformer]:
    """
    Prepare features for model training/prediction

    Args:
        df: Input DataFrame with all features and target
        preprocessor: Optional pre-fitted preprocessor
        fit: Whether to fit the preprocessor (True for training, False for inference)

    Returns:
        Tuple of (transformed features, target values, preprocessor)
    """
    # Validate columns
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    logger.info(f"Preparing features from {len(df)} samples")

    # Create or use existing preprocessor
    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()

    # Transform features
    if fit:
        X_transformed = preprocessor.fit_transform(X)
        logger.info(f"Fitted and transformed features: shape = {X_transformed.shape}")
    else:
        X_transformed = preprocessor.transform(X)
        logger.info(f"Transformed features: shape = {X_transformed.shape}")

    return X_transformed, y, preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get output feature names from fitted preprocessor

    Args:
        preprocessor: Fitted ColumnTransformer

    Returns:
        List of feature names
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions
        feature_names = []

        # Numeric features (standardized)
        feature_names.extend([f"num_{f}" for f in NUMERIC_FEATURES])

        # Categorical features (one-hot encoded) - approximate names
        for cat_feat in CATEGORICAL_FEATURES:
            feature_names.append(f"cat_{cat_feat}")

        return feature_names


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str) -> None:
    """
    Save fitted preprocessor to disk

    Args:
        preprocessor: Fitted ColumnTransformer
        output_path: Path to save the preprocessor
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logger.info(f"Preprocessor saved to: {output_path}")


def load_preprocessor(input_path: str) -> ColumnTransformer:
    """
    Load preprocessor from disk

    Args:
        input_path: Path to the saved preprocessor

    Returns:
        Loaded ColumnTransformer
    """
    preprocessor = joblib.load(input_path)
    logger.info(f"Preprocessor loaded from: {input_path}")
    return preprocessor


def transform_single_sample(
    sample: dict, preprocessor: ColumnTransformer
) -> np.ndarray:
    """
    Transform a single sample for prediction

    Args:
        sample: Dictionary with feature values
        preprocessor: Fitted ColumnTransformer

    Returns:
        Transformed feature array
    """
    # Convert to DataFrame
    df = pd.DataFrame([sample])

    # Ensure column order
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # Transform
    X_transformed = preprocessor.transform(df)

    return X_transformed


if __name__ == "__main__":
    # Test the pipeline
    # Load sample data
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_path = project_root / "data" / "processed" / "heart_disease_preprocessed.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples")

        # Prepare features
        X, y, preprocessor = prepare_features(df, fit=True)
        print(f"Transformed features shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")

        # Save preprocessor
        output_dir = project_root / "models"
        save_preprocessor(preprocessor, output_dir / "preprocessor.joblib")
        print("Preprocessor saved!")
    else:
        print(f"Data file not found: {data_path}")
        print("Please run download_data.py and preprocess.py first")
