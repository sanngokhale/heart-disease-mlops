"""
Data Preprocessing Module for Heart Disease Dataset
Handles missing values, encoding, and data cleaning
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from CSV file

    Args:
        file_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality and return summary statistics

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "duplicate_records": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    logger.info("Data Quality Report:")
    logger.info(f"  Total records: {quality_report['total_records']}")
    logger.info(f"  Total missing values: {quality_report['total_missing']}")
    logger.info(f"  Duplicate records: {quality_report['duplicate_records']}")

    return quality_report


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Handle missing values in the dataset

    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'drop')

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        logger.info("No missing values found")
        return df

    logger.info(f"Handling missing values in columns: {missing_cols}")

    for col in missing_cols:
        missing_count = df[col].isnull().sum()

        if strategy == "drop":
            df = df.dropna(subset=[col])
            logger.info(f"  Dropped {missing_count} rows with missing {col}")
        elif strategy == "median":
            if df[col].dtype in ["float64", "int64"]:
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Filled {col} with median: {fill_value}")
            else:
                fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Filled {col} with mode: {fill_value}")
        elif strategy == "mean":
            if df[col].dtype in ["float64", "int64"]:
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Filled {col} with mean: {fill_value:.2f}")
        elif strategy == "mode":
            fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
            df[col].fillna(fill_value, inplace=True)
            logger.info(f"  Filled {col} with mode: {fill_value}")

    return df


def encode_target(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """
    Convert multi-class target to binary classification
    0 = no heart disease, 1 = has heart disease (original values 1-4)

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        DataFrame with binary encoded target
    """
    df = df.copy()

    original_values = df[target_col].unique()
    logger.info(f"Original target values: {sorted(original_values)}")

    # Convert to binary: 0 = no disease, >0 = disease
    df[target_col] = (df[target_col] > 0).astype(int)

    new_values = df[target_col].value_counts()
    logger.info("Binary target distribution:")
    logger.info(f"  No disease (0): {new_values.get(0, 0)}")
    logger.info(f"  Has disease (1): {new_values.get(1, 0)}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)

    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate records")
    else:
        logger.info("No duplicate records found")

    return df


def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types for all columns

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with corrected data types
    """
    df = df.copy()

    # Numeric columns that should be integers
    int_columns = [
        "age",
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    # Numeric columns that should be floats
    float_columns = ["trestbps", "chol", "thalach", "oldpeak"]

    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    logger.info("Data types validated and corrected")
    return df


def preprocess_data(
    input_path: str, output_dir: str = None, missing_strategy: str = "median"
) -> Tuple[pd.DataFrame, str]:
    """
    Main preprocessing pipeline

    Args:
        input_path: Path to raw data CSV
        output_dir: Directory to save processed data
        missing_strategy: Strategy for handling missing values

    Returns:
        Tuple of (processed DataFrame, output file path)
    """
    logger.info("=" * 50)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("=" * 50)

    # Determine output directory
    if output_dir is None:
        input_path_obj = Path(input_path)
        output_dir = input_path_obj.parent.parent / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_path)

    # Check initial quality
    logger.info("\n--- Initial Data Quality ---")
    _ = check_data_quality(df)

    # Remove duplicates
    logger.info("\n--- Removing Duplicates ---")
    df = remove_duplicates(df)

    # Handle missing values
    logger.info("\n--- Handling Missing Values ---")
    df = handle_missing_values(df, strategy=missing_strategy)

    # Validate data types
    logger.info("\n--- Validating Data Types ---")
    df = validate_data_types(df)

    # Handle any remaining missing after type conversion
    df = handle_missing_values(df, strategy=missing_strategy)

    # Encode target variable
    logger.info("\n--- Encoding Target Variable ---")
    df = encode_target(df)

    # Final quality check
    logger.info("\n--- Final Data Quality ---")
    _ = check_data_quality(df)

    # Save processed data
    output_path = output_dir / "heart_disease_preprocessed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Processed data saved to: {output_path}")
    logger.info(f"Final shape: {df.shape}")

    logger.info("=" * 50)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 50)

    return df, str(output_path)


if __name__ == "__main__":
    import sys

    # Get input path from command line or use default
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Default path
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        input_path = project_root / "data" / "raw" / "heart_disease.csv"

    if not Path(input_path).exists():
        print(f"❌ Data file not found: {input_path}")
        print("Please run download_data.py first")
        sys.exit(1)

    df, output_path = preprocess_data(str(input_path))
    print("\n✅ Preprocessing complete!")
    print(f"Output saved to: {output_path}")
