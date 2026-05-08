"""
Data Download Script for Heart Disease UCI Dataset
Downloads the processed Cleveland dataset from UCI ML Repository
"""

import pandas as pd
import requests
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Column names as per UCI documentation
COLUMN_NAMES = [
    "age",  # Age in years
    "sex",  # Sex (1 = male; 0 = female)
    "cp",  # Chest pain type (0-3)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",  # Serum cholesterol (mg/dl)
    "fbs",  # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",  # Resting ECG results (0-2)
    "thalach",  # Maximum heart rate achieved
    "exang",  # Exercise induced angina (1 = yes; 0 = no)
    "oldpeak",  # ST depression induced by exercise
    "slope",  # Slope of peak exercise ST segment (0-2)
    "ca",  # Number of major vessels colored by fluoroscopy (0-3)
    "thal",  # Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)
    "target",  # Diagnosis of heart disease (0 = no disease; 1-4 = disease)
]


def download_heart_disease_data(output_dir: str = None) -> str:
    """
    Download the Heart Disease UCI dataset (Cleveland)

    Args:
        output_dir: Directory to save the data (default: data/raw relative to script)

    Returns:
        Path to the saved CSV file
    """
    # Determine output directory
    if output_dir is None:
        # Get the project root (2 levels up from this script)
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "data" / "raw"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # URL for the processed Cleveland dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    logger.info("Downloading Heart Disease dataset from UCI ML Repository...")
    logger.info(f"URL: {url}")

    try:
        # Download the data
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save raw data file
        raw_data_path = output_dir / "processed.cleveland.data"
        with open(raw_data_path, "w") as f:
            f.write(response.text)
        logger.info(f"Raw data saved to: {raw_data_path}")

        # Load into pandas DataFrame
        df = pd.read_csv(raw_data_path, names=COLUMN_NAMES, na_values="?")

        # Save as CSV with proper headers
        output_path = output_dir / "heart_disease.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing values per column:\n{df.isnull().sum()}")

        return str(output_path)

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


def load_local_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load heart disease data from local UCI data files
    (Using the data already present in the workspace)

    Args:
        data_dir: Directory containing the heart disease data

    Returns:
        DataFrame with the heart disease data
    """
    if data_dir is None:
        # Try to find the data in the workspace
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent  # Go up to MLOps folder
        data_dir = project_root / "heart+disease"
    else:
        data_dir = Path(data_dir)

    # Check for cleve.mod file (Cleveland data)
    cleve_path = data_dir / "cleve.mod"

    if cleve_path.exists():
        logger.info(f"Loading local data from: {cleve_path}")
        df = pd.read_csv(cleve_path, names=COLUMN_NAMES, na_values="?")
        return df

    raise FileNotFoundError(f"Data not found in {data_dir}")


if __name__ == "__main__":
    # Try downloading from UCI first
    try:
        output_path = download_heart_disease_data()
        print(f"\n✅ Dataset successfully downloaded to: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Could not download from UCI: {e}")
        print("Attempting to use local data...")

        try:
            df = load_local_data()
            # Save to project data folder
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent.parent
            output_dir = project_root / "data" / "raw"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / "heart_disease.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Local data saved to: {output_path}")
            print(f"Shape: {df.shape}")
        except Exception as e2:
            print(f"❌ Failed to load local data: {e2}")
