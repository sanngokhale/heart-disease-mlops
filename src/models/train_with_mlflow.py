"""
Model Training with MLflow Experiment Tracking
Trains and evaluates classification models for heart disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime
import logging
import os
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.feature_engineering import prepare_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow configuration
EXPERIMENT_NAME = "heart-disease-classification"
TRACKING_URI = "file:./mlruns"


def setup_mlflow():
    """Configure MLflow tracking"""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {TRACKING_URI}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_dir: str
) -> str:
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    save_path = os.path.join(
        save_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_dir: str
) -> str:
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(
        save_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


def plot_feature_importance(
    model, feature_names: list, model_name: str, save_dir: str
) -> str:
    """Plot feature importance for tree-based models"""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(
        range(len(indices)),
        [feature_names[i] if i < len(feature_names) else f"F{i}" for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()

    save_path = os.path.join(
        save_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics, y_pred, y_prob


def train_and_log_model(
    model,
    model_name: str,
    params: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None,
) -> tuple:
    """Train model and log to MLflow"""

    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*50}")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        metrics["cv_roc_auc_mean"] = cv_scores.mean()
        metrics["cv_roc_auc_std"] = cv_scores.std()

        # Log metrics
        mlflow.log_metrics(metrics)

        # Create plots directory
        plots_dir = "temp_plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Generate and log plots
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name, plots_dir)
        mlflow.log_artifact(cm_path)

        roc_path = plot_roc_curve(y_test, y_prob, model_name, plots_dir)
        mlflow.log_artifact(roc_path)

        if feature_names:
            fi_path = plot_feature_importance(
                model, feature_names, model_name, plots_dir
            )
            if fi_path:
                mlflow.log_artifact(fi_path)

        # Log model
        mlflow.sklearn.log_model(model, model_name.lower().replace(" ", "_"))

        # Clean up temp plots
        for f in os.listdir(plots_dir):
            os.remove(os.path.join(plots_dir, f))
        os.rmdir(plots_dir)

        # Print results
        logger.info(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return model, metrics


def get_models_and_params() -> list:
    """Define models and their hyperparameters for training"""
    models = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        },
        {
            "name": "Gradient Boosting",
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        },
    ]
    return models


def run_experiments(data_path: str = None) -> dict:
    """
    Run all experiments with MLflow tracking

    Args:
        data_path: Path to preprocessed data CSV

    Returns:
        Dictionary with results for all models
    """
    # Setup MLflow
    setup_mlflow()

    # Determine data path
    if data_path is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        data_path = (
            project_root / "data" / "processed" / "heart_disease_preprocessed.csv"
        )

    logger.info(f"Loading data from: {data_path}")

    # Load and prepare data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Prepare features
    X, y, preprocessor = prepare_features(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(y_train)} samples")
    logger.info(f"Test set: {len(y_test)} samples")
    logger.info(f"Class distribution (train): {dict(y_train.value_counts())}")

    # Get feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = None

    # Train models
    results = {}
    models_config = get_models_and_params()

    for config in models_config:
        model, metrics = train_and_log_model(
            model=config["model"],
            model_name=config["name"],
            params=config["params"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
        )
        results[config["name"]] = {"model": model, "metrics": metrics}

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]["metrics"]["roc_auc"])
    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]

    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    logger.info(f"{'='*50}")

    # Save best model and preprocessor
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    # Save model info
    model_info = {
        "model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "metrics": {k: float(v) for k, v in best_metrics.items()},
        "training_date": datetime.now().isoformat(),
        "feature_count": X.shape[1],
        "training_samples": len(y_train),
        "test_samples": len(y_test),
    }

    with open(models_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)

    logger.info(f"\n✅ Best model saved to: {model_path}")
    logger.info(f"✅ Preprocessor saved to: {preprocessor_path}")
    logger.info(f"✅ Model info saved to: {models_dir / 'model_info.json'}")

    return results


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "=" * 60)
    print("  Heart Disease Prediction - Model Training with MLflow")
    print("=" * 60 + "\n")

    results = run_experiments(data_path)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("  View experiments: mlflow ui --port 5000")
    print("=" * 60 + "\n")
