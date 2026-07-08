"""
train_model.py
==============

Train a calibrated XGBoost model for stockout probability prediction.

Responsibilities
----------------
- Load cleaned dataset
- Train XGBoost classifier
- Calibrate predicted probabilities
- Train separate XGBoost model for SHAP
- Evaluate model performance
- Save trained artifacts

Run:
    python train_model.py

Input:
    inventory_dataset_clean.csv

Output:
    model_artifacts.joblib
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "inventory_dataset_clean.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifacts.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "current_stock",
    "avg_daily_sales",
    "sales_volatility",
    "supplier_lead_time_days",
    "safety_stock_level",
    "reorder_point",
    "seasonality_index",
    "historical_stockout_count",
]

TARGET = "will_stockout_next_30_days"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data
# =============================================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """Load cleaned dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}\n"
            "Please run data_prep.py first."
        )

    return pd.read_csv(path)


# =============================================================================
# Model
# =============================================================================


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """Create base XGBoost classifier."""

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(model, X_test, y_test):
    """Evaluate calibrated model."""

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    logger.info("")
    logger.info("=" * 55)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 55)

    logger.info("ROC-AUC        : %.4f", roc_auc_score(y_test, y_prob))
    logger.info("Brier Score    : %.4f", brier_score_loss(y_test, y_prob))
    logger.info("Precision      : %.4f", precision_score(y_test, y_pred))
    logger.info("Recall         : %.4f", recall_score(y_test, y_pred))
    logger.info("F1 Score       : %.4f", f1_score(y_test, y_pred))

    logger.info("")
    logger.info("Confusion Matrix")
    logger.info("\n%s", confusion_matrix(y_test, y_pred))

    logger.info("")
    logger.info("Classification Report")
    logger.info("\n%s", classification_report(y_test, y_pred))


# =============================================================================
# Artifact
# =============================================================================


def save_artifacts(
    calibrated_model,
    shap_model,
    feature_names,
    path: Path,
):
    """Save all model artifacts."""

    artifacts = {
        "calibrated_model": calibrated_model,
        "shap_model": shap_model,
        "features": feature_names,
    }

    joblib.dump(artifacts, path)

    logger.info("")
    logger.info("Artifacts saved to:")
    logger.info("%s", path)


# =============================================================================
# Main
# =============================================================================


def main():

    logger.info("Loading cleaned dataset...")

    df = load_dataset(DATA_PATH)

    logger.info("Rows : %d", len(df))
    logger.info("Columns : %d", len(df.columns))

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    logger.info("")
    logger.info("Train samples : %d", len(X_train))
    logger.info("Test samples  : %d", len(X_test))

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    logger.info("scale_pos_weight : %.3f", scale_pos_weight)

    logger.info("")
    logger.info("Training calibrated model...")

    base_model = build_xgboost(scale_pos_weight)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=5,
    )

    calibrated_model.fit(X_train, y_train)

    logger.info("Training SHAP model...")

    shap_model = build_xgboost(scale_pos_weight)

    shap_model.fit(X_train, y_train)

    evaluate_model(
        calibrated_model,
        X_test,
        y_test,
    )

    save_artifacts(
        calibrated_model,
        shap_model,
        FEATURES,
        ARTIFACT_PATH,
    )

    logger.info("")
    logger.info("=" * 55)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()