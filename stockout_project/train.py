"""
train_model.py
Melatih model klasifikasi Stockout Probability menggunakan XGBoost,
lalu mengkalibrasi probabilitasnya dengan CalibratedClassifierCV
agar output predict_proba() valid secara statistik untuk manajemen risiko.

Jalankan: python train_model.py
Output  : model_artifacts.joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    classification_report,
)
from xgboost import XGBClassifier

DATA_PATH = "inventory_dataset.csv"
ARTIFACT_PATH = "model_artifacts.joblib"

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


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle sedikit imbalance lewat scale_pos_weight, bukan resampling,
    # supaya distribusi asli tetap terjaga untuk kalibrasi probabilitas.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    # Kalibrasi probabilitas (isotonic cocok untuk dataset >1000 baris)
    calibrated_model = CalibratedClassifierCV(
        base_model, method="isotonic", cv=5
    )
    calibrated_model.fit(X_train, y_train)

    # Model "mentah" terpisah (tanpa kalibrasi) khusus untuk SHAP,
    # karena SHAP TreeExplainer butuh akses langsung ke struktur pohon XGBoost.
    shap_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    shap_model.fit(X_train, y_train)

    # Evaluasi
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("=== Evaluasi Model (Test Set) ===")
    print(f"ROC-AUC       : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Brier Score   : {brier_score_loss(y_test, y_proba):.4f}  (makin rendah makin baik-terkalibrasi)")
    print(classification_report(y_test, y_pred))

    joblib.dump(
        {
            "calibrated_model": calibrated_model,
            "shap_model": shap_model,
            "features": FEATURES,
        },
        ARTIFACT_PATH,
    )
    print(f"\nArtifact tersimpan di: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()