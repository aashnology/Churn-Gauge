"""
Churn Gauge Layer 2 – Gradient Boosting (sklearn HistGradientBoosting) on medium data.
Fallback while XGBoost install is blocked by PyPI issues. SHAP-compatible later.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "synthetic" / "churn_gauge_v2_medium_9000.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows. Churn rate: {df['churned_90d'].mean():.1%}")

    target = "churned_90d"
    numeric_features = [
        "seats_purchased", "days_since_onboarding", "login_freq_30d",
        "seat_utilization", "feature_adoption_pct", "api_calls_daily",
        "support_tickets_30d", "escalated_tickets_30d", "nps_score",
        "csat_score", "days_to_renewal", "monthly_revenue",
    ]
    categorical_features = ["plan", "industry"]

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # HistGradientBoosting handles NaNs natively; only need to encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n=== Layer 2 HistGradientBoosting Results ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance (permutation-style not needed; use native if available)
    # HistGBM does not expose classic feature_importances_ the same way; we will use SHAP later.
    joblib.dump(clf, MODEL_DIR / "hgb_v2_medium.joblib")

    metrics = {
        "model": "HistGradientBoosting_v2",
        "dataset": "churn_gauge_v2_medium_9000",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": float(auc),
        "churn_rate_test": float(y_test.mean()),
        "note": "XGBoost blocked by PyPI; using sklearn HistGBM as interim boosting model",
    }
    with open(MODEL_DIR / "hgb_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR}")
    return auc


if __name__ == "__main__":
    main()
