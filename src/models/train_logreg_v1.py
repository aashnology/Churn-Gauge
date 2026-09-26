"""
Churn Gauge Layer 1 – Logistic Regression baseline on clean synthetic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "synthetic" / "churn_gauge_v1_clean_3000.csv"
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n=== Layer 1 Logistic Regression Results ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Coefficient interpretation
    feature_names = (
        numeric_features
        + list(clf.named_steps["preprocessor"]
               .named_transformers_["cat"]
               .get_feature_names_out(categorical_features))
    )
    coefs = clf.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    print("\nTop 12 features by |coefficient| (positive = increases churn risk):")
    print(coef_df.head(12).to_string(index=False))

    # Persist
    joblib.dump(clf, MODEL_DIR / "logreg_v1_clean.joblib")
    coef_df.to_csv(MODEL_DIR / "logreg_v1_coefficients.csv", index=False)

    metrics = {
        "model": "LogisticRegression_v1",
        "dataset": "churn_gauge_v1_clean_3000",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": float(auc),
        "churn_rate_test": float(y_test.mean()),
    }
    with open(MODEL_DIR / "logreg_v1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel + coefficients + metrics saved to {MODEL_DIR}")
    return auc, coef_df


if __name__ == "__main__":
    main()
