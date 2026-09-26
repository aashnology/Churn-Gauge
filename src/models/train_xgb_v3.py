"""
Churn Gauge Layer 3 – Clean messy categoricals then train XGBoost on v3 large dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import joblib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "synthetic" / "churn_gauge_v3_large_messy_20000.csv"
MODEL_DIR = ROOT / "models"


def clean_plan(s: str) -> str:
    if pd.isna(s):
        return "Unknown"
    s = str(s).strip().lower()
    if s in ("starter",):
        return "Starter"
    if s in ("growth",):
        return "Growth"
    if s in ("enterprise", "enterprise "):
        return "Enterprise"
    return "Unknown"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Raw: {len(df)} rows, churn {df['churned_90d'].mean():.1%}")

    df["plan"] = df["plan"].apply(clean_plan)
    print("Plan after clean:\n", df["plan"].value_counts())

    numeric = [
        "seats_purchased", "days_since_onboarding", "login_freq_30d",
        "seat_utilization", "feature_adoption_pct", "api_calls_daily",
        "support_tickets_30d", "escalated_tickets_30d", "nps_score",
        "csat_score", "days_to_renewal", "monthly_revenue",
    ]
    for c in numeric:
        df[c] = df[c].fillna(df[c].median())

    cat = ["plan", "industry"]
    X = df[numeric + cat]
    y = df["churned_90d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    X_tr = pre.fit_transform(X_train)
    X_te = pre.transform(X_test)
    feat_names = numeric + list(pre.named_transformers_["cat"].get_feature_names_out(cat))

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        random_state=42, eval_metric="auc", n_jobs=-1,
    )
    model.fit(X_tr, y_train)

    prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"\nXGBoost v3 ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, (prob >= 0.5).astype(int), target_names=["Retained", "Churned"]))

    imp = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("\nTop features:\n", imp.head(10).to_string(index=False))

    joblib.dump({"preprocessor": pre, "model": model, "feature_names": feat_names},
                MODEL_DIR / "xgb_v3_large.joblib")
    imp.to_csv(MODEL_DIR / "xgb_v3_importance.csv", index=False)
    with open(MODEL_DIR / "xgb_v3_metrics.json", "w") as f:
        json.dump({"model": "XGBoost_v3", "roc_auc": float(auc), "n_train": len(X_train),
                   "n_test": len(X_test), "churn_rate": float(y.mean())}, f, indent=2)
    print("Saved.")


if __name__ == "__main__":
    main()
