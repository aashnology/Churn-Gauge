"""
Churn Gauge Layer 2 – XGBoost + SHAP on medium-complexity data.
Produces global feature importance and saves SHAP summary data for later comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import shap
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

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

    # Simple median fill for numeric NaNs (XGBoost can handle but we keep pipeline consistent)
    for col in numeric_features:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # Fit preprocessor first so we can get feature names for SHAP
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    ohe = preprocessor.named_transformers_["cat"]
    feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="auc",
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train)

    y_prob = model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== Layer 2 XGBoost Results ===")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 12 XGBoost feature importances:")
    print(imp.head(12).to_string(index=False))

    # SHAP (TreeExplainer)
    print("\nComputing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    sample_idx = np.random.choice(X_test_t.shape[0], size=min(500, X_test_t.shape[0]), replace=False)
    shap_values = explainer.shap_values(X_test_t[sample_idx])

    # Mean absolute SHAP for global ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_rank = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 12 features by mean |SHAP|:")
    print(shap_rank.head(12).to_string(index=False))

    # Persist everything
    full_pipe = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": feature_names,
    }
    joblib.dump(full_pipe, MODEL_DIR / "xgb_shap_v2_medium.joblib")
    imp.to_csv(MODEL_DIR / "xgb_v2_feature_importance.csv", index=False)
    shap_rank.to_csv(MODEL_DIR / "xgb_v2_shap_global.csv", index=False)

    metrics = {
        "model": "XGBoost_v2",
        "dataset": "churn_gauge_v2_medium_9000",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": float(auc),
        "churn_rate_test": float(y_test.mean()),
        "shap_sample_size": len(sample_idx),
    }
    with open(MODEL_DIR / "xgb_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Side-by-side comparison file
    logreg_coef = pd.read_csv(MODEL_DIR / "logreg_v2_coefficients.csv")
    comparison = shap_rank.merge(
        logreg_coef[["feature", "coefficient"]], on="feature", how="left"
    ).merge(imp, on="feature", how="left")
    comparison.to_csv(MODEL_DIR / "layer2_logreg_vs_xgb_shap_comparison.csv", index=False)
    print(f"\nComparison table saved. Artifacts in {MODEL_DIR}")
    return auc


if __name__ == "__main__":
    main()
