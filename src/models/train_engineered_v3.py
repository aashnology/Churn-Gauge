"""
Train LogReg + XGBoost on fully engineered v3 features and compare lift.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import joblib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.engineer import get_feature_lists

DATA = ROOT / "data" / "processed" / "churn_gauge_v3_engineered.csv"
MODEL_DIR = ROOT / "models"


def main():
    df = pd.read_csv(DATA)
    print(f"Loaded engineered: {df.shape}, churn {df['churned_90d'].mean():.1%}")

    num, cat = get_feature_lists(df)
    # Drop extremely sparse missing indicators that are all-zero
    num = [c for c in num if df[c].nunique() > 1]

    X = df[num + cat]
    y = df["churned_90d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----- Logistic Regression -----
    pre_lr = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    lr = Pipeline([
        ("pre", pre_lr),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs")),
    ])
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, prob_lr)
    print(f"\nLogReg (full FE) ROC-AUC: {auc_lr:.4f}")

    # Coefficients
    ohe = lr.named_steps["pre"].named_transformers_["cat"]
    feat_names_lr = num + list(ohe.get_feature_names_out(cat))
    coef = pd.DataFrame({
        "feature": feat_names_lr,
        "coefficient": lr.named_steps["clf"].coef_[0],
    })
    coef["abs"] = coef["coefficient"].abs()
    coef = coef.sort_values("abs", ascending=False)
    print("Top 15 LogReg coefficients:")
    print(coef.head(15).to_string(index=False))

    # ----- XGBoost -----
    pre_xgb = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    X_tr = pre_xgb.fit_transform(X_train)
    X_te = pre_xgb.transform(X_test)
    feat_names_xgb = num + list(pre_xgb.named_transformers_["cat"].get_feature_names_out(cat))

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        random_state=42, eval_metric="auc", n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_train)
    prob_xgb = xgb_model.predict_proba(X_te)[:, 1]
    auc_xgb = roc_auc_score(y_test, prob_xgb)
    print(f"\nXGBoost (full FE) ROC-AUC: {auc_xgb:.4f}")
    print(classification_report(y_test, (prob_xgb >= 0.5).astype(int), target_names=["Retained", "Churned"]))

    imp = pd.DataFrame({"feature": feat_names_xgb, "importance": xgb_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("Top 15 XGBoost importances:")
    print(imp.head(15).to_string(index=False))

    # Persist
    joblib.dump(lr, MODEL_DIR / "logreg_v3_engineered.joblib")
    joblib.dump({"preprocessor": pre_xgb, "model": xgb_model, "feature_names": feat_names_xgb},
                MODEL_DIR / "xgb_v3_engineered.joblib")
    coef.to_csv(MODEL_DIR / "logreg_v3_engineered_coefs.csv", index=False)
    imp.to_csv(MODEL_DIR / "xgb_v3_engineered_importance.csv", index=False)

    metrics = {
        "logreg_auc": float(auc_lr),
        "xgb_auc": float(auc_xgb),
        "n_features_numeric": len(num),
        "n_features_categorical": len(cat),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "previous_xgb_v3_auc_without_full_fe": 0.740,
    }
    with open(MODEL_DIR / "engineered_v3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nLift vs previous XGBoost v3 (no full FE): {auc_xgb - 0.740:+.4f}")
    print("Artifacts saved.")


if __name__ == "__main__":
    main()
