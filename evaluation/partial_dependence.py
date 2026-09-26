"""
Churn Gauge – Partial Dependence Plot investigation
Computes 1-way PDPs for the most important features using the engineered XGBoost model.
Saves numeric PDP data + matplotlib figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_v3_engineered.joblib"
DATA_PATH = ROOT / "data" / "processed" / "churn_gauge_v3_engineered.csv"
OUT_DIR = ROOT / "evaluation" / "pdp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading model and data...")
    bundle = joblib.load(MODEL_PATH)
    pre = bundle["preprocessor"]
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    df = pd.read_csv(DATA_PATH)
    # Reconstruct the same feature matrix the model saw
    # We need the original column order used at training time
    # From engineer: numeric list + cat
    from src.features.engineer import get_feature_lists
    num, cat = get_feature_lists(df)
    num = [c for c in num if df[c].nunique() > 1]
    X = df[num + cat]

    # Transform
    X_t = pre.transform(X)

    # Key features to examine (consensus + engineered)
    candidates = [
        "login_freq_30d",
        "seat_utilization",
        "feature_adoption_pct",
        "support_tickets_30d",
        "escalated_tickets_30d",
        "simple_health",
        "engagement_score",
        "near_renewal",
        "days_to_renewal",
        "api_calls_per_seat",
    ]
    # Map to column indices in the transformed matrix
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    features_to_plot = []
    for c in candidates:
        if c in name_to_idx:
            features_to_plot.append((c, name_to_idx[c]))
        else:
            print(f"  (skip {c} – not in model features)")

    print(f"Computing PDPs for: {[f[0] for f in features_to_plot]}")

    # Compute and plot one-by-one for clarity
    results = {}
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()

    for ax_i, (fname, idx) in enumerate(features_to_plot):
        pd_result = partial_dependence(
            model, X_t, features=[idx],
            kind="average", grid_resolution=40,
        )
        avg = pd_result["average"][0]
        grid = pd_result["grid_values"][0]

        results[fname] = {"grid": grid.tolist(), "average": avg.tolist()}

        axes[ax_i].plot(grid, avg, color="#1f77b4", lw=2)
        axes[ax_i].set_title(fname, fontsize=10)
        axes[ax_i].set_xlabel("Feature value")
        axes[ax_i].set_ylabel("PD (avg pred)")
        axes[ax_i].grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(features_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Partial Dependence – Churn Gauge XGBoost (engineered v3)", fontsize=14)
    plt.tight_layout()
    fig_path = OUT_DIR / "pdp_1way_key_features.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {fig_path}")

    # Save numeric data
    import json
    with open(OUT_DIR / "pdp_values.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Numeric PDP values → {OUT_DIR / 'pdp_values.json'}")

    # Short textual summary of directionality
    print("\n=== PDP direction summary (higher PD = higher predicted churn risk) ===")
    for fname, data in results.items():
        grid = np.array(data["grid"])
        avg = np.array(data["average"])
        # Simple slope from first to last quartile
        n = len(avg)
        slope = avg[-1] - avg[0]
        direction = "↑ risk as value increases" if slope > 0.01 else ("↓ risk as value increases" if slope < -0.01 else "mostly flat")
        print(f"  {fname:30s}  {direction}  (Δ PD ≈ {slope:+.3f})")


if __name__ == "__main__":
    main()
