"""
Churn Gauge – Adaptive Launcher (Layer 4 foundation)
Asks the user about their requirement / technical comfort, then recommends
(and can later auto-launch) the most suitable framework: Streamlit, Gradio or Dash.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Churn Gauge", page_icon="📉", layout="wide")

st.title("Churn Gauge")
st.caption("Customer Health Predictor for SaaS – Adaptive Demo")

# ---------- Expertise / requirement intake ----------
st.header("1. Tell us about your needs")

expertise = st.radio(
    "How would you describe your technical comfort level?",
    options=[
        "Complete beginner – I just want to upload a CSV and see results",
        "Some experience – I can run simple Python apps",
        "Comfortable – I want full control and rich dashboards",
        "Advanced – I may want to embed or call via API later",
    ],
    index=0,
)

primary_need = st.selectbox(
    "What is the main thing you want to do right now?",
    options=[
        "Quick prediction demo on sample or my own data",
        "Explore feature importance and why accounts are at risk",
        "Build / customise a full customer-health dashboard",
        "Integrate predictions into another system",
    ],
)

# Simple decision logic
def recommend_host(expertise: str, need: str) -> tuple[str, str]:
    if "Complete beginner" in expertise or "Quick prediction" in need:
        return "Gradio", "Fastest path for upload → predict → see actions. Minimal code surface."
    if "full customer-health dashboard" in need or "Comfortable" in expertise:
        return "Streamlit", "Best balance of speed and rich multi-chart dashboards. You already use Streamlit."
    if "Advanced" in expertise or "Integrate" in need:
        return "Dash", "Most production-grade control and callback flexibility for embedding."
    return "Streamlit", "Default solid choice for interactive SaaS health exploration."

host, reason = recommend_host(expertise, primary_need)

st.success(f"**Recommended host: {host}**\n\n{reason}")

st.info(
    "In later versions the launcher will automatically start the chosen framework. "
    "For now the core prediction demo below runs inside this Streamlit shell so you can test immediately."
)

# ---------- Core demo (works for all expertise levels) ----------
st.header("2. Predict churn risk & get recommended actions")

from src.actions.recommender import recommend_actions, risk_band
import joblib
import numpy as np

MODEL_PATH = ROOT / "models" / "xgb_shap_v2_medium.joblib"
SAMPLE_PATH = ROOT / "data" / "synthetic" / "churn_gauge_v2_medium_9000.csv"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_sample(n=5):
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH).sample(n, random_state=42)
    return None

pipe = load_model()
sample_df = load_sample()

use_sample = st.checkbox("Use built-in sample accounts (recommended for first try)", value=True)

if use_sample and sample_df is not None:
    st.dataframe(sample_df[["account_id", "plan", "login_freq_30d", "seat_utilization",
                            "feature_adoption_pct", "support_tickets_30d", "churned_90d"]],
                 use_container_width=True)
    selected = st.selectbox("Pick an account to score", sample_df["account_id"].tolist())
    row = sample_df[sample_df["account_id"] == selected].iloc[0]
else:
    st.write("Upload path will be fully enabled after quality gate (Layer 4). For now use sample.")
    row = None

if row is not None and pipe is not None:
    # Prepare single-row features
    numeric_features = [
        "seats_purchased", "days_since_onboarding", "login_freq_30d",
        "seat_utilization", "feature_adoption_pct", "api_calls_daily",
        "support_tickets_30d", "escalated_tickets_30d", "nps_score",
        "csat_score", "days_to_renewal", "monthly_revenue",
    ]
    categorical_features = ["plan", "industry"]
    X_one = row[numeric_features + categorical_features].to_frame().T
    # Fill any remaining NaNs
    for c in numeric_features:
        if pd.isna(X_one[c].iloc[0]):
            X_one[c] = 0.0

    X_t = pipe["preprocessor"].transform(X_one)
    prob = float(pipe["model"].predict_proba(X_t)[0, 1])
    band = risk_band(prob)

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Risk Band", band)
    col3.metric("Actual (sample)", "Churned" if row["churned_90d"] == 1 else "Retained")

    # Simple driver heuristic for demo (full SHAP per-row can be added later)
    drivers = []
    if row["login_freq_30d"] < 8:
        drivers.append("login_freq_30d")
    if row.get("seat_utilization", 1) < 0.4:
        drivers.append("seat_utilization")
    if row.get("feature_adoption_pct", 1) < 0.35:
        drivers.append("feature_adoption_pct")
    if row["support_tickets_30d"] >= 3:
        drivers.append("support_tickets_30d")
    if row["escalated_tickets_30d"] >= 1:
        drivers.append("escalated_tickets_30d")
    if row.get("days_to_renewal", 365) < 60:
        drivers.append("days_to_renewal")
    if row["plan"] == "Enterprise" and prob >= 0.45:
        drivers.append("plan_Enterprise")

    recs = recommend_actions(row, prob, top_drivers=drivers or None)
    st.subheader("Recommended Actions")
    if recs:
        for r in recs:
            st.markdown(f"**{r['action']}**  \n*{r['rationale']}*  \nTriggered by: `{r.get('triggered_by', 'risk band')}`")
    else:
        st.write("Account appears healthy – consider expansion play if utilisation is high.")

st.markdown("---")
st.caption("Churn Gauge · Layer 2 complete · Adaptive host recommendation active · Full multi-framework launch coming next")
