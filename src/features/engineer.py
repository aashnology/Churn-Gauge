"""
Churn Gauge – Full Feature Engineering Module
Implements all techniques discussed:
1. Relative / baseline features
2. Velocity / trend proxies
3. Expanded ratio & intensity features
4. Missingness indicators
5. Interaction terms
6. Robust scaling helpers + clean categorical encoding
7. Domain-specific health constructs
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "plan" in df.columns:
        def _clean_plan(s):
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
        df["plan"] = df["plan"].apply(_clean_plan)
    if "industry" in df.columns:
        df["industry"] = df["industry"].fillna("Other").astype(str).str.strip()
        rare = df["industry"].value_counts()
        rare = rare[rare < max(10, 0.005 * len(df))].index
        df.loc[df["industry"].isin(rare), "industry"] = "Other"
    return df


def add_missing_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"is_missing_{c}"] = df[c].isna().astype(int)
    return df


def impute_numeric(df: pd.DataFrame, cols: List[str], strategy: str = "median") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        if strategy == "median":
            fill = df[c].median()
        else:
            fill = df[c].mean()
        df[c] = df[c].fillna(fill)
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Core intensity ratios
    df["api_calls_per_seat"] = np.where(
        df["seats_purchased"] > 0,
        df["api_calls_daily"] / df["seats_purchased"],
        0.0,
    )
    df["tickets_per_seat"] = np.where(
        df["seats_purchased"] > 0,
        df["support_tickets_30d"] / df["seats_purchased"],
        df["support_tickets_30d"],
    )
    df["revenue_per_seat"] = np.where(
        df["seats_purchased"] > 0,
        df["monthly_revenue"] / df["seats_purchased"],
        df["monthly_revenue"],
    )
    # Support burden relative to engagement
    df["tickets_per_login"] = np.where(
        df["login_freq_30d"] > 0,
        df["support_tickets_30d"] / df["login_freq_30d"],
        df["support_tickets_30d"] * 2,  # penalise zero-login + tickets
    )
    # Escalation rate
    df["escalation_rate"] = np.where(
        df["support_tickets_30d"] > 0,
        df["escalated_tickets_30d"] / df["support_tickets_30d"],
        0.0,
    )
    return df


def add_relative_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cohort-relative features (same plan).
    In a real system these would be vs account's own history;
    here we use plan-level medians as a clean proxy.
    """
    df = df.copy()
    for col in ["login_freq_30d", "seat_utilization", "feature_adoption_pct", "api_calls_daily"]:
        if col not in df.columns:
            continue
        med = df.groupby("plan")[col].transform("median")
        df[f"{col}_vs_plan_median"] = df[col] - med
        df[f"{col}_ratio_to_plan"] = np.where(med > 0, df[col] / med, 1.0)
    # Overall percentile rank (0-1) within plan
    for col in ["login_freq_30d", "seat_utilization", "feature_adoption_pct"]:
        if col in df.columns:
            df[f"{col}_plan_pctile"] = df.groupby("plan")[col].rank(pct=True)
    return df


def add_velocity_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snapshot data has no true time series, so we create proxies:
    - Recency pressure from days_to_renewal and days_since_onboarding
    - Engagement intensity relative to tenure
    """
    df = df.copy()
    # Tenure-normalised activity (new accounts expected to be lower)
    df["login_per_tenure_month"] = df["login_freq_30d"] / np.clip(df["days_since_onboarding"] / 30, 0.5, None)
    df["adoption_per_tenure"] = df["feature_adoption_pct"] / np.clip(df["days_since_onboarding"] / 90, 0.3, None)
    # Near-renewal flag and interaction-ready continuous
    df["near_renewal"] = (df["days_to_renewal"] < 60).astype(int)
    df["renewal_urgency"] = 1 - np.clip(df["days_to_renewal"] / 180, 0, 1)
    # Early-life risk (onboarding still in progress)
    df["is_early_life"] = (df["days_since_onboarding"] < 90).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["low_adoption_x_near_renewal"] = (
        (df["feature_adoption_pct"] < 0.35).astype(int) * df["near_renewal"]
    )
    df["high_tickets_x_low_util"] = (
        (df["support_tickets_30d"] >= 3).astype(int)
        * (df["seat_utilization"] < 0.4).astype(int)
    )
    df["low_login_x_low_adoption"] = (
        (df["login_freq_30d"] < 8).astype(int)
        * (df["feature_adoption_pct"] < 0.35).astype(int)
    )
    df["enterprise_x_declining_proxy"] = (
        (df["plan"] == "Enterprise").astype(int)
        * (df["login_freq_30d"] < 12).astype(int)
    )
    # Continuous interaction for trees / linear models
    df["util_x_adoption"] = df["seat_utilization"] * df["feature_adoption_pct"]
    df["tickets_x_renewal_urgency"] = df["support_tickets_30d"] * df["renewal_urgency"]
    return df


def add_domain_health_constructs(df: pd.DataFrame) -> pd.DataFrame:
    """Composite constructs that mirror real CS health-score pillars."""
    df = df.copy()
    # Engagement score (0-1-ish)
    df["engagement_score"] = (
        0.40 * np.clip(df["login_freq_30d"] / 20, 0, 1)
        + 0.35 * df["seat_utilization"]
        + 0.25 * df["feature_adoption_pct"]
    )
    # Support risk score
    df["support_risk_score"] = (
        0.5 * np.clip(df["support_tickets_30d"] / 5, 0, 1)
        + 0.5 * np.clip(df["escalated_tickets_30d"] / 2, 0, 1)
    )
    # Value realisation proxy
    df["value_realisation"] = (
        0.5 * df["feature_adoption_pct"]
        + 0.3 * df["seat_utilization"]
        + 0.2 * np.clip(df["api_calls_per_seat"] / 50, 0, 1)
    )
    # Overall simple health (higher = healthier)
    df["simple_health"] = (
        0.35 * df["engagement_score"]
        + 0.25 * (1 - df["support_risk_score"])
        + 0.20 * np.clip((df["nps_score"] + 100) / 200, 0, 1)
        + 0.20 * df["value_realisation"]
    )
    return df


def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean → missing flags → impute → ratios → relative → velocity → interactions → domain."""
    df = clean_categoricals(df)

    numeric_candidates = [
        "seats_purchased", "days_since_onboarding", "login_freq_30d",
        "seat_utilization", "feature_adoption_pct", "api_calls_daily",
        "support_tickets_30d", "escalated_tickets_30d", "nps_score",
        "csat_score", "days_to_renewal", "monthly_revenue",
    ]
    present_numeric = [c for c in numeric_candidates if c in df.columns]

    df = add_missing_indicators(df, present_numeric)
    df = impute_numeric(df, present_numeric, strategy="median")
    df = add_ratio_features(df)
    df = add_relative_baseline_features(df)
    df = add_velocity_proxies(df)
    df = add_interaction_features(df)
    df = add_domain_health_constructs(df)

    # Final safety: replace any remaining inf / nan
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_features, categorical_features) after engineering."""
    cat = [c for c in ["plan", "industry"] if c in df.columns]
    exclude = set(cat + ["account_id", "churned_90d"])
    num = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return num, cat


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    raw = pd.read_csv(root / "data" / "synthetic" / "churn_gauge_v3_large_messy_20000.csv")
    print("Raw shape:", raw.shape)
    eng = engineer_all(raw)
    print("Engineered shape:", eng.shape)
    num, cat = get_feature_lists(eng)
    print(f"Numeric features: {len(num)}, Categorical: {len(cat)}")
    out = root / "data" / "processed" / "churn_gauge_v3_engineered.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    eng.to_csv(out, index=False)
    print(f"Saved → {out}")
    print("New feature sample:", [c for c in num if c not in raw.columns][:15])
