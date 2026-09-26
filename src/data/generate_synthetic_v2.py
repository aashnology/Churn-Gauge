"""
Churn Gauge - Phase 2 Synthetic Dataset Generator
Milestone: Layer 2 - Medium complexity, controlled noise, mild outliers, ~5-7% missing values.
~9,000 accounts. Builds on v1 signals but adds realism.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_medium_saas_dataset(n_accounts: int = 9000) -> pd.DataFrame:
    """Generate medium-complexity synthetic SaaS usage dataset with controlled messiness."""
    account_id = [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)]

    plan = RNG.choice(["Starter", "Growth", "Enterprise"], size=n_accounts, p=[0.42, 0.38, 0.20])
    industry = RNG.choice(
        ["SaaS", "FinTech", "E-commerce", "Healthcare", "Other"],
        size=n_accounts,
        p=[0.28, 0.22, 0.18, 0.16, 0.16],
    )
    seats_purchased = np.where(
        plan == "Starter",
        RNG.integers(1, 10, n_accounts),
        np.where(plan == "Growth", RNG.integers(5, 40, n_accounts), RNG.integers(15, 200, n_accounts)),
    ).astype(int)

    days_since_onboarding = RNG.integers(14, 900, n_accounts)

    # Usage signals with slightly higher variance
    base_login = np.where(plan == "Starter", 7, np.where(plan == "Growth", 13, 20))
    login_freq_30d = np.clip(RNG.normal(base_login, 4.5, n_accounts), 0, 30).round().astype(int)

    seat_utilization = np.clip(
        np.where(plan == "Starter", RNG.normal(0.45, 0.18, n_accounts),
                 np.where(plan == "Growth", RNG.normal(0.58, 0.16, n_accounts),
                          RNG.normal(0.75, 0.14, n_accounts))),
        0.02, 1.0,
    )

    feature_adoption_pct = np.clip(
        np.where(plan == "Starter", RNG.normal(0.32, 0.14, n_accounts),
                 np.where(plan == "Growth", RNG.normal(0.52, 0.15, n_accounts),
                          RNG.normal(0.70, 0.13, n_accounts))),
        0.02, 1.0,
    )

    api_calls_daily = np.clip(
        np.where(plan == "Starter", RNG.normal(35, 30, n_accounts),
                 np.where(plan == "Growth", RNG.normal(160, 90, n_accounts),
                          RNG.normal(600, 280, n_accounts))),
        0, None,
    ).astype(int)

    support_tickets_30d = RNG.poisson(1.5, n_accounts)
    escalated_tickets_30d = np.where(support_tickets_30d > 0, RNG.binomial(support_tickets_30d, 0.18), 0)

    nps_score = np.clip(RNG.normal(38, 25, n_accounts), -100, 100).round().astype(int)
    csat_score = np.clip(RNG.normal(3.7, 0.85, n_accounts), 1.0, 5.0).round(1)

    days_to_renewal = RNG.integers(3, 400, n_accounts)
    monthly_revenue = np.where(
        plan == "Starter",
        seats_purchased * RNG.uniform(10, 28, n_accounts),
        np.where(plan == "Growth", seats_purchased * RNG.uniform(25, 60, n_accounts),
                 seats_purchased * RNG.uniform(55, 140, n_accounts)),
    ).round(2)

    # Stronger but still clean-ish risk signal + moderate noise
    risk_score = (
        -0.50 * (login_freq_30d / 30)
        -0.42 * seat_utilization
        -0.38 * feature_adoption_pct
        +0.28 * (support_tickets_30d / 4)
        +0.22 * (escalated_tickets_30d / 1.5)
        -0.20 * ((nps_score + 100) / 200)
        -0.14 * (csat_score / 5)
        +0.16 * (1 - np.clip(days_to_renewal / 100, 0, 1))
        + RNG.normal(0, 0.10, n_accounts)
    )
    churn_prob = 1 / (1 + np.exp(-4.2 * (risk_score + 0.02)))
    churned_90d = (RNG.random(n_accounts) < churn_prob).astype(int)

    df = pd.DataFrame({
        "account_id": account_id,
        "plan": plan,
        "industry": industry,
        "seats_purchased": seats_purchased,
        "days_since_onboarding": days_since_onboarding,
        "login_freq_30d": login_freq_30d,
        "seat_utilization": seat_utilization.round(3),
        "feature_adoption_pct": feature_adoption_pct.round(3),
        "api_calls_daily": api_calls_daily,
        "support_tickets_30d": support_tickets_30d,
        "escalated_tickets_30d": escalated_tickets_30d,
        "nps_score": nps_score,
        "csat_score": csat_score,
        "days_to_renewal": days_to_renewal,
        "monthly_revenue": monthly_revenue,
        "churned_90d": churned_90d,
    })

    # --- Controlled messiness ---
    # 1. Mild outliers (clip extreme values then re-introduce a few extremes)
    for col in ["api_calls_daily", "monthly_revenue", "seats_purchased"]:
        q99 = df[col].quantile(0.99)
        outlier_idx = RNG.choice(df.index, size=int(0.015 * n_accounts), replace=False)
        df.loc[outlier_idx, col] = (q99 * RNG.uniform(1.8, 3.5, len(outlier_idx))).astype(df[col].dtype)

    # 2. Missing values (~6% overall on selected columns)
    missing_cols = ["nps_score", "csat_score", "feature_adoption_pct", "seat_utilization", "api_calls_daily"]
    for col in missing_cols:
        miss_idx = RNG.choice(df.index, size=int(0.06 * n_accounts), replace=False)
        df.loc[miss_idx, col] = np.nan

    # 3. A few inconsistent plan/seat combinations (messiness)
    bad_idx = RNG.choice(df.index, size=int(0.01 * n_accounts), replace=False)
    df.loc[bad_idx, "seats_purchased"] = RNG.integers(1, 5, len(bad_idx))

    return df


if __name__ == "__main__":
    df = generate_medium_saas_dataset(9000)
    out_path = OUT_DIR / "churn_gauge_v2_medium_9000.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows → {out_path}")
    print(f"Churn rate: {df['churned_90d'].mean():.1%}")
    print(f"Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(df.describe(include="all").T[["count", "mean", "std"]].head(12))
