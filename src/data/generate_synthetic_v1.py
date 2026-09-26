"""
Churn Gauge - Phase 1 Synthetic Dataset Generator (Clean, Small, Low Complexity)
Milestone: Layer 1 - Clean baseline data for Logistic Regression
~3,000 accounts, realistic but low outlier rate, clear signal correlations.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_clean_saas_dataset(n_accounts: int = 3000) -> pd.DataFrame:
    """Generate a clean, low-complexity synthetic SaaS usage dataset."""
    account_id = [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)]

    # Plan & firmographics (simple categorical)
    plan = RNG.choice(["Starter", "Growth", "Enterprise"], size=n_accounts, p=[0.45, 0.40, 0.15])
    industry = RNG.choice(
        ["SaaS", "FinTech", "E-commerce", "Healthcare", "Other"],
        size=n_accounts,
        p=[0.30, 0.20, 0.20, 0.15, 0.15],
    )
    seats_purchased = np.where(
        plan == "Starter",
        RNG.integers(1, 8, n_accounts),
        np.where(plan == "Growth", RNG.integers(5, 30, n_accounts), RNG.integers(20, 150, n_accounts)),
    ).astype(int)

    days_since_onboarding = RNG.integers(30, 730, n_accounts)

    # Core usage signals (clean, strong correlations to retention)
    # Baseline login frequency (higher for larger plans)
    base_login = np.where(plan == "Starter", 8, np.where(plan == "Growth", 14, 22))
    login_freq_30d = np.clip(
        RNG.normal(base_login, 3.5, n_accounts), 0, 30
    ).round().astype(int)

    seat_utilization = np.clip(
        RNG.normal(0.55 if plan[0] != "Enterprise" else 0.70, 0.18, n_accounts), 0.05, 1.0
    )
    # Adjust per plan more carefully
    seat_utilization = np.clip(
        np.where(plan == "Starter", RNG.normal(0.48, 0.15, n_accounts),
                 np.where(plan == "Growth", RNG.normal(0.62, 0.14, n_accounts),
                          RNG.normal(0.78, 0.12, n_accounts))),
        0.05, 1.0,
    )

    feature_adoption_pct = np.clip(
        np.where(plan == "Starter", RNG.normal(0.35, 0.12, n_accounts),
                 np.where(plan == "Growth", RNG.normal(0.55, 0.13, n_accounts),
                          RNG.normal(0.72, 0.11, n_accounts))),
        0.05, 1.0,
    )

    api_calls_daily = np.clip(
        np.where(plan == "Starter", RNG.normal(40, 25, n_accounts),
                 np.where(plan == "Growth", RNG.normal(180, 80, n_accounts),
                          RNG.normal(650, 250, n_accounts))),
        0, None,
    ).astype(int)

    # Support (low volume, mostly clean)
    support_tickets_30d = RNG.poisson(1.2, n_accounts)
    escalated_tickets_30d = np.where(support_tickets_30d > 0, RNG.binomial(support_tickets_30d, 0.15), 0)

    nps_score = np.clip(RNG.normal(42, 22, n_accounts), -100, 100).round().astype(int)
    csat_score = np.clip(RNG.normal(3.9, 0.7, n_accounts), 1.0, 5.0).round(1)

    # Commercial
    days_to_renewal = RNG.integers(5, 365, n_accounts)
    monthly_revenue = np.where(
        plan == "Starter",
        seats_purchased * RNG.uniform(12, 25, n_accounts),
        np.where(plan == "Growth", seats_purchased * RNG.uniform(28, 55, n_accounts),
                 seats_purchased * RNG.uniform(60, 120, n_accounts)),
    ).round(2)

    # Target: churn within 90 days (clean, strong signal for Layer 1)
    # Higher risk when low usage, low adoption, many tickets, low NPS, near renewal
    risk_score = (
        -0.55 * (login_freq_30d / 30)
        -0.45 * seat_utilization
        -0.40 * feature_adoption_pct
        +0.25 * (support_tickets_30d / 4)
        +0.20 * (escalated_tickets_30d / 1.5)
        -0.22 * ((nps_score + 100) / 200)
        -0.15 * (csat_score / 5)
        +0.18 * (1 - np.clip(days_to_renewal / 120, 0, 1))
        + RNG.normal(0, 0.06, n_accounts)  # very small noise for clean baseline
    )
    # Convert to probability and sample (stronger separation)
    churn_prob = 1 / (1 + np.exp(-5.0 * (risk_score + 0.05)))
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
    return df


if __name__ == "__main__":
    df = generate_clean_saas_dataset(3000)
    out_path = OUT_DIR / "churn_gauge_v1_clean_3000.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows → {out_path}")
    print(f"Churn rate: {df['churned_90d'].mean():.1%}")
    print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]].head(20))
