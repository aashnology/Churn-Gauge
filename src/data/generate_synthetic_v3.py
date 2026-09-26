"""
Churn Gauge – Phase 3 Synthetic Dataset
Larger (~20k), higher messiness: more outliers, higher missing rate, some corrupted categoricals,
mild concept drift between early/late accounts.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(123)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_v3(n_accounts: int = 20000) -> pd.DataFrame:
    account_id = [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)]
    plan = RNG.choice(["Starter", "Growth", "Enterprise"], size=n_accounts, p=[0.40, 0.38, 0.22])
    industry = RNG.choice(
        ["SaaS", "FinTech", "E-commerce", "Healthcare", "Other"],
        size=n_accounts, p=[0.27, 0.23, 0.18, 0.16, 0.16],
    )
    seats = np.where(
        plan == "Starter", RNG.integers(1, 12, n_accounts),
        np.where(plan == "Growth", RNG.integers(4, 50, n_accounts), RNG.integers(12, 250, n_accounts)),
    ).astype(int)
    days_onboard = RNG.integers(7, 1100, n_accounts)

    base_login = np.where(plan == "Starter", 6.5, np.where(plan == "Growth", 12, 19))
    login = np.clip(RNG.normal(base_login, 5.0, n_accounts), 0, 30).round().astype(int)
    util = np.clip(np.where(plan == "Starter", RNG.normal(0.42, 0.20, n_accounts),
                            np.where(plan == "Growth", RNG.normal(0.55, 0.18, n_accounts),
                                     RNG.normal(0.72, 0.15, n_accounts))), 0.01, 1.0)
    feat = np.clip(np.where(plan == "Starter", RNG.normal(0.30, 0.16, n_accounts),
                            np.where(plan == "Growth", RNG.normal(0.50, 0.17, n_accounts),
                                     RNG.normal(0.68, 0.14, n_accounts))), 0.01, 1.0)
    api = np.clip(np.where(plan == "Starter", RNG.normal(30, 35, n_accounts),
                           np.where(plan == "Growth", RNG.normal(150, 100, n_accounts),
                                    RNG.normal(580, 300, n_accounts))), 0, None).astype(int)
    tickets = RNG.poisson(1.8, n_accounts)
    escalated = np.where(tickets > 0, RNG.binomial(tickets, 0.20), 0)
    nps = np.clip(RNG.normal(35, 28, n_accounts), -100, 100).round().astype(int)
    csat = np.clip(RNG.normal(3.6, 0.95, n_accounts), 1.0, 5.0).round(1)
    days_renew = RNG.integers(1, 420, n_accounts)
    revenue = np.where(
        plan == "Starter", seats * RNG.uniform(9, 30, n_accounts),
        np.where(plan == "Growth", seats * RNG.uniform(22, 65, n_accounts),
                 seats * RNG.uniform(50, 150, n_accounts)),
    ).round(2)

    # Risk with more noise
    risk = (
        -0.48 * (login / 30) - 0.40 * util - 0.36 * feat
        + 0.30 * (tickets / 4) + 0.25 * (escalated / 1.5)
        - 0.18 * ((nps + 100) / 200) - 0.12 * (csat / 5)
        + 0.15 * (1 - np.clip(days_renew / 90, 0, 1))
        + RNG.normal(0, 0.14, n_accounts)
    )
    # Mild concept drift: later accounts have slightly higher base risk
    risk = risk + 0.08 * (np.arange(n_accounts) / n_accounts)
    prob = 1 / (1 + np.exp(-3.8 * (risk + 0.03)))
    churn = (RNG.random(n_accounts) < prob).astype(int)

    df = pd.DataFrame({
        "account_id": account_id, "plan": plan, "industry": industry,
        "seats_purchased": seats, "days_since_onboarding": days_onboard,
        "login_freq_30d": login, "seat_utilization": util.round(3),
        "feature_adoption_pct": feat.round(3), "api_calls_daily": api,
        "support_tickets_30d": tickets, "escalated_tickets_30d": escalated,
        "nps_score": nps, "csat_score": csat, "days_to_renewal": days_renew,
        "monthly_revenue": revenue, "churned_90d": churn,
    })

    # Higher messiness
    # Outliers
    for col in ["api_calls_daily", "monthly_revenue", "seats_purchased"]:
        q99 = df[col].quantile(0.99)
        idx = RNG.choice(df.index, size=int(0.025 * n_accounts), replace=False)
        df.loc[idx, col] = (q99 * RNG.uniform(2.0, 4.5, len(idx))).astype(df[col].dtype)

    # Missing ~9 %
    for col in ["nps_score", "csat_score", "feature_adoption_pct", "seat_utilization",
                "api_calls_daily", "login_freq_30d"]:
        idx = RNG.choice(df.index, size=int(0.09 * n_accounts), replace=False)
        df.loc[idx, col] = np.nan

    # Corrupted categoricals (rare)
    bad = RNG.choice(df.index, size=int(0.008 * n_accounts), replace=False)
    df.loc[bad, "plan"] = RNG.choice(["starter", "GROWTH", "enterprise ", "Unknown"], size=len(bad))

    return df


if __name__ == "__main__":
    df = generate_v3(20000)
    path = OUT_DIR / "churn_gauge_v3_large_messy_20000.csv"
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} → {path}")
    print(f"Churn rate: {df['churned_90d'].mean():.1%}")
    print(f"Missing total: {df.isna().sum().sum()}")
    print(df["plan"].value_counts(dropna=False).head(10))
