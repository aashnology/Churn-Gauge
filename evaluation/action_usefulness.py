"""
Simple action-usefulness rubric evaluator.
For each high-risk account, check whether the recommender surfaces at least one
relevant action given the dominant drivers. Score = % of high-risk accounts
that receive a non-empty, non-generic recommendation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.actions.recommender import recommend_actions, risk_band

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "churn_gauge_v2_medium_9000.csv"


def main():
    df = pd.read_csv(DATA)
    # Use actual churn as proxy for "should have been flagged"
    high_risk = df[df["churned_90d"] == 1].copy()
    # Also sample some high-prob looking accounts from features
    print(f"Evaluating on {len(high_risk)} actual churned accounts (proxy for high risk)")

    useful = 0
    total = 0
    examples = []
    for _, row in high_risk.sample(min(300, len(high_risk)), random_state=42).iterrows():
        # Synthetic probability proxy from features (rough)
        score = 0.0
        if row.get("login_freq_30d", 15) < 8:
            score += 0.25
        if row.get("seat_utilization", 0.5) < 0.4:
            score += 0.2
        if row.get("feature_adoption_pct", 0.5) < 0.35:
            score += 0.2
        if row.get("support_tickets_30d", 0) >= 3:
            score += 0.15
        if row.get("escalated_tickets_30d", 0) >= 1:
            score += 0.15
        prob = min(0.95, 0.3 + score)
        recs = recommend_actions(row, prob)
        total += 1
        if recs and any(r["action"] != "Expansion / Upsell Opportunity" for r in recs):
            useful += 1
            if len(examples) < 3:
                examples.append((row["account_id"], prob, [r["action"] for r in recs]))

    pct = useful / max(total, 1)
    print(f"Action-usefulness score: {pct:.1%} ({useful}/{total})")
    print("Example recommendations:")
    for e in examples:
        print(e)

    out = ROOT / "evaluation" / "action_usefulness_v2.json"
    import json
    with open(out, "w") as f:
        json.dump({"score": pct, "n": total, "useful": useful}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
