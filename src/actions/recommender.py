"""
Churn Gauge – Action recommender
Uses risk band + individual drivers + interaction effects (from PDP & interaction analysis).
Combined red flags raise urgency beyond single-signal triggers.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


def risk_band(prob: float) -> str:
    if prob >= 0.70:
        return "Critical"
    if prob >= 0.45:
        return "High"
    if prob >= 0.25:
        return "Medium"
    return "Low"


ACTION_MAP = {
    "login_freq_30d": {
        "low": "CSM Outreach + Value Check-in",
        "rationale": "Login frequency is a leading engagement signal. A drop warrants personal outreach to uncover blockers.",
    },
    "seat_utilization": {
        "low": "CSM Outreach + Value Check-in",
        "rationale": "Low seat utilization indicates under-adoption of purchased capacity.",
    },
    "feature_adoption_pct": {
        "low": "Re-engage Core Feature Adoption",
        "rationale": "Narrow feature usage is strongly linked to higher churn. Surface high-value under-used features.",
    },
    "support_tickets_30d": {
        "high": "Support Health Review & Escalation Cleanup",
        "rationale": "Elevated ticket volume is a frustration signal that often precedes cancellation.",
    },
    "escalated_tickets_30d": {
        "high": "Support Health Review & Escalation Cleanup",
        "rationale": "Any escalations require accelerated resolution and proactive status communication.",
    },
    "nps_score": {
        "low": "CSM Outreach + Value Check-in",
        "rationale": "Low NPS is a lagging but important sentiment signal.",
    },
    "csat_score": {
        "low": "Support Health Review & Escalation Cleanup",
        "rationale": "Low CSAT points to recent negative service experiences.",
    },
    "days_to_renewal": {
        "low": "Billing & Commercial Health Check",
        "rationale": "Approaching renewal with other red signals needs commercial conversation.",
    },
    "plan_Enterprise": {
        "high_risk": "Executive Sponsor / Champion Check-in",
        "rationale": "Enterprise accounts at risk benefit from executive-level re-anchoring.",
    },
    # Interaction-driven triggers (from PDP + interaction analysis)
    "high_tickets_x_low_util": {
        "high": "Support Health Review & Escalation Cleanup",
        "rationale": "Interaction: escalations/tickets combined with low utilisation is a high-priority risk combination.",
    },
    "low_login_x_low_adoption": {
        "high": "Re-engage Core Feature Adoption",
        "rationale": "Interaction: both login and adoption are weak — double engagement failure.",
    },
    "low_adoption_x_near_renewal": {
        "high": "Billing & Commercial Health Check",
        "rationale": "Interaction: low adoption near renewal needs commercial + adoption outreach.",
    },
    "tickets_x_renewal_urgency": {
        "high": "Support Health Review & Escalation Cleanup",
        "rationale": "Interaction: support burden rising as renewal approaches.",
    },
}


def _detect_interaction_flags(row: pd.Series) -> List[str]:
    """Detect active interaction flags from row values (works with or without pre-computed columns)."""
    flags = []
    util = row.get("seat_utilization", 1.0)
    if pd.isna(util):
        util = 1.0
    tickets = row.get("support_tickets_30d", 0) or 0
    escalated = row.get("escalated_tickets_30d", 0) or 0
    login = row.get("login_freq_30d", 15)
    if pd.isna(login):
        login = 15
    adoption = row.get("feature_adoption_pct", 0.5)
    if pd.isna(adoption):
        adoption = 0.5
    days_renew = row.get("days_to_renewal", 365) or 365

    # Pre-computed columns if present
    if row.get("high_tickets_x_low_util", 0) == 1 or (tickets >= 3 and util < 0.4) or (escalated >= 1 and util < 0.4):
        flags.append("high_tickets_x_low_util")
    if row.get("low_login_x_low_adoption", 0) == 1 or (login < 8 and adoption < 0.35):
        flags.append("low_login_x_low_adoption")
    if row.get("low_adoption_x_near_renewal", 0) == 1 or (adoption < 0.35 and days_renew < 60):
        flags.append("low_adoption_x_near_renewal")
    if row.get("tickets_x_renewal_urgency", 0) > 0.5 or (tickets >= 2 and days_renew < 60):
        flags.append("tickets_x_renewal_urgency")
    return flags


def recommend_actions(
    account_row: pd.Series,
    churn_prob: float,
    top_drivers: Optional[List[str]] = None,
    max_actions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Ranked actions for one account.
    Uses individual drivers + interaction flags (both PDP-informed and engineered).
    Multiple simultaneous red flags raise priority.
    """
    band = risk_band(churn_prob)
    actions: List[Dict[str, Any]] = []
    seen = set()

    if band == "Low":
        if (account_row.get("feature_adoption_pct", 0) or 0) > 0.6 and (account_row.get("seat_utilization", 0) or 0) > 0.6:
            actions.append({
                "action": "Expansion / Upsell Opportunity",
                "priority": "Low",
                "rationale": "Healthy high-adoption account is a candidate for expansion.",
                "triggered_by": "healthy_profile",
            })
        return actions[:max_actions]

    # --- Interaction flags first (higher urgency when multiple fire) ---
    interaction_flags = _detect_interaction_flags(account_row)
    for flag in interaction_flags:
        entry = ACTION_MAP.get(flag)
        if not entry:
            continue
        action_name = entry.get("high") or entry.get("low") or entry.get("high_risk")
        if action_name and action_name not in seen:
            seen.add(action_name)
            actions.append({
                "action": action_name,
                "priority": "Critical" if band == "Critical" or len(interaction_flags) >= 2 else band,
                "rationale": entry["rationale"],
                "triggered_by": flag,
            })

    # --- Individual drivers ---
    if top_drivers is None:
        candidates = []
        if (account_row.get("login_freq_30d") or 15) < 8:
            candidates.append("login_freq_30d")
        if (account_row.get("seat_utilization") or 1) < 0.4:
            candidates.append("seat_utilization")
        if (account_row.get("feature_adoption_pct") or 1) < 0.35:
            candidates.append("feature_adoption_pct")
        if (account_row.get("support_tickets_30d") or 0) >= 3:
            candidates.append("support_tickets_30d")
        if (account_row.get("escalated_tickets_30d") or 0) >= 1:
            candidates.append("escalated_tickets_30d")
        if (account_row.get("nps_score") or 50) < 20:
            candidates.append("nps_score")
        if (account_row.get("days_to_renewal") or 365) < 60:
            candidates.append("days_to_renewal")
        if account_row.get("plan") == "Enterprise" and churn_prob >= 0.45:
            candidates.append("plan_Enterprise")
        top_drivers = candidates

    for feat in top_drivers:
        if feat not in ACTION_MAP:
            continue
        entry = ACTION_MAP[feat]
        action_name = entry.get("low") or entry.get("high") or entry.get("high_risk")
        if action_name and action_name not in seen:
            seen.add(action_name)
            actions.append({
                "action": action_name,
                "priority": band,
                "rationale": entry["rationale"],
                "triggered_by": feat,
            })
        if len(actions) >= max_actions:
            break

    # Critical band always surfaces win-back
    if band == "Critical" and "Win-back / Last-chance Play (Critical only)" not in seen:
        actions.append({
            "action": "Win-back / Last-chance Play (Critical only)",
            "priority": "Critical",
            "rationale": "Highest-risk accounts require executive intervention and possible success-plan reset.",
            "triggered_by": "risk_band",
        })

    # If multiple interaction flags fired, boost first action priority
    if len(interaction_flags) >= 2 and actions:
        actions[0]["priority"] = "Critical"
        actions[0]["rationale"] = (
            actions[0]["rationale"]
            + f" [Multiple interaction flags active: {', '.join(interaction_flags)}]"
        )

    return actions[:max_actions]


if __name__ == "__main__":
    # Smoke tests
    high_risk = pd.Series({
        "login_freq_30d": 3,
        "seat_utilization": 0.22,
        "feature_adoption_pct": 0.18,
        "support_tickets_30d": 5,
        "escalated_tickets_30d": 2,
        "nps_score": 5,
        "days_to_renewal": 35,
        "plan": "Growth",
    })
    print("High-risk multi-flag account:")
    for r in recommend_actions(high_risk, 0.78):
        print(r)

    healthy = pd.Series({
        "login_freq_30d": 22,
        "seat_utilization": 0.85,
        "feature_adoption_pct": 0.78,
        "support_tickets_30d": 0,
        "escalated_tickets_30d": 0,
        "days_to_renewal": 200,
        "plan": "Enterprise",
    })
    print("\nHealthy account:")
    for r in recommend_actions(healthy, 0.12):
        print(r)
