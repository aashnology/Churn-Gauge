# Churn Gauge – Recommended Action Catalogue (v1)

These actions are triggered by the model’s churn-risk probability and supporting feature signals.
They are designed to be practical for Customer Success / Account Management teams.

## Risk Bands
| Band              | Churn Probability | Health Label     | Urgency      |
|-------------------|-------------------|------------------|--------------|
| Critical          | ≥ 0.70            | Critical Risk    | Immediate    |
| High              | 0.45 – 0.69       | At Risk          | High         |
| Medium            | 0.25 – 0.44       | Needs Monitoring | Medium       |
| Low / Healthy     | < 0.25            | Healthy          | Low          |

## Action Catalogue

### 1. Re-engage Core Feature Adoption
- **Trigger signals**: Low `feature_adoption_pct` (< 0.40) + moderate-to-high risk
- **Action**: Automated in-app guided tour or email sequence highlighting 1–2 under-used high-value features that correlate with retention.
- **Owner**: Product + CS Automation
- **Expected impact**: Lift adoption → reduce churn probability

### 2. CSM Outreach + Value Check-in
- **Trigger signals**: Drop in `login_freq_30d` or `seat_utilization` + risk ≥ Medium
- **Action**: Personal outreach from assigned CSM within 48 h. Share usage insights and ask about goals / blockers.
- **Owner**: Customer Success Manager
- **Expected impact**: Surface issues early, rebuild engagement

### 3. Executive Sponsor / Champion Check-in
- **Trigger signals**: High risk + Enterprise plan OR declining multi-user activity
- **Action**: Schedule short call with economic buyer / executive sponsor. Focus on strategic value and roadmap alignment.
- **Owner**: Account Executive / CS Leadership
- **Expected impact**: Re-anchor relationship at decision-maker level

### 4. Support Health Review & Escalation Cleanup
- **Trigger signals**: Elevated `support_tickets_30d` or any `escalated_tickets_30d` + risk ≥ Medium
- **Action**: Audit open tickets, accelerate resolution, and send proactive status update to customer.
- **Owner**: Support + CS
- **Expected impact**: Reduce frustration signal that often precedes churn

### 5. Billing & Commercial Health Check
- **Trigger signals**: Low `days_to_renewal` (< 60) combined with any red usage signal, or past payment issues (future feature)
- **Action**: Proactive commercial conversation – clarify value, discuss plan fit, or offer flexible terms if appropriate.
- **Owner**: Account Manager / Finance Ops
- **Expected impact**: Prevent silent non-renewal

### 6. Expansion / Upsell Opportunity (Healthy accounts only)
- **Trigger signals**: Risk < 0.25 + high utilization + high feature adoption
- **Action**: Light-touch expansion play (additional seats, higher tier, or add-on feature).
- **Owner**: Account Executive
- **Expected impact**: Grow NRR while protecting healthy base

### 7. Onboarding / Re-onboarding Boost
- **Trigger signals**: Low `days_since_onboarding` (< 90) + low adoption / login
- **Action**: Trigger accelerated onboarding sequence or live training session.
- **Owner**: Onboarding Specialist / CS
- **Expected impact**: Improve early value realization

### 8. Win-back / Last-chance Play (Critical only)
- **Trigger signals**: Risk ≥ 0.70 and/or explicit cancellation signals
- **Action**: Executive-level intervention + possible concession package or success plan reset.
- **Owner**: CS Leadership + Exec Sponsor
- **Expected impact**: Last opportunity to reverse decision

---

**Usage in pipeline**:  
After prediction, the system ranks the top 1–3 most relevant actions based on the dominant risk drivers (via coefficient magnitudes for LogReg or SHAP values for XGBoost) and presents them with a short rationale.
