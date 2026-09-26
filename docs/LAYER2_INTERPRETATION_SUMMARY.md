# Churn Gauge – Layer 2 Interpretation Summary

## Datasets
| Version | Rows | Churn rate | Messiness |
|---------|------|------------|-----------|
| v1 clean | 3 000 | 3.8 % | Minimal noise, no missing |
| v2 medium | 9 000 | 8.8 % | ~6 % missing, mild outliers, slight inconsistencies |

## Model Performance (hold-out)
| Model | Dataset | ROC-AUC | Notes |
|-------|---------|---------|-------|
| Logistic Regression | v1 clean | 0.777 | Strong, transparent baseline |
| Logistic Regression | v2 medium | 0.754 | Mild drop expected with noise/missing |
| HistGradientBoosting | v2 medium | 0.710 | Interim while XGBoost was blocked |
| XGBoost | v2 medium | 0.709 | Comparable; SHAP used for true importance |

## Key Drivers – Side-by-side

**SHAP (XGBoost) – most trustworthy ranking for non-linear model**
1. login_freq_30d
2. seat_utilization
3. feature_adoption_pct
4. support_tickets_30d
5. escalated_tickets_30d
6. api_calls_daily
7. monthly_revenue
8. days_to_renewal
9. nps_score
10. plan_Starter

**Logistic Regression coefficients (v2) – direction & relative strength**
- Strong protective: login_freq_30d, seat_utilization, feature_adoption_pct, plan_Enterprise
- Strong risk-increasing: support_tickets_30d, escalated_tickets_30d

**Consensus signals (appear high in both views)**
- Login frequency
- Seat utilization
- Feature adoption %
- Support ticket volume / escalations

These four become the primary triggers for the Action Catalogue in the recommender.

## Action Mapping (already live in src/actions/recommender.py)
- Low login / utilization → CSM Outreach + Value Check-in
- Low feature adoption → Re-engage Core Feature Adoption
- High tickets / escalations → Support Health Review
- Near renewal + other red flags → Billing & Commercial Health Check
- Enterprise + high risk → Executive Sponsor Check-in
- Critical band → Win-back / Last-chance Play

## Next inside pipeline
- Generate v3 (larger + higher messiness)
- Begin Kaggle synthetic alignment
- Scaffold adaptive multi-framework launcher (asks expertise → recommends Streamlit / Gradio / Dash)
- Only after quality gate: enable real user CSV upload path
