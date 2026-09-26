# Churn Gauge

**Customer Health Predictor for SaaS**

Upload usage data → predict churn risk → recommend concrete actions.

## Current Status (Layer 1 complete)
- Project scaffolded under `Churn Gauge/`
- Clean synthetic dataset generated (`data/synthetic/churn_gauge_v1_clean_3000.csv`)
- Logistic Regression baseline trained (ROC-AUC ≈ 0.78)
- Action Catalogue defined (`docs/ACTION_CATALOGUE.md`)
- Milestone roadmap written (`docs/MILESTONES.md`)

## Planned Sequence
1. Logistic Regression (transparent coefficients) on clean data
2. XGBoost + SHAP on progressively messier / larger data
3. Combine interpretations for holistic view
4. Adaptive multi-framework UI (Streamlit / Gradio / Dash) that asks user expertise and recommends host
5. Open user CSV uploads only after quality gate

## Quick Start (after later layers)
```bash
# (future)
streamlit run apps/adaptive_launcher.py
```

## Structure
```
Churn Gauge/
├── data/synthetic/     # generated datasets (v1 clean → later messier)
├── src/data/           # generators
├── src/models/         # training scripts
├── src/actions/        # recommendation logic (next)
├── src/ui/             # adaptive host decision (later)
├── models/             # saved artifacts
├── docs/               # milestones, action catalogue
└── ...
```
