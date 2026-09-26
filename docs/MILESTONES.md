# Churn Gauge – Milestone Checkpoints

Each layer builds on the previous. We start clean & simple, then progressively introduce messiness, outliers, and richer interpretations.

## Layer 0 – Project Scaffolding ✓
- Project named **Churn Gauge**
- Directory structure created
- Core packages verified
- Action Catalogue defined (8 practical interventions)

## Layer 1 – Clean Synthetic Baseline + Logistic Regression ✓
- 3 000-account clean dataset
- Logistic Regression → ROC-AUC **0.777**
- Coefficients business-aligned

## Layer 2 – Medium Complexity + Controlled Noise + Interpretations ✓
- 9 000-account medium dataset (~6 % missing, mild outliers)
- Logistic Regression → ROC-AUC **0.754**
- XGBoost → ROC-AUC **0.709** + full SHAP global ranking
- Side-by-side comparison produced
- Action recommender live and mapped to consensus drivers
- Interpretation summary written
- Adaptive launcher foundation (Streamlit shell + host recommendation logic)

## Layer 3 – Higher Realism + Kaggle Alignment (starting now)
- Larger / messier synthetic set
- Align structure with public Kaggle SaaS health / churn synthetics
- Feature expansion
- Holistic LogReg + XGBoost + SHAP synthesis
- Action-usefulness evaluation

## Layer 4 – Adaptive UI & Multi-Framework Ready
- Full Gradio + Dash entry points
- Auto-launch based on expertise answer
- User CSV upload path after quality gate

## Layer 5 – Production Hardening
- Tests, versioning, documentation polish
- Ready for repository push
