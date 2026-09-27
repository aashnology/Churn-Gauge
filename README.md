# Churn Gauge

**Customer Health Predictor for SaaS**

Upload usage data → predict churn risk → recommend concrete actions.

## Quick start

```bash
pip install -r requirements.txt

# Run a full layer (generate → train → register → quality gate)
python -m src.pipeline.run_layer --layer 3 --skip-generate --register-production

# Quality gate only
python -m src.pipeline.run_layer --gate-only

# Adaptive demo UI
streamlit run apps/adaptive_launcher.py
```

## Architecture

1. **Data** – progressive synthetic generators (clean → medium → messy)
2. **Features** – `src/features/engineer.py` + `src/pipeline/feature_spec.yaml`
3. **Models** – LogReg → XGBoost + SHAP; artifacts in `models/` + `models/registry.json`
4. **Quality gate** – AUC + action-usefulness + engagement sanity before user uploads
5. **Actions** – interaction-aware recommender + action catalogue
6. **UI** – adaptive launcher (recommends Streamlit / Gradio / Dash by expertise)

## Layers

| Layer | Data | Models |
|-------|------|--------|
| 1 | Clean 3k | LogReg |
| 2 | Medium 9k | LogReg + XGBoost/SHAP |
| 3 | Large messy 20k + full FE | LogReg + XGBoost, gate |

## Project layout

```
Churn Gauge/
├── apps/adaptive_launcher.py
├── data/synthetic/  data/processed/
├── docs/  docs/runs/
├── evaluation/  (PDP, interactions, action-usefulness)
├── models/  (artifacts + registry.json)
├── src/data/  src/features/  src/models/  src/actions/  src/pipeline/
└── requirements.txt
```

## Status

- Layers 0–3 implemented
- Quality gate: **passed** (upload enabled)
- Production model: `xgb_v3_engineered` (see `models/registry.json`)
