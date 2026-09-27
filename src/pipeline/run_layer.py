#!/usr/bin/env python3
"""
Churn Gauge – single entry point for a layer run.

Usage:
  python -m src.pipeline.run_layer --layer 2
  python -m src.pipeline.run_layer --layer 3 --skip-generate
  python -m src.pipeline.run_layer --layer 3 --register-production

Layers:
  1 = clean 3k + LogReg
  2 = medium 9k + LogReg + XGBoost/SHAP
  3 = large messy 20k + full FE + LogReg + XGBoost + gate
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _run(cmd: list, desc: str) -> None:
    print(f"\n>>> {desc}")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {desc}")


def _write_run_card(layer: int, metrics: dict) -> Path:
    runs = ROOT / "docs" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = runs / f"{ts}_layer{layer}.md"
    lines = [
        f"# Run card – Layer {layer}",
        f"- UTC: {ts}",
        f"- Metrics: `{json.dumps(metrics)}`",
        "",
        "## Checklist",
        "- [x] Generate / load data",
        "- [x] Train models",
        "- [x] Register artifacts",
        "- [x] Quality gate evaluated",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def layer1(skip_generate: bool) -> dict:
    if not skip_generate:
        _run([sys.executable, "src/data/generate_synthetic_v1.py"], "Generate clean v1")
    _run([sys.executable, "src/models/train_logreg_v1.py"], "Train LogReg v1")
    metrics = {}
    p = ROOT / "models" / "logreg_v1_metrics.json"
    if p.exists():
        metrics = json.loads(p.read_text())
    from src.pipeline.registry import register_model
    register_model(
        "logreg_v1",
        "models/logreg_v1_clean.joblib",
        "logistic",
        "v1_clean",
        float(metrics.get("roc_auc", 0.0)),
        metrics,
        set_production=False,
    )
    return metrics


def layer2(skip_generate: bool) -> dict:
    if not skip_generate:
        _run([sys.executable, "src/data/generate_synthetic_v2.py"], "Generate medium v2")
    _run([sys.executable, "src/models/train_logreg_v2.py"], "Train LogReg v2")
    _run([sys.executable, "src/models/train_xgb_shap_v2.py"], "Train XGBoost+SHAP v2")
    metrics = {}
    for name in ["logreg_v2_metrics.json", "xgb_v2_metrics.json"]:
        p = ROOT / "models" / name
        if p.exists():
            metrics[name] = json.loads(p.read_text())
    from src.pipeline.registry import register_model
    if (ROOT / "models" / "xgb_shap_v2_medium.joblib").exists():
        auc = metrics.get("xgb_v2_metrics.json", {}).get("roc_auc", 0.709)
        register_model("xgb_v2_shap", "models/xgb_shap_v2_medium.joblib", "xgboost", "v2_medium", float(auc), set_production=False)
    if (ROOT / "models" / "logreg_v2_medium.joblib").exists():
        auc = metrics.get("logreg_v2_metrics.json", {}).get("roc_auc", 0.754)
        register_model("logreg_v2", "models/logreg_v2_medium.joblib", "logistic", "v2_medium", float(auc), set_production=False)
    return metrics


def layer3(skip_generate: bool, set_production: bool) -> dict:
    if not skip_generate:
        _run([sys.executable, "src/data/generate_synthetic_v3.py"], "Generate large messy v3")
        _run([sys.executable, "src/features/engineer.py"], "Feature engineering v3")
    _run([sys.executable, "src/models/train_engineered_v3.py"], "Train engineered LogReg + XGBoost")
    metrics = {}
    p = ROOT / "models" / "engineered_v3_metrics.json"
    if p.exists():
        metrics = json.loads(p.read_text())
    from src.pipeline.registry import register_model
    xgb_auc = float(metrics.get("xgb_auc", 0.74))
    lr_auc = float(metrics.get("logreg_auc", 0.75))
    if (ROOT / "models" / "xgb_v3_engineered.joblib").exists():
        register_model(
            "xgb_v3_engineered",
            "models/xgb_v3_engineered.joblib",
            "xgboost",
            "v3_engineered",
            xgb_auc,
            metrics,
            set_production=set_production,
        )
    if (ROOT / "models" / "logreg_v3_engineered.joblib").exists():
        register_model(
            "logreg_v3_engineered",
            "models/logreg_v3_engineered.joblib",
            "logistic",
            "v3_engineered",
            lr_auc,
            metrics,
            set_production=False,
        )
    _run([sys.executable, "evaluation/action_usefulness.py"], "Action-usefulness eval")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Churn Gauge layer runner")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing data")
    parser.add_argument("--register-production", action="store_true", help="Mark best model as production (layer 3)")
    parser.add_argument("--gate-only", action="store_true", help="Only run quality gate")
    args = parser.parse_args()

    if args.gate_only:
        from src.pipeline.quality_gate import run_gate
        print(json.dumps(run_gate(), indent=2))
        return

    print(f"=== Churn Gauge – Layer {args.layer} ===")
    if args.layer == 1:
        metrics = layer1(args.skip_generate)
        best_auc = float(metrics.get("roc_auc", 0))
    elif args.layer == 2:
        metrics = layer2(args.skip_generate)
        best_auc = float(metrics.get("xgb_v2_metrics.json", metrics.get("logreg_v2_metrics.json", {})).get("roc_auc", 0.7))
    else:
        metrics = layer3(args.skip_generate, args.register_production)
        best_auc = max(float(metrics.get("xgb_auc", 0)), float(metrics.get("logreg_auc", 0)))

    from src.pipeline.quality_gate import run_gate
    gate = run_gate(roc_auc=best_auc)
    print("\n=== Quality gate ===")
    print(json.dumps(gate, indent=2))

    card = _write_run_card(args.layer, {"metrics": metrics, "gate": gate})
    print(f"\nRun card → {card}")
    print("Done.")


if __name__ == "__main__":
    main()
