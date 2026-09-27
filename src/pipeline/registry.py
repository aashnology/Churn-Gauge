"""
Model registry – single source of truth for which model is production.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"version": 1, "current_production": None, "models": []}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(reg: Dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def register_model(
    model_id: str,
    path: str,
    model_type: str,
    dataset: str,
    roc_auc: float,
    metrics: Optional[Dict[str, Any]] = None,
    set_production: bool = False,
) -> Dict[str, Any]:
    reg = load_registry()
    entry = {
        "model_id": model_id,
        "path": path,
        "model_type": model_type,
        "dataset": dataset,
        "roc_auc": roc_auc,
        "metrics": metrics or {},
        "created_at": _now(),
    }
    reg["models"] = [m for m in reg["models"] if m["model_id"] != model_id]
    reg["models"].append(entry)
    if set_production or reg.get("current_production") is None:
        reg["current_production"] = model_id
    save_registry(reg)
    return entry


def get_production() -> Optional[Dict[str, Any]]:
    reg = load_registry()
    pid = reg.get("current_production")
    if not pid:
        return None
    for m in reg["models"]:
        if m["model_id"] == pid:
            return m
    return None


def list_models() -> List[Dict[str, Any]]:
    return load_registry().get("models", [])


if __name__ == "__main__":
    reg = load_registry()
    if not reg["models"]:
        candidates = [
            ("xgb_v3_engineered", "models/xgb_v3_engineered.joblib", "xgboost", "v3_engineered", 0.74),
            ("logreg_v3_engineered", "models/logreg_v3_engineered.joblib", "logistic", "v3_engineered", 0.757),
            ("xgb_v2_shap", "models/xgb_shap_v2_medium.joblib", "xgboost", "v2_medium", 0.709),
            ("logreg_v2", "models/logreg_v2_medium.joblib", "logistic", "v2_medium", 0.754),
        ]
        for i, (mid, path, mtype, ds, auc) in enumerate(candidates):
            p = ROOT / path
            if p.exists():
                register_model(mid, path, mtype, ds, auc, set_production=(i == 0))
        print("Seeded registry:")
    for m in list_models():
        print(f"  {m['model_id']}: AUC={m['roc_auc']}  prod={m['model_id']==load_registry().get('current_production')}")
    print("Production:", get_production())
