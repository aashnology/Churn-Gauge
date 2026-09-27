"""
Quality gate – must pass before enabling user CSV upload in the adaptive launcher.
Checks ROC-AUC, action-usefulness, and basic PDP/coefficient sanity.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).parent / "feature_spec.yaml"
GATE_FLAG = ROOT / "models" / "quality_gate_passed.json"


def _load_spec() -> Dict[str, Any]:
    try:
        import yaml
        with open(SPEC_PATH) as f:
            return yaml.safe_load(f)
    except Exception:
        return {
            "quality_gate": {
                "min_roc_auc": 0.72,
                "min_action_usefulness": 0.85,
                "require_protective_engagement": True,
            }
        }


def check_roc_auc(roc_auc: float, min_auc: float) -> Dict[str, Any]:
    ok = roc_auc >= min_auc
    return {"check": "roc_auc", "value": roc_auc, "threshold": min_auc, "passed": ok}


def check_action_usefulness(score: Optional[float], min_score: float) -> Dict[str, Any]:
    if score is None:
        p = ROOT / "evaluation" / "action_usefulness_v2.json"
        if p.exists():
            score = json.loads(p.read_text()).get("score")
    ok = score is not None and score >= min_score
    return {"check": "action_usefulness", "value": score, "threshold": min_score, "passed": ok}


def check_engagement_protective(coef_or_shap_path: Optional[Path] = None) -> Dict[str, Any]:
    protective_features = ["login_freq_30d", "seat_utilization", "feature_adoption_pct", "simple_health", "engagement_score"]
    found_protective = 0
    details = []
    for name in [
        "logreg_v3_engineered_coefs.csv",
        "logreg_v2_coefficients.csv",
        "logreg_v1_coefficients.csv",
    ]:
        p = ROOT / "models" / name
        if not p.exists():
            continue
        import pandas as pd
        df = pd.read_csv(p)
        col = "coefficient" if "coefficient" in df.columns else None
        if col is None:
            continue
        for feat in protective_features:
            row = df[df["feature"] == feat]
            if len(row) and float(row.iloc[0][col]) < 0:
                found_protective += 1
                details.append(f"{feat}<0 in {name}")
        break
    for name in ["xgb_v3_engineered_importance.csv", "xgb_v2_feature_importance.csv"]:
        p = ROOT / "models" / name
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p)
            if "simple_health" in set(df["feature"].head(5)):
                found_protective += 1
                details.append("simple_health in top-5 XGB importance")
            break
    ok = found_protective >= 2
    return {
        "check": "engagement_protective",
        "value": found_protective,
        "threshold": 2,
        "passed": ok,
        "details": details,
    }


def run_gate(
    roc_auc: Optional[float] = None,
    action_usefulness: Optional[float] = None,
) -> Dict[str, Any]:
    spec = _load_spec().get("quality_gate", {})
    min_auc = float(spec.get("min_roc_auc", 0.72))
    min_au = float(spec.get("min_action_usefulness", 0.85))
    if roc_auc is None:
        try:
            from src.pipeline.registry import get_production
            prod = get_production()
            if prod:
                roc_auc = float(prod["roc_auc"])
        except Exception:
            pass
    checks: List[Dict[str, Any]] = []
    if roc_auc is not None:
        checks.append(check_roc_auc(roc_auc, min_auc))
    else:
        checks.append({"check": "roc_auc", "value": None, "threshold": min_auc, "passed": False, "note": "no AUC provided"})
    checks.append(check_action_usefulness(action_usefulness, min_au))
    if spec.get("require_protective_engagement", True):
        checks.append(check_engagement_protective())
    passed = all(c["passed"] for c in checks)
    result = {
        "passed": passed,
        "checks": checks,
        "upload_enabled": passed,
    }
    GATE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_FLAG, "w") as f:
        json.dump(result, f, indent=2)
    return result


def is_upload_enabled() -> bool:
    if not GATE_FLAG.exists():
        return False
    return bool(json.loads(GATE_FLAG.read_text()).get("upload_enabled"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    r = run_gate()
    print(json.dumps(r, indent=2))
    print("Upload enabled:" if r["passed"] else "Upload BLOCKED:", r["passed"])
