"""
retrain_trigger.py
==================
Evaluates all four trigger types, logs the decision as an MLflow run,
and returns a structured result that pipeline.py acts on.

MLflow integration:
  - Creates a trigger-evaluation run under 'iris_ct_monitoring' experiment
  - Logs PSI, confidence, and SLO metrics that drove the decision
  - Tags run with final retrain=true/false decision and reasons
  - Links to the current Production model run_id

Usage:
    python src/retrain_trigger.py
    python src/retrain_trigger.py --type drift
    python src/retrain_trigger.py --dry-run
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import mlflow
import yaml
from mlflow.tracking import MlflowClient
from scipy import stats

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FEATURE_COLS = CFG["features"]["names"]
TARGET_COL = CFG["features"]["target"]
PATHS = CFG["paths"]
TRIGGERS = CFG["triggers"]
DRIFT_CFG = CFG["drift"]
SLO = CFG["slo"]
MLFLOW_CFG = CFG["mlflow"]
MLFLOW_CFG["tracking_uri"] = os.getenv(
    "MLFLOW_TRACKING_URI", MLFLOW_CFG["tracking_uri"]
)


def _safe(obj):
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _load_bundle():
    if not os.path.exists(PATHS["current_model"]):
        return None
    with open(PATHS["current_model"], "rb") as f:
        return pickle.load(f)


def _psi(ref, prod, n=10):
    rc, pc = ref.dropna(), prod.dropna()
    bp = np.unique(np.percentile(rc, np.linspace(0, 100, n + 1)))
    if len(bp) < 3:
        return 0.0
    rp = np.histogram(rc, bins=bp)[0] / len(rc) + 1e-8
    pp = np.histogram(pc, bins=bp)[0] / len(pc) + 1e-8
    return float(round(np.sum((pp - rp) * np.log(pp / rp)), 4))


# ── Current production model run_id from Registry ────────────────────────────
def _prod_run_id() -> str:
    try:
        mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
        client = MlflowClient()
        pvs = client.get_latest_versions(
            MLFLOW_CFG["registered_model"], stages=["Production"]
        )
        return pvs[0].run_id if pvs else "N/A"
    except Exception:
        return "N/A"


# ── Last retrain timestamp from retrain log ───────────────────────────────────
def _last_retrain():
    path = PATHS.get("retrain_log", "logs/retrain_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        hist = json.load(f)
    promo = [e for e in hist if e.get("event") == "model_promoted"]
    if not promo:
        return None
    return datetime.fromisoformat(promo[-1]["timestamp"]).replace(tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════════════════════════════
# Trigger 1 — Time-based
# ══════════════════════════════════════════════════════════════════════════════
def check_time(mlflow_metrics: dict) -> Dict:
    if not TRIGGERS["time_based"]["enabled"]:
        return {"enabled": False, "fire": False, "reason": "disabled"}
    interval = TRIGGERS["time_based"]["interval_days"]
    last_ts = _last_retrain()
    if last_ts is None:
        mlflow_metrics["time_days_since_retrain"] = 9999.0
        return {
            "enabled": True,
            "fire": True,
            "reason": "no_training_history",
            "interval_days": interval,
        }
    days = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400
    fire = days >= interval
    mlflow_metrics["time_days_since_retrain"] = round(days, 2)
    mlflow_metrics["time_trigger"] = float(fire)
    return {
        "enabled": True,
        "fire": bool(fire),
        "reason": f"{days:.1f}_days_since_last_retrain" if fire else "within_schedule",
        "days_since": round(days, 2),
        "interval_days": interval,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Trigger 2 — Performance-based
# ══════════════════════════════════════════════════════════════════════════════
def check_performance(prod_df: pd.DataFrame, mlflow_metrics: dict) -> Dict:
    if not TRIGGERS["performance_based"]["enabled"]:
        return {"enabled": False, "fire": False, "reason": "disabled"}
    if TARGET_COL not in prod_df.columns:
        return {"enabled": True, "fire": False, "reason": "no_labels"}
    lab = prod_df.dropna(subset=[TARGET_COL])
    min_n = TRIGGERS["performance_based"]["min_samples_for_evaluation"]
    if len(lab) < min_n:
        return {
            "enabled": True,
            "fire": False,
            "reason": f"insufficient_samples_{len(lab)}_of_{min_n}",
        }
    bundle = _load_bundle()
    if bundle is None:
        return {"enabled": True, "fire": False, "reason": "no_current_model"}
    from sklearn.metrics import accuracy_score, f1_score

    pred = bundle["model"].predict(lab[FEATURE_COLS].values)
    acc = float(accuracy_score(lab[TARGET_COL].values, pred))
    f1 = float(f1_score(lab[TARGET_COL].values, pred, average="macro"))
    fire = acc < SLO["accuracy"]["critical"] or f1 < SLO["f1_macro"]["critical"]
    mlflow_metrics["perf_accuracy"] = acc
    mlflow_metrics["perf_f1_macro"] = f1
    mlflow_metrics["perf_trigger"] = float(fire)
    reasons = []
    if acc < SLO["accuracy"]["critical"]:
        reasons.append(f"acc={acc:.4f}<{SLO['accuracy']['critical']}")
    if f1 < SLO["f1_macro"]["critical"]:
        reasons.append(f"f1={f1:.4f}<{SLO['f1_macro']['critical']}")
    return {
        "enabled": True,
        "fire": bool(fire),
        "reason": ",".join(reasons) or "within_slo",
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Trigger 3 — Drift-based
# ══════════════════════════════════════════════════════════════════════════════
def check_drift(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame, mlflow_metrics: dict
) -> Dict:
    if not TRIGGERS["drift_based"]["enabled"]:
        return {"enabled": False, "fire": False, "reason": "disabled"}
    min_n = TRIGGERS["drift_based"]["min_samples_for_drift"]
    if len(prod_df) < min_n:
        return {
            "enabled": True,
            "fire": False,
            "reason": f"insufficient_samples_{len(prod_df)}_of_{min_n}",
        }
    features = TRIGGERS["drift_based"]["features_to_monitor"]
    drifted = []
    per_feat = {}
    for feat in features:
        psi = _psi(ref_df[feat], prod_df[feat])
        ks_s, ks_p = stats.ks_2samp(ref_df[feat].dropna(), prod_df[feat].dropna())
        fire = (
            psi > DRIFT_CFG["psi"]["trigger"]
            or ks_p < DRIFT_CFG["ks_pvalue"]["trigger"]
        )
        if fire:
            drifted.append(feat)
        per_feat[feat] = {
            "psi": psi,
            "ks_pvalue": round(float(ks_p), 6),
            "fire": bool(fire),
        }
        mlflow_metrics[f"trigger_psi_{feat}"] = psi
        mlflow_metrics[f"trigger_ksp_{feat}"] = float(ks_p)
        mlflow_metrics[f"trigger_drift_{feat}"] = float(fire)
    mlflow_metrics["drift_n_features_fired"] = float(len(drifted))
    mlflow_metrics["drift_trigger"] = float(len(drifted) > 0)
    return {
        "enabled": True,
        "fire": bool(len(drifted) > 0),
        "reason": f"drift_on:{','.join(drifted)}" if drifted else "no_drift",
        "drifted_features": drifted,
        "per_feature": per_feat,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Trigger 4 — Active learning (uncertainty-based)
# ══════════════════════════════════════════════════════════════════════════════
def check_active_learning(prod_df: pd.DataFrame, mlflow_metrics: dict) -> Dict:
    bundle = _load_bundle()
    if bundle is None:
        return {"enabled": True, "fire": False, "reason": "no_current_model"}
    probs = bundle["model"].predict_proba(prod_df[FEATURE_COLS].values)
    max_prob = probs.max(axis=1)
    conf_thr = 0.70
    fire_thr = 0.20
    unc_frac = float((max_prob < conf_thr).mean())
    fire = unc_frac > fire_thr
    mlflow_metrics["al_uncertain_fraction"] = unc_frac
    mlflow_metrics["al_mean_confidence"] = float(max_prob.mean())
    mlflow_metrics["al_trigger"] = float(fire)
    return {
        "enabled": True,
        "fire": bool(fire),
        "reason": f"{unc_frac:.1%}_samples_uncertain" if fire else "confidence_ok",
        "uncertain_fraction": round(unc_frac, 4),
        "confidence_threshold": conf_thr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def run_trigger_evaluation(trigger_type=None, dry_run=False) -> Dict:
    print("=" * 55)
    print("  Iris CT — Retrain Trigger Evaluation  (MLflow ON)")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    ref_df = pd.read_csv(PATHS["reference_data"])
    prod_df = (
        pd.read_csv(PATHS["production_data"])
        if os.path.exists(PATHS["production_data"])
        else ref_df.head(0)
    )

    all_mlflow_metrics = {}
    results = {}
    run_all = trigger_type is None

    if run_all or trigger_type == "time":
        r = check_time(all_mlflow_metrics)
        results["time"] = r
        _print_trigger("Time-based", r)

    if run_all or trigger_type == "performance":
        r = check_performance(prod_df, all_mlflow_metrics)
        results["performance"] = r
        _print_trigger("Performance", r)

    if run_all or trigger_type == "drift":
        r = check_drift(ref_df, prod_df, all_mlflow_metrics)
        results["drift"] = r
        _print_trigger("Drift-based", r)

    if (run_all or trigger_type == "active") and len(prod_df) > 0:
        r = check_active_learning(prod_df, all_mlflow_metrics)
        results["active_learning"] = r
        _print_trigger("Active Learning", r)

    fired = {n: r for n, r in results.items() if r.get("fire")}
    retrain = bool(len(fired) > 0)
    reasons = list(fired.keys())
    all_mlflow_metrics["overall_retrain"] = float(retrain)

    print("\n" + "=" * 55)
    print(f"  DECISION: {'RETRAIN 🚨' if retrain else 'NO ACTION ✅'}")
    if retrain:
        print(f"  Triggered by: {', '.join(reasons)}")
    if dry_run:
        print("  [DRY RUN — MLflow run still logged]")
    print("=" * 55)

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    prod_run = _prod_run_id()
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"] + "_monitoring")
    with mlflow.start_run(
        nested=True,
        run_name=f"trigger_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    ) as trun:
        mlflow.log_metrics(all_mlflow_metrics)
        mlflow.set_tags(
            {
                "run_type": "trigger_evaluation",
                "retrain_decision": str(retrain),
                "triggered_by": ",".join(reasons) or "none",
                "dry_run": str(dry_run),
                "production_run_id": prod_run,
            }
        )
        print(f"\nTrigger decision logged → MLflow run {trun.info.run_id[:8]}…")

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrain": retrain,
        "triggered_by": reasons,
        "trigger_results": results,
        "dry_run": dry_run,
    }

    if not dry_run:
        os.makedirs("logs", exist_ok=True)
        path = "logs/trigger_history.json"
        history = json.load(open(path)) if os.path.exists(path) else []
        history.append(_safe(decision))
        json.dump(history, open(path, "w"), indent=2)

    return decision


def _print_trigger(name, r):
    s = (
        "🚨 FIRE"
        if r.get("fire")
        else ("⏭  SKIP" if not r.get("enabled", True) else "✅ OK  ")
    )
    print(f"\n  [{s}]  {name}")
    print(f"         reason: {r.get('reason','n/a')}")
    if r.get("drifted_features"):
        print(f"         drifted: {', '.join(r['drifted_features'])}")
    if "days_since" in r and r["days_since"]:
        print(f"         days since retrain: {r['days_since']:.1f}")
    if "uncertain_fraction" in r:
        print(f"         uncertain: {r['uncertain_fraction']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", choices=["time", "performance", "drift", "active"], default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_trigger_evaluation(trigger_type=args.type, dry_run=args.dry_run)
