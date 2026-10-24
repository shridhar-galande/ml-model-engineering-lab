"""
monitor.py
==========
Monitors the deployed Iris model for drift and performance degradation.
Logs all monitoring results as a tagged MLflow run for full auditability.

MLflow integration:
  - Creates a monitoring run under the 'iris_ct_monitoring' experiment
  - Logs PSI, KS stats, JSD, class-shift as MLflow metrics
  - Tags run with drift status and trigger decision
  - Links to the current Production model run_id in tags

Usage:
    python src/monitor.py
    python src/monitor.py --production data/new_batch.csv
    python src/monitor.py --report-only   # EvidentlyAI HTML only, no MLflow
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import mlflow
import yaml
from mlflow.tracking import MlflowClient
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FEATURE_COLS = CFG["features"]["names"]
TARGET_COL = CFG["features"]["target"]
DRIFT_CFG = CFG["drift"]
PATHS = CFG["paths"]
SLO = CFG["slo"]
MLFLOW_CFG = CFG["mlflow"]
MLFLOW_CFG["tracking_uri"] = os.getenv(
    "MLFLOW_TRACKING_URI", MLFLOW_CFG["tracking_uri"]
)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


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


# ══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════════


def compute_psi(ref: pd.Series, prod: pd.Series, n_bins=10) -> Optional[float]:
    rc, pc = ref.dropna(), prod.dropna()
    bp = np.unique(np.percentile(rc, np.linspace(0, 100, n_bins + 1)))
    if len(bp) < 3:
        return None
    rp = np.histogram(rc, bins=bp)[0] / len(rc) + 1e-8
    pp = np.histogram(pc, bins=bp)[0] / len(pc) + 1e-8
    return round(float(np.sum((pp - rp) * np.log(pp / rp))), 4)


def compute_ks(ref: pd.Series, prod: pd.Series) -> Dict:
    s, p = stats.ks_2samp(ref.dropna(), prod.dropna())
    return {
        "statistic": round(float(s), 4),
        "p_value": round(float(p), 6),
        "drifted": bool(p < DRIFT_CFG["ks_pvalue"]["trigger"]),
    }


def compute_jsd(ref: pd.Series, prod: pd.Series, n_bins=20) -> float:
    vals = pd.concat([ref, prod]).dropna()
    bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
    rh = np.histogram(ref.dropna(), bins=bins)[0].astype(float) + 1e-8
    ph = np.histogram(prod.dropna(), bins=bins)[0].astype(float) + 1e-8
    return round(float(jensenshannon(rh / rh.sum(), ph / ph.sum()) ** 2), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Feature drift
# ══════════════════════════════════════════════════════════════════════════════


def run_feature_drift(ref_df, prod_df, features) -> Dict:
    results = {}
    n_drift = 0
    psi_max = 0.0
    mlflow_metrics = {}

    print("\nFeature Drift Detection:")
    print(f"  {'Feature':<18} {'PSI':>7} {'KS p-val':>10} {'JSD':>7}  Status")
    print("  " + "-" * 52)

    for feat in features:
        psi = compute_psi(ref_df[feat], prod_df[feat]) or 0.0
        ks = compute_ks(ref_df[feat], prod_df[feat])
        jsd = compute_jsd(ref_df[feat], prod_df[feat])

        fire = psi > DRIFT_CFG["psi"]["trigger"] or ks["drifted"]
        if fire:
            n_drift += 1
        psi_max = max(psi_max, psi)

        status = (
            "DRIFT 🚨"
            if fire
            else ("WATCH ⚠️" if psi > DRIFT_CFG["psi"]["watch"] else "OK ✅")
        )
        results[feat] = {
            "psi": psi,
            "ks": ks,
            "jsd": jsd,
            "status": status,
            "fire": fire,
        }
        print(f"  {feat:<18} {psi:>7.4f} {ks['p_value']:>10.6f} {jsd:>7.4f}  {status}")

        # Flat keys for MLflow
        mlflow_metrics[f"drift_psi_{feat}"] = psi
        mlflow_metrics[f"drift_ks_stat_{feat}"] = ks["statistic"]
        mlflow_metrics[f"drift_ks_pval_{feat}"] = ks["p_value"]
        mlflow_metrics[f"drift_jsd_{feat}"] = jsd

    trig = n_drift > 0 or psi_max > DRIFT_CFG["psi"]["trigger"]
    mlflow_metrics["drift_max_psi"] = psi_max
    mlflow_metrics["drift_n_features"] = float(n_drift)
    mlflow_metrics["drift_trigger"] = float(trig)

    print(
        f"\n  Drifted: {n_drift}/{len(features)}  |  Max PSI: {psi_max:.4f}  |  "
        f"{'🚨 TRIGGER' if trig else '✅ OK'}"
    )

    return {
        "per_feature": results,
        "n_drifted": n_drift,
        "max_psi": psi_max,
        "trigger_retraining": trig,
        "_mlflow_metrics": mlflow_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Prediction distribution monitoring
# ══════════════════════════════════════════════════════════════════════════════


def monitor_prediction_distribution(bundle, ref_df, prod_df) -> Dict:
    model = bundle["model"]
    rp = model.predict(ref_df[FEATURE_COLS].values)
    pp = model.predict(prod_df[FEATURE_COLS].values)
    rb = model.predict_proba(ref_df[FEATURE_COLS].values)
    pb = model.predict_proba(prod_df[FEATURE_COLS].values)

    ref_d = {c: float((rp == i).mean()) for i, c in enumerate(CLASS_NAMES)}
    prod_d = {c: float((pp == i).mean()) for i, c in enumerate(CLASS_NAMES)}
    max_s = max(abs(ref_d[c] - prod_d[c]) for c in CLASS_NAMES)
    trig = max_s > DRIFT_CFG["prediction_shift"]["trigger"]

    mlflow_metrics = {}
    for i, cls in enumerate(CLASS_NAMES):
        shift = abs(ref_d[cls] - prod_d[cls])
        jsd = compute_jsd(pd.Series(rb[:, i]), pd.Series(pb[:, i]))
        mlflow_metrics[f"pred_shift_{cls}"] = shift
        mlflow_metrics[f"pred_score_jsd_{cls}"] = jsd

    mlflow_metrics["pred_max_class_shift"] = max_s
    mlflow_metrics["pred_dist_trigger"] = float(trig)

    print("\nPrediction Distribution:")
    print(f"  {'Class':<14} {'Ref':>6} {'Prod':>6} {'Shift':>7}")
    print("  " + "-" * 38)
    for cls in CLASS_NAMES:
        shift = abs(ref_d[cls] - prod_d[cls])
        flag = " 🚨" if shift > DRIFT_CFG["prediction_shift"]["trigger"] else ""
        print(f"  {cls:<14} {ref_d[cls]:>6.1%} {prod_d[cls]:>6.1%} {shift:>7.1%}{flag}")
    print(f"\n  Max shift: {max_s:.1%}  |  {'🚨 TRIGGER' if trig else '✅ OK'}")

    return {
        "reference_distribution": ref_d,
        "production_distribution": prod_d,
        "max_class_shift": round(max_s, 4),
        "trigger_retraining": trig,
        "_mlflow_metrics": mlflow_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Performance monitoring (with labels)
# ══════════════════════════════════════════════════════════════════════════════


def monitor_performance(bundle, prod_df) -> Optional[Dict]:
    from sklearn.metrics import accuracy_score, f1_score

    if TARGET_COL not in prod_df.columns:
        print("\nPerformance: labels not available.")
        return None
    X, y = prod_df[FEATURE_COLS].values, prod_df[TARGET_COL].values
    y_pred = bundle["model"].predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="macro"))
    trig = acc < SLO["accuracy"]["critical"] or f1 < SLO["f1_macro"]["critical"]
    print(
        f"\nPerformance:  acc={acc:.4f}  f1={f1:.4f}  "
        f"{'🚨 SLO BREACH' if trig else '✅ Within SLO'}"
    )
    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "trigger_retraining": trig,
        "_mlflow_metrics": {
            "prod_accuracy": acc,
            "prod_f1_macro": f1,
            "perf_trigger": float(trig),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# EvidentlyAI report
# ══════════════════════════════════════════════════════════════════════════════


def generate_evidently_report(ref_df, prod_df, report_path) -> Dict:
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric
        from evidently.report import Report

        report = Report(
            metrics=[DataDriftPreset(), DataQualityPreset(), DatasetDriftMetric()]
        )
        ref_s = ref_df.sample(min(len(prod_df), len(ref_df)), random_state=42)
        cols = FEATURE_COLS + ([TARGET_COL] if TARGET_COL in prod_df.columns else [])
        report.run(reference_data=ref_s[cols], current_data=prod_df[cols])
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report.save_html(report_path)
        d = report.as_dict()["metrics"][-1]["result"]
        print(
            f"\nEvidentlyAI → {report_path}  "
            f"drifted={d.get('dataset_drift',False)}  "
            f"features={d.get('number_of_drifted_columns','?')}"
        )
        return {
            "dataset_drifted": d.get("dataset_drift", False),
            "n_features_drifted": d.get("number_of_drifted_columns", 0),
        }
    except Exception as e:
        print(f"\nEvidentlyAI report skipped: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Main monitoring cycle — logged as an MLflow run
# ══════════════════════════════════════════════════════════════════════════════


def run_monitoring_cycle(production_path=None, report_only=False) -> Dict:
    print("=" * 55)
    print("  Iris CT — Monitoring Cycle  (MLflow tracking ON)")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    ref_df = pd.read_csv(PATHS["reference_data"])
    prod_df = pd.read_csv(production_path or PATHS["production_data"])
    print(f"Reference: {len(ref_df)} rows  |  Production: {len(prod_df)} rows")

    bundle = None
    if os.path.exists(PATHS["current_model"]):
        with open(PATHS["current_model"], "rb") as f:
            bundle = pickle.load(f)
        print(
            f"Model: {PATHS['current_model']}  (trained {bundle.get('trained_at','?')})"
        )
    else:
        print("No current model found.")

    # Get current Production model run_id from MLflow Registry
    prod_run_id = "N/A"
    try:
        mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
        client = MlflowClient()
        pvs = client.get_latest_versions(
            MLFLOW_CFG["registered_model"], stages=["Production"]
        )
        if pvs:
            prod_run_id = pvs[0].run_id
    except Exception:
        pass

    results = {"timestamp": datetime.now(timezone.utc).isoformat()}

    if report_only:
        results["evidently"] = generate_evidently_report(
            ref_df, prod_df, PATHS["drift_report"]
        )
        return results

    # Run monitoring checks
    all_mlflow_metrics = {}

    min_n = CFG["triggers"]["drift_based"]["min_samples_for_drift"]
    if len(prod_df) >= min_n:
        fd = run_feature_drift(
            ref_df, prod_df, CFG["triggers"]["drift_based"]["features_to_monitor"]
        )
        results["feature_drift"] = {
            k: v for k, v in fd.items() if k != "_mlflow_metrics"
        }
        all_mlflow_metrics.update(fd.get("_mlflow_metrics", {}))
    else:
        print(f"\nDrift check: insufficient samples ({len(prod_df)} < {min_n})")

    if bundle:
        pd_res = monitor_prediction_distribution(bundle, ref_df, prod_df)
        results["prediction_drift"] = {
            k: v for k, v in pd_res.items() if k != "_mlflow_metrics"
        }
        all_mlflow_metrics.update(pd_res.get("_mlflow_metrics", {}))

        perf = monitor_performance(bundle, prod_df)
        if perf:
            results["performance"] = {
                k: v for k, v in perf.items() if k != "_mlflow_metrics"
            }
            all_mlflow_metrics.update(perf.get("_mlflow_metrics", {}))

    ev = generate_evidently_report(ref_df, prod_df, PATHS["drift_report"])
    results["evidently"] = ev

    # Aggregate trigger
    triggers = []
    if results.get("feature_drift", {}).get("trigger_retraining"):
        triggers.append("feature_drift")
    if results.get("prediction_drift", {}).get("trigger_retraining"):
        triggers.append("prediction_drift")
    if results.get("performance", {}) and results["performance"].get(
        "trigger_retraining"
    ):
        triggers.append("performance_slo_breach")

    results["trigger_retraining"] = bool(len(triggers) > 0)
    results["trigger_reasons"] = triggers
    all_mlflow_metrics["overall_trigger"] = float(results["trigger_retraining"])

    # ── Log to MLflow as a monitoring run ──────────────────────────────────────
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"] + "_monitoring")
    with mlflow.start_run(
        nested=True,
        run_name=f"monitor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    ) as mrun:
        mlflow.log_metrics(all_mlflow_metrics)
        mlflow.set_tags(
            {
                "run_type": "monitoring",
                "trigger_retraining": str(results["trigger_retraining"]),
                "trigger_reasons": ",".join(triggers) or "none",
                "n_production_rows": str(len(prod_df)),
                "production_model_run_id": prod_run_id,
            }
        )
        # Log EvidentlyAI report as artefact
        if os.path.exists(PATHS["drift_report"]):
            mlflow.log_artifact(PATHS["drift_report"], artifact_path="drift_reports")
        print(f"\nMonitoring logged → MLflow run {mrun.info.run_id[:8]}…")

    # Print verdict
    print("\n" + "=" * 55)
    if results["trigger_retraining"]:
        print(f"  RETRAINING TRIGGERED 🚨  reasons: {', '.join(triggers)}")
        print("  Run: python src/pipeline.py")
    else:
        print("  All checks passed ✅  No retraining needed.")
    print("=" * 55)

    # Save JSON log
    os.makedirs("logs", exist_ok=True)
    path = "logs/monitoring_history.json"
    history = json.load(open(path)) if os.path.exists(path) else []
    history.append(_safe(results))
    json.dump(history, open(path, "w"), indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=str, default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    run_monitoring_cycle(production_path=args.production, report_only=args.report_only)
