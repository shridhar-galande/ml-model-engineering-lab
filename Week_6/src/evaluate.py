"""
evaluate.py
===========
Runs all 5 evaluation gates before a candidate model can be promoted
to production. Uses MLflow Model Registry for the full
  None → Staging → Production transition.

MLflow integration:
  - Compares candidate run metrics vs current Production model metrics
  - Transitions candidate: Staging → Production (if all gates pass)
  - Archives old Production model version
  - Logs gate pass/fail results back to the candidate MLflow run
  - Tags runs with evaluation outcome

Usage:
    python src/evaluate.py              # evaluate & print results
    python src/evaluate.py --promote    # auto-promote if all gates pass
"""

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FEATURE_COLS = CFG["features"]["names"]
TARGET_COL = CFG["features"]["target"]
GATES = CFG["evaluation_gates"]
PATHS = CFG["paths"]
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


def _setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    return MlflowClient()


def _load_local(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(bundle: Dict, df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    return bundle["model"].predict(X), bundle["model"].predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# Fetch current Production model metrics from MLflow Registry
# ══════════════════════════════════════════════════════════════════════════════


def get_production_model_metrics(client: MlflowClient) -> Optional[Dict]:
    """
    Return the metrics of the currently-Production model version from MLflow.
    Returns None if no Production version exists yet.
    """
    try:
        versions = client.get_latest_versions(
            MLFLOW_CFG["registered_model"], stages=[MLFLOW_CFG["production_stage"]]
        )
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return {
            k: float(v)
            for k, v in run.data.metrics.items()
            if not k.startswith("importance_")
        }
    except Exception:
        return None


def get_candidate_run_id(bundle: Dict) -> Optional[str]:
    return bundle.get("metrics", {}).get("mlflow_run_id")


# ══════════════════════════════════════════════════════════════════════════════
# Gate 1 — Baseline comparison (candidate vs Production in Registry)
# ══════════════════════════════════════════════════════════════════════════════


def gate_baseline_comparison(
    candidate_bundle: Dict, production_metrics: Optional[Dict], test_df: pd.DataFrame
) -> Tuple[bool, Dict]:
    print("\nGate 1 — Baseline Comparison (vs MLflow Production model)")
    print("-" * 50)

    y_true = test_df[TARGET_COL].values
    y_pred, _ = predict(candidate_bundle, test_df)
    cand_acc = float(accuracy_score(y_true, y_pred))

    if production_metrics is None:
        print("  No Production model in Registry — first deployment. PASS ✅")
        return True, {
            "result": "first_deployment",
            "candidate_accuracy": round(cand_acc, 4),
        }

    prod_acc = production_metrics.get("accuracy", 0.0)
    margin = GATES["beat_margin"]
    passed = cand_acc >= (prod_acc - margin)
    delta = cand_acc - prod_acc

    print(f"  Production accuracy (MLflow Registry): {prod_acc:.4f}")
    print(f"  Candidate accuracy:                    {cand_acc:.4f}  (Δ {delta:+.4f})")
    print(f"  Required margin: ≥ {-margin:.4f}")
    print(f"  Result: {'PASS ✅' if passed else 'FAIL 🚨'}")
    return passed, {
        "production_accuracy": round(prod_acc, 4),
        "candidate_accuracy": round(cand_acc, 4),
        "delta": round(delta, 4),
        "passed": bool(passed),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gate 2 — Minimum thresholds
# ══════════════════════════════════════════════════════════════════════════════


def gate_minimum_thresholds(bundle: Dict, test_df: pd.DataFrame) -> Tuple[bool, Dict]:
    print("\nGate 2 — Minimum Absolute Thresholds")
    print("-" * 50)
    y_true = test_df[TARGET_COL].values
    y_pred, _ = predict(bundle, test_df)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    passed = acc >= GATES["minimum_accuracy"] and f1 >= GATES["minimum_f1_macro"]
    print(
        f"  Accuracy:   {acc:.4f}  (min={GATES['minimum_accuracy']})  "
        f"{'✅' if acc>=GATES['minimum_accuracy'] else '🚨'}"
    )
    print(
        f"  F1 (macro): {f1:.4f}  (min={GATES['minimum_f1_macro']})  "
        f"{'✅' if f1>=GATES['minimum_f1_macro'] else '🚨'}"
    )
    print(f"  Result: {'PASS ✅' if passed else 'FAIL 🚨'}")
    return passed, {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "passed": bool(passed),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gate 3 — Golden dataset (100% on curated known examples)
# ══════════════════════════════════════════════════════════════════════════════


def gate_golden_dataset(bundle: Dict, golden_df: pd.DataFrame) -> Tuple[bool, Dict]:
    print("\nGate 3 — Golden Dataset Assertion")
    print("-" * 50)
    y_true = golden_df[TARGET_COL].values
    y_pred, _ = predict(bundle, golden_df)
    acc = float(accuracy_score(y_true, y_pred))
    n_wrong = int((y_pred != y_true).sum())
    passed = acc >= GATES["golden_accuracy"]
    print(
        f"  {len(golden_df)} examples ({len(golden_df)//3}/class)  |  "
        f"wrong={n_wrong}  |  acc={acc:.4f}"
    )
    if n_wrong > 0:
        for idx in np.where(y_pred != y_true)[0]:
            r = golden_df.iloc[idx]
            print(
                f"    Row {idx}: true={CLASS_NAMES[int(y_true[idx])]} "
                f"pred={CLASS_NAMES[int(y_pred[idx])]} "
                f"features={r[FEATURE_COLS].to_dict()}"
            )
    print(f"  Result: {'PASS ✅' if passed else 'FAIL 🚨'}")
    return passed, {
        "n_golden": len(golden_df),
        "n_wrong": n_wrong,
        "accuracy": round(acc, 4),
        "passed": bool(passed),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gate 4 — Per-class recall floor
# ══════════════════════════════════════════════════════════════════════════════


def gate_per_class_recall(bundle: Dict, test_df: pd.DataFrame) -> Tuple[bool, Dict]:
    print("\nGate 4 — Per-Class Recall")
    print("-" * 50)
    y_true = test_df[TARGET_COL].values
    y_pred, _ = predict(bundle, test_df)
    min_r = GATES["per_class_min_recall"]
    cls_res = {}
    all_pass = True
    for i, cls in enumerate(CLASS_NAMES):
        r = float(recall_score(y_true == i, y_pred == i, zero_division=0))
        ok = r >= min_r
        if not ok:
            all_pass = False
        cls_res[cls] = {"recall": round(r, 4), "passed": bool(ok)}
        print(f"  {cls:<14} recall={r:.4f}  (min={min_r})  {'✅' if ok else '🚨'}")
    print(f"  Result: {'PASS ✅' if all_pass else 'FAIL 🚨'}")
    return all_pass, {"per_class": cls_res, "passed": bool(all_pass)}


# ══════════════════════════════════════════════════════════════════════════════
# Gate 5 — Training data health
# ══════════════════════════════════════════════════════════════════════════════


def gate_distribution_check(df: pd.DataFrame) -> Tuple[bool, Dict]:
    print("\nGate 5 — Training Data Distribution")
    print("-" * 50)
    issues = []
    # Missing
    missing = df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()
    ok_miss = missing == 0
    if not ok_miss:
        issues.append(f"missing_values={missing}")
    print(f"  Missing values: {'✅ None' if ok_miss else f'🚨 {missing}'}")
    # Class balance
    min_pct = df[TARGET_COL].value_counts(normalize=True).min()
    ok_bal = min_pct >= 0.05
    if not ok_bal:
        issues.append(f"class_imbalance={min_pct:.1%}")
    print(f"  Class balance:  {'✅' if ok_bal else '🚨'}  min={min_pct:.1%}")
    # Sample size
    ok_n = len(df) >= 30
    if not ok_n:
        issues.append(f"too_few_samples={len(df)}")
    print(f"  Sample size:    {'✅' if ok_n else '🚨'}  n={len(df)}")
    passed = len(issues) == 0
    print(f"  Result: {'PASS ✅' if passed else 'FAIL 🚨'}")
    return passed, {"passed": bool(passed), "issues": issues}


# ══════════════════════════════════════════════════════════════════════════════
# MLflow Registry promotion
# ══════════════════════════════════════════════════════════════════════════════


def promote_in_registry(client: MlflowClient, run_id: str) -> str:
    """
    1. Find the model version registered from this run_id
    2. Transition it None → Staging → Production
    3. Archive the previous Production version
    """
    reg_name = MLFLOW_CFG["registered_model"]

    # Find version for this run
    all_versions = client.search_model_versions(f"name='{reg_name}'")
    candidate_version = next((v for v in all_versions if v.run_id == run_id), None)
    if candidate_version is None:
        raise ValueError(f"No registered version found for run_id={run_id}")

    ver_num = candidate_version.version
    print(f"\n  Candidate version: {ver_num}  (run_id={run_id[:8]}…)")

    # Archive current Production versions
    prod_versions = client.get_latest_versions(reg_name, stages=["Production"])
    for pv in prod_versions:
        print(f"  Archiving old Production version: {pv.version}")
        client.transition_model_version_stage(
            name=reg_name,
            version=pv.version,
            stage="Archived",
            archive_existing_versions=False,
        )

    # Transition candidate → Staging → Production
    client.transition_model_version_stage(
        name=reg_name, version=ver_num, stage="Staging"
    )
    print(f"  Version {ver_num} → Staging")

    client.transition_model_version_stage(
        name=reg_name, version=ver_num, stage="Production"
    )
    print(f"  Version {ver_num} → Production ✅")

    # Tag with promotion timestamp
    client.set_model_version_tag(
        name=reg_name,
        version=ver_num,
        key="promoted_at",
        value=datetime.now(timezone.utc).isoformat(),
    )
    client.set_model_version_tag(
        name=reg_name, version=ver_num, key="gate_evaluation", value="all_passed"
    )

    return ver_num


def promote_local_pkl() -> None:
    """Also update the local current_model.pkl for pipeline.py compatibility."""
    import shutil

    src = PATHS["candidate_model"]
    dst = PATHS["current_model"]
    if not os.path.exists(src):
        return
    if os.path.exists(dst):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        arc = dst.replace(".pkl", f"_archived_{ts}.pkl")
        shutil.copy2(dst, arc)
    shutil.copy2(src, dst)
    print(f"  Local pkl promoted: {src} → {dst}")


def promote_candidate(candidate_path: str, production_path: str) -> None:
    """Compatibility wrapper used by CI/CD workflows."""
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate model not found: {candidate_path}")
    os.makedirs(os.path.dirname(production_path) or ".", exist_ok=True)
    import shutil

    if os.path.exists(production_path):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archived = production_path.replace(".pkl", f"_archived_{ts}.pkl")
        shutil.copy2(production_path, archived)
    shutil.copy2(candidate_path, production_path)
    print(f"  Promoted {candidate_path} → {production_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════


def run_evaluation(candidate_path: str = None, auto_promote: bool = False) -> Dict:
    print("=" * 55)
    print("  Iris CT — Evaluation Gates  (MLflow Registry)")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    client = _setup_mlflow()
    cand_path = candidate_path or PATHS["candidate_model"]
    candidate = _load_local(cand_path)
    if candidate is None:
        raise FileNotFoundError(
            f"Candidate not found: {cand_path}. Run train.py first."
        )

    run_id = get_candidate_run_id(candidate)
    print(f"\nCandidate pkl:    {cand_path}")
    print(f"MLflow run_id:    {run_id or 'NOT TRACKED'}")
    print(f"Trained at:       {candidate.get('trained_at','?')}")

    prod_metrics = get_production_model_metrics(client)
    if prod_metrics:
        print(
            f"Production model: accuracy={prod_metrics.get('accuracy','?')} (from MLflow Registry)"
        )
    else:
        print("Production model: None (first deployment)")

    # Load test + golden data
    ref_df = pd.read_csv(PATHS["reference_data"])
    _, test_df = train_test_split(
        ref_df, test_size=0.20, random_state=42, stratify=ref_df[TARGET_COL]
    )
    golden_df = pd.read_csv(PATHS["golden_dataset"])

    # Run gates
    gate_results = {}
    all_passed = True

    g1p, g1 = gate_baseline_comparison(candidate, prod_metrics, test_df)
    g2p, g2 = gate_minimum_thresholds(candidate, test_df)
    g3p, g3 = gate_golden_dataset(candidate, golden_df)
    g4p, g4 = gate_per_class_recall(candidate, test_df)
    g5p, g5 = gate_distribution_check(ref_df)

    gate_results.update(
        {
            "gate_1_baseline": g1,
            "gate_2_min_thresh": g2,
            "gate_3_golden": g3,
            "gate_4_per_class": g4,
            "gate_5_distribution": g5,
        }
    )
    all_passed = all([g1p, g2p, g3p, g4p, g5p])

    # Summary
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    for label, passed in [
        ("Gate 1 — Baseline vs Production", g1p),
        ("Gate 2 — Minimum thresholds", g2p),
        ("Gate 3 — Golden dataset", g3p),
        ("Gate 4 — Per-class recall", g4p),
        ("Gate 5 — Distribution check", g5p),
    ]:
        print(f"  {'✅ PASS' if passed else '🚨 FAIL'}  {label}")
    print(
        f"\n  Overall: {'ALL GATES PASSED ✅' if all_passed else 'EVALUATION FAILED 🚨'}"
    )

    # Log gate results back to the MLflow run
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_metrics(
                    {
                        "gate_baseline_pass": float(g1p),
                        "gate_min_thresh_pass": float(g2p),
                        "gate_golden_pass": float(g3p),
                        "gate_per_class_pass": float(g4p),
                        "gate_distribution_pass": float(g5p),
                        "all_gates_passed": float(all_passed),
                    }
                )
                mlflow.set_tags(
                    {
                        "evaluation_status": "passed" if all_passed else "failed",
                        "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            print(f"\nGate results logged back to MLflow run {run_id[:8]}…")
        except Exception as e:
            print(f"  (Could not log gate results to MLflow: {e})")

    # Promote
    promoted = False
    model_version = None
    if all_passed and auto_promote and run_id:
        print("\nPromoting candidate in MLflow Registry…")
        try:
            model_version = promote_in_registry(client, run_id)
            promote_local_pkl()
            promoted = True
            # Log the retrain history
            os.makedirs("logs", exist_ok=True)
            hist_path = PATHS["retrain_log"]
            history = json.load(open(hist_path)) if os.path.exists(hist_path) else []
            history.append(
                {
                    "event": "model_promoted",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "model_version": model_version,
                    "registry_name": MLFLOW_CFG["registered_model"],
                }
            )
            json.dump(_safe(history), open(hist_path, "w"), indent=2)
        except Exception as e:
            print(f"  Registry promotion failed: {e}")
            promote_local_pkl()
            promoted = True
    elif all_passed and not auto_promote:
        print("\nCandidate ready. Run: python src/evaluate.py --promote")
    elif not all_passed:
        print("\nCandidate REJECTED.")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": bool(all_passed),
        "gate_results": gate_results,
        "promoted": bool(promoted),
        "model_version": model_version,
        "run_id": run_id,
    }

    # Save evaluation log
    os.makedirs("logs", exist_ok=True)
    path = "logs/evaluation_history.json"
    history = json.load(open(path)) if os.path.exists(path) else []
    history.append(_safe(result))
    json.dump(history, open(path, "w"), indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default=None)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    run_evaluation(candidate_path=args.candidate, auto_promote=args.promote)
