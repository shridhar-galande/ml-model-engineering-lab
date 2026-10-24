"""
pipeline.py
===========
End-to-end orchestration of the Iris continuous training pipeline.
Every pipeline run is recorded as an MLflow parent run, with child
runs for monitoring, trigger evaluation, training, and evaluation.

MLflow parent run structure:
  iris_ct experiment
  └── pipeline_run  (parent)
      ├── monitoring run   (child)
      ├── trigger run      (child — in iris_ct_monitoring)
      ├── training run     (child)
      └── evaluation run   (child)

Usage:
    python src/pipeline.py               # full automated pipeline
    python src/pipeline.py --retrain     # skip trigger, force retrain
    python src/pipeline.py --monitor-only
    python src/pipeline.py --dry-run     # evaluate but do NOT promote
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from typing import Dict

import mlflow
import yaml

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS = CFG["paths"]
MLFLOW_CFG = CFG["mlflow"]
MLFLOW_CFG["tracking_uri"] = os.getenv(
    "MLFLOW_TRACKING_URI", MLFLOW_CFG["tracking_uri"]
)


def _safe(obj):
    import numpy as np

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


def _section(label, n, total=6):
    print(f"\n{'─'*55}")
    print(f"  Stage {n}/{total}  {label}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════


def run_pipeline(force_retrain=False, monitor_only=False, dry_run=False) -> Dict:
    start = datetime.now(timezone.utc)
    print("=" * 55)
    print("  Iris CT — Continuous Training Pipeline")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  MLflow tracking: {MLFLOW_CFG['tracking_uri']}")
    if dry_run:
        print("  [DRY RUN — promotion disabled]")
    if force_retrain:
        print("  [FORCE RETRAIN]")
    print("=" * 55)

    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    run_record: Dict = {
        "start_time": start.isoformat(),
        "force_retrain": force_retrain,
        "monitor_only": monitor_only,
        "dry_run": dry_run,
        "outcome": "started",
    }

    # ── Parent MLflow run (wraps the whole pipeline) ───────────────────────────
    with mlflow.start_run(
        run_name=f"pipeline_{start.strftime('%Y%m%d_%H%M%S')}"
    ) as parent_run:

        parent_run_id = parent_run.info.run_id
        print(f"\nMLflow parent run: {parent_run_id}")
        print(f"MLflow UI: mlflow ui --backend-store-uri {MLFLOW_CFG['tracking_uri']}")

        mlflow.set_tags(
            {
                "run_type": "pipeline",
                "force_retrain": str(force_retrain),
                "monitor_only": str(monitor_only),
                "dry_run": str(dry_run),
            }
        )

        try:
            # ── Stage 1: Monitor ──────────────────────────────────────────────
            _section("MONITOR — Drift & Performance", 1)
            from monitor import run_monitoring_cycle

            mon = run_monitoring_cycle()
            run_record["monitor"] = {
                "trigger_retraining": mon.get("trigger_retraining"),
                "trigger_reasons": mon.get("trigger_reasons", []),
                "feature_drift": mon.get("feature_drift", {}),
                "performance": mon.get("performance", {}),
            }
            mlflow.log_metrics(
                {
                    "monitor_trigger": float(mon.get("trigger_retraining", False)),
                }
            )
            print(f"  ✅ Monitoring complete  trigger={mon.get('trigger_retraining')}")

            if monitor_only:
                run_record["outcome"] = "monitor_only"
                mlflow.set_tag("outcome", "monitor_only")
                _log_and_exit(run_record, parent_run_id)
                return run_record

            # ── Stage 2: Trigger ──────────────────────────────────────────────
            _section("TRIGGER — Should we retrain?", 2)
            if not force_retrain:
                from retrain_trigger import run_trigger_evaluation

                trig = run_trigger_evaluation()
                run_record["trigger"] = {
                    "retrain": trig["retrain"],
                    "triggered_by": trig["triggered_by"],
                }
                run_record["trigger_results"] = trig["trigger_results"]
                run_record["trigger_reasons"] = trig["triggered_by"]
                mlflow.log_metric("trigger_retrain", float(trig["retrain"]))

                if not trig["retrain"]:
                    run_record["outcome"] = "no_retrain_needed"
                    mlflow.set_tag("outcome", "no_retrain_needed")
                    print("  ✅ No triggers fired — pipeline complete.")
                    _log_and_exit(run_record, parent_run_id)
                    return run_record
                print(f"  🚨 Retraining triggered by: {trig['triggered_by']}")
            else:
                run_record["trigger"] = {"retrain": True, "triggered_by": ["forced"]}
                run_record["trigger_reasons"] = ["forced"]
                mlflow.log_metric("trigger_retrain", 1.0)
                print("  🔧 Force retrain — skipping trigger check.")

            # ── Stage 3: Train ────────────────────────────────────────────────
            _section("TRAIN — Retrain on combined data", 3)
            from train import run_training

            # Nest training as a child run of the pipeline
            with mlflow.start_run(
                run_name="train_child",
                nested=True,
                tags={"run_type": "training", "parent_pipeline_run": parent_run_id},
            ):
                train_metrics = run_training(combine=True)

            run_record["train_metrics"] = {
                "accuracy": train_metrics.get("accuracy"),
                "f1_macro": train_metrics.get("f1_macro"),
            }
            mlflow.log_metrics(
                {
                    "train_accuracy": train_metrics.get("accuracy", 0),
                    "train_f1_macro": train_metrics.get("f1_macro", 0),
                }
            )
            print(
                f"  ✅ Training complete  "
                f"acc={train_metrics.get('accuracy')} f1={train_metrics.get('f1_macro')}"
            )

            # ── Stage 4: Evaluate ─────────────────────────────────────────────
            _section("EVALUATE — All 5 gates", 4)
            from evaluate import run_evaluation

            eval_result = run_evaluation(auto_promote=(not dry_run))
            run_record["evaluation"] = {
                "all_passed": eval_result["all_passed"],
                "promoted": eval_result.get("promoted", False),
            }
            mlflow.log_metrics(
                {
                    "eval_all_passed": float(eval_result["all_passed"]),
                    "eval_promoted": float(eval_result.get("promoted", False)),
                }
            )
            mlflow.set_tag(
                "model_version", str(eval_result.get("model_version", "N/A"))
            )

            if not eval_result["all_passed"]:
                run_record["outcome"] = "evaluation_failed"
                mlflow.set_tag("outcome", "evaluation_failed")
                print("  🚨 Evaluation failed — no promotion.")
                _log_and_exit(run_record, parent_run_id)
                return run_record

            # ── Stage 5 ───────────────────────────────────────────────────────
            if dry_run:
                run_record["outcome"] = "dry_run_complete"
                run_record["promoted"] = False
                mlflow.set_tag("outcome", "dry_run_complete")
                print("  ⏭  Dry run — promotion skipped.")
            else:
                run_record["promoted"] = eval_result.get("promoted", False)
                run_record["outcome"] = "promoted"
                mlflow.set_tag("outcome", "promoted")
                print(
                    f"  ✅ Model promoted to Production  "
                    f"version={eval_result.get('model_version')}"
                )

        except Exception as exc:
            run_record["outcome"] = f"error: {exc}"
            mlflow.set_tag(
                "outcome", "error"
            )
            mlflow.set_tag("error_message", str(exc))
            print(f"  🚨 Pipeline error: {exc}")
            import traceback

            traceback.print_exc()

        finally:
            end = datetime.now(timezone.utc)
            run_record["end_time"] = end.isoformat()
            run_record["duration_secs"] = round((end - start).total_seconds(), 2)

            mlflow.log_metrics({"pipeline_duration_secs": run_record["duration_secs"]})
            mlflow.set_tag("final_outcome", run_record.get("outcome", "unknown"))

            # ── Stage 6: Notify ───────────────────────────────────────────────
            _section("NOTIFY — Log outcome", 6)
            os.makedirs("logs", exist_ok=True)
            path = "logs/pipeline_runs.json"
            history = json.load(open(path)) if os.path.exists(path) else []
            history.append(_safe(run_record))
            json.dump(history, open(path, "w"), indent=2)
            print(f"  ✅ Pipeline run logged → {path}  ({len(history)} total)")
            print(f"  MLflow parent run: {parent_run_id}")
            print(f"  View: mlflow ui --backend-store-uri {MLFLOW_CFG['tracking_uri']}")

    # Summary
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print(f"  Outcome:  {run_record['outcome'].upper()}")
    print(f"  Duration: {run_record.get('duration_secs', 0):.1f}s")
    print(f"  MLflow:   {MLFLOW_CFG['tracking_uri']}")
    print("=" * 55)
    return run_record


def _log_and_exit(rec, run_id):
    import mlflow as _ml

    end = datetime.now(timezone.utc)
    rec["end_time"] = end.isoformat()
    rec["duration_secs"] = round(
        (end - datetime.fromisoformat(rec["start_time"])).total_seconds(), 2
    )
    _ml.log_metric(
        "pipeline_duration_secs",
        rec["duration_secs"],
    )
    os.makedirs("logs", exist_ok=True)
    path = "logs/pipeline_runs.json"
    history = json.load(open(path)) if os.path.exists(path) else []
    history.append(_safe(rec))
    json.dump(history, open(path, "w"), indent=2)
    print(f"  Pipeline run logged → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        force_retrain=args.retrain,
        monitor_only=args.monitor_only,
        dry_run=args.dry_run,
    )
