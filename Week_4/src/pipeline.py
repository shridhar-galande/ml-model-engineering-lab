# ============================================================
#  STEP 5 — THE PIPELINE  (your "CI/CD trigger")
#  src/pipeline.py
#
#  WHAT THIS DOES:
#    Runs ALL steps in order with one command.
#    This is the equivalent of a CI/CD pipeline being triggered
#    by a git push.
#
#    Step 1 → Step 2 → Step 3 → Step 4
#    data     train    eval     register
#    check
#
#    If ANY step fails, the pipeline stops immediately.
#    That's a "quality gate" blocking a bad run from proceeding.
#
#  RUN IT:
#    python src/pipeline.py
#    python src/pipeline.py --n-estimators 50
#    python src/pipeline.py --n-estimators 50 --max-depth 3
# ============================================================

import sys
import os
import time
import argparse

# Add the src/ directory to Python's path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from data_check import run_checks
from train      import train
from evaluate   import evaluate
from register   import register


def run_pipeline(n_estimators: int = 100, max_depth: int = None):
    """
    Run the full ML pipeline end-to-end.
    """
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  🚀  MLOPS PIPELINE STARTING")
    print("=" * 60)
    print(f"  Config: n_estimators={n_estimators}, max_depth={max_depth}")
    print("=" * 60 + "\n")

    # ── STEP 1: Data quality gate ──────────────────────────
    print("\n📋  STEP 1/4  —  Data Quality Check\n")
    t0 = time.time()
    try:
        # fix=True so pipeline auto-cleans data and continues
        run_checks(fix=True)
        print(f"   ⏱  Done in {time.time()-t0:.1f}s")
    except SystemExit:
        print("\n⛔  PIPELINE ABORTED at Step 1 (data quality)")
        sys.exit(1)

    # ── STEP 2: Train ──────────────────────────────────────
    print("\n\n🌲  STEP 2/4  —  Training\n")
    t0 = time.time()
    try:
        accuracy, run_id = train(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        print(f"   ⏱  Done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n⛔  PIPELINE ABORTED at Step 2 (training): {e}")
        sys.exit(1)

    # ── STEP 3: Evaluate ───────────────────────────────────
    print("\n\n📊  STEP 3/4  —  Evaluation Gate\n")
    t0 = time.time()
    try:
        evaluate()
        print(f"   ⏱  Done in {time.time()-t0:.1f}s")
    except SystemExit:
        print("\n⛔  PIPELINE ABORTED at Step 3 (model evaluation)")
        print("    The model did not meet quality standards.")
        print("    No model was registered or deployed.")
        sys.exit(1)

    # ── STEP 4: Register ───────────────────────────────────
    print("\n\n📦  STEP 4/4  —  Model Registration\n")
    t0 = time.time()
    try:
        model_path = register()
        print(f"   ⏱  Done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n⛔  PIPELINE ABORTED at Step 4 (registration): {e}")
        sys.exit(1)

    # ── Done ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time   : {elapsed:.1f}s")
    print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  MLflow run   : {run_id[:8]}...")
    print(f"  Model saved  : {model_path}")
    print(f"\n  👉  View all runs:  mlflow ui  →  http://localhost:5000")
    print("=" * 60 + "\n")


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Iris MLOps pipeline"
    )
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max tree depth (default: unlimited)")
    args = parser.parse_args()

    run_pipeline(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )

    # ── TEACHING MOMENT ───────────────────────────────────
    # Try running the pipeline a few times with different params:
    #
    #   python src/pipeline.py --n-estimators 10
    #   python src/pipeline.py --n-estimators 200
    #   python src/pipeline.py --n-estimators 100 --max-depth 2
    #
    # Then open MLflow UI and compare the runs side-by-side.
    # Every run is logged with all its params and metrics.
    # THAT is experiment lineage — you can see exactly what
    # changed between runs and why one was better than another.
