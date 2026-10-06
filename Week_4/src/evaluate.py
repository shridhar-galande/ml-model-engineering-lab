# ============================================================
#  STEP 3 — MODEL EVALUATION GATE
#  src/evaluate.py
#
#  WHAT THIS DOES:
#    After training, we don't just blindly deploy.
#    We check:
#      (a) Is the new model good ENOUGH? (absolute threshold)
#      (b) Is it BETTER than the current champion? (challenger test)
#    If either check fails → model is rejected.
#
#  This is "model validation" in MLOps.
#
#  RUN IT:
#    python src/evaluate.py
# ============================================================

import json
import os
import sys

# ── CONFIG ────────────────────────────────────────────────────
LATEST_METRICS_PATH  = "models/latest_metrics.json"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"

# Absolute threshold: model must be at LEAST this accurate.
# In real projects this is set by the business (e.g. "must beat random by X%")
ACCURACY_THRESHOLD = 0.88


# ── HELPER ───────────────────────────────────────────────────
def gate(condition: bool, message: str):
    """Print PASS or FAIL. Exit on failure."""
    if condition:
        print(f"  ✅  PASS  — {message}")
    else:
        print(f"  ❌  FAIL  — {message}")
        print("\n⛔  Model rejected. It will NOT be registered.")
        sys.exit(1)


def load_metrics(path: str):
    """Load a JSON metrics file, return None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── MAIN ─────────────────────────────────────────────────────
def evaluate():
    print("=" * 55)
    print("  MODEL EVALUATION GATE")
    print("=" * 55)

    # ── Load latest metrics ────────────────────────────────
    latest = load_metrics(LATEST_METRICS_PATH)
    if latest is None:
        print(f"\n❌  No trained model found at {LATEST_METRICS_PATH}")
        print("    Run  python src/train.py  first.")
        sys.exit(1)

    accuracy     = latest["accuracy"]
    run_id       = latest["run_id"]
    n_estimators = latest["n_estimators"]

    print(f"\n📊  New model  (run: {run_id[:8]}...)")
    print(f"    Accuracy     : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"    n_estimators : {n_estimators}")

    # ── CHECK 1: Absolute threshold ────────────────────────
    print(f"\n[ Check 1 ] Does model meet minimum accuracy threshold ({ACCURACY_THRESHOLD})?")
    gate(
        accuracy >= ACCURACY_THRESHOLD,
        f"Accuracy {accuracy:.4f} ≥ threshold {ACCURACY_THRESHOLD}"
    )

    # ── CHECK 2: Champion/challenger ───────────────────────
    # Is there an existing champion model to beat?
    print(f"\n[ Check 2 ] Does new model beat (or match) the current champion?")
    champion = load_metrics(CHAMPION_METRICS_PATH)

    if champion is None:
        # No champion yet — first model automatically wins
        print(f"  ✅  PASS  — No champion exists yet. First model wins automatically.")
    else:
        champ_acc = champion["accuracy"]
        champ_run = champion["run_id"]
        print(f"    Champion (run: {champ_run[:8]}...)  accuracy: {champ_acc:.4f}")

        gate(
            accuracy >= champ_acc,
            f"New model ({accuracy:.4f}) ≥ champion ({champ_acc:.4f})"
        )

    # ── All checks passed ──────────────────────────────────
    print(f"\n✅  Model approved! Ready to register.")
    return True


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()

    # ── TEACHING MOMENT ───────────────────────────────────
    # Try this:
    #   Open evaluate.py and change ACCURACY_THRESHOLD = 0.99
    #   Then re-run: python src/evaluate.py
    #   → Watch the gate REJECT the model.
    #
    # This is exactly what CI/CD does in production — it refuses
    # to deploy a model that doesn't meet the bar.
