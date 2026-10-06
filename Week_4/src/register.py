# ============================================================
#  STEP 4 — REGISTER THE MODEL
#  src/register.py
#
#  WHAT THIS DOES:
#    After evaluation passes, we "register" the model.
#    That means:
#      1. Save it to disk as a versioned artifact
#      2. Write a model card (metadata about the run)
#      3. Promote it to "champion" so future runs are compared to it
#
#  In real MLOps this step pushes to a model registry
#  (MLflow Model Registry, Vertex AI, SageMaker, etc.)
#  Here we keep it simple: just local files.
#
#  RUN IT:
#    python src/register.py
# ============================================================

import json
import os
import pickle
import mlflow
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

LATEST_METRICS_PATH  = "models/latest_metrics.json"
CHAMPION_METRICS_PATH = "models/champion_metrics.json"
MODEL_DIR = "models"
RANDOM_SEED = 42


def register():
    print("=" * 55)
    print("  MODEL REGISTRATION")
    print("=" * 55)

    # ── Load latest metrics ────────────────────────────────
    if not os.path.exists(LATEST_METRICS_PATH):
        print(f"\n❌  No trained model metrics found.")
        print("    Run train.py → evaluate.py → register.py in order.")
        return

    with open(LATEST_METRICS_PATH) as f:
        metrics = json.load(f)

    run_id   = metrics["run_id"]
    accuracy = metrics["accuracy"]
    print(f"\n📦  Registering model from run: {run_id[:8]}...")

    # ── Load model from MLflow ─────────────────────────────
    # We pull the model back out of the MLflow run we logged it to.
    model_uri = f"runs:/{run_id}/model"
    print(f"    Loading from MLflow: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # ── Save model as a versioned file ─────────────────────
    # Use a timestamp so every registered model has a unique filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"iris_rf_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved model artifact → {model_path}")

    # ── Write a model card ─────────────────────────────────
    # A "model card" is documentation about a model:
    # what it does, how it was trained, what its limits are.
    model_card = {
        "model_name":      "iris-random-forest",
        "version":         timestamp,
        "mlflow_run_id":   run_id,
        "accuracy":        accuracy,
        "n_estimators":    metrics["n_estimators"],
        "max_depth":       metrics["max_depth"],
        "trained_on":      datetime.now().isoformat(),
        "dataset":         "Iris (UCI)",
        "features":        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes":         ["setosa", "versicolor", "virginica"],
        "artifact_path":   model_path,
        "framework":       "scikit-learn",
    }

    card_path = os.path.join(MODEL_DIR, f"model_card_{timestamp}.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"    Saved model card   → {card_path}")

    # ── Promote to champion ────────────────────────────────
    # "Champion" = the model we'll compare future runs against.
    # We copy the current metrics into champion_metrics.json.
    with open(CHAMPION_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Promoted to champion → {CHAMPION_METRICS_PATH}")

    # ── Quick smoke-test ───────────────────────────────────
    # Before declaring success, predict on one sample to make
    # sure the saved model loads and runs correctly.
    print(f"\n🔍  Smoke test (predict on 1 sample) ...")
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                          columns=["sepal_length", "sepal_width",
                                   "petal_length", "petal_width"])
    prediction = model.predict(sample)[0]
    print(f"    Input : [5.1, 3.5, 1.4, 0.2]")
    print(f"    Output: {prediction}  (expected: setosa)")

    print(f"\n✅  Model registered successfully!")
    print(f"    Artifact : {model_path}")
    print(f"    Model card: {card_path}")
    print(f"    Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)\n")

    return model_path


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    register()
