# ============================================================
#  STEP 2 — TRAIN + TRACK EXPERIMENT
#  src/train.py
#
#  WHAT THIS DOES:
#    1. Loads the clean data
#    2. Trains a Random Forest classifier
#    3. Logs EVERYTHING to MLflow:
#         - parameters (n_estimators, max_depth, random_seed)
#         - metrics   (accuracy, per-class precision/recall)
#         - the model itself as a versioned artifact
#         - the Python + library versions (reproducibility!)
#
#  RUN IT:
#    python src/train.py
#    python src/train.py --n-estimators 50   ← try different params
#
#  THEN: mlflow ui  →  open http://localhost:5000
# ============================================================

import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import platform
import sklearn
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── CONFIG ────────────────────────────────────────────────────
CLEAN_DATA_PATH = "data/iris_with_issues_clean.csv"
FALLBACK_DATA   = "data/iris_with_issues.csv"      # if clean doesn't exist yet
MLFLOW_EXPERIMENT = "iris-classifier"
RANDOM_SEED = 42                                   # fixed seed = reproducible splits


# ── MAIN ─────────────────────────────────────────────────────
def train(n_estimators: int = 100, max_depth: int = None):
    """
    Train a Random Forest and log the run to MLflow.

    Parameters
    ----------
    n_estimators : int   — number of trees in the forest
    max_depth    : int   — max depth of each tree (None = unlimited)
    """

    # ── Load data ──────────────────────────────────────────
    data_path = CLEAN_DATA_PATH if os.path.exists(CLEAN_DATA_PATH) else FALLBACK_DATA
    print(f"\n[Data]  Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Features (X) and label (y)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    # Split — SAME random_state every run = reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"    Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    # ── Set up MLflow experiment ───────────────────────────
    # An "experiment" groups related runs together.
    # Think of it like a folder for all your iris experiments.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n[MLflow]  Run started  (id: {run_id[:8]}...)")

        # ── ARTIFACT: log parameters ───────────────────────
        # Everything that affects the model output should be logged.
        # This is what lets you replay any run exactly.
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth",    max_depth)
        mlflow.log_param("random_seed",  RANDOM_SEED)
        mlflow.log_param("data_path",    data_path)
        mlflow.log_param("train_rows",   len(X_train))
        mlflow.log_param("test_rows",    len(X_test))

        # ── ARTIFACT: log environment (reproducibility) ────
        # If you need to re-run this in 6 months, you need to
        # know exactly which library versions were used.
        mlflow.log_param("python_version",  platform.python_version())
        mlflow.log_param("sklearn_version", sklearn.__version__)

        # ── Train ──────────────────────────────────────────
        print(f"\n[Train]  RandomForest  "
              f"(n_estimators={n_estimators}, max_depth={max_depth})")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_SEED   # ← fixed seed = same trees every run
        )
        model.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────────────────
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # ── ARTIFACT: log metrics ──────────────────────────
        mlflow.log_metric("accuracy", accuracy)
        for species in ["setosa", "versicolor", "virginica"]:
            if species in report:
                mlflow.log_metric(f"{species}_f1",       report[species]["f1-score"])
                mlflow.log_metric(f"{species}_precision", report[species]["precision"])
                mlflow.log_metric(f"{species}_recall",    report[species]["recall"])

        # ── ARTIFACT: log the model itself ─────────────────
        # MLflow saves the model in a versioned folder.
        # You can load it later with: mlflow.sklearn.load_model(...)
        deployment_model_path = os.path.join("models", f"mlflow_model_{run_id}")
        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.save_model(
            model,
            path=deployment_model_path,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # ── ARTIFACT: save metrics to a JSON file too ──────
        # This is what evaluate.py will read to compare runs.
        metrics_path = "models/latest_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "run_id":       run_id,
                "accuracy":     accuracy,
                "n_estimators": n_estimators,
                "max_depth":    str(max_depth),
                "model_path":   deployment_model_path,
            }, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # ── Print summary ──────────────────────────────────
        print(f"\n{'-'*45}")
        print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.1f}%)")
        print(f"  MLflow run   : {run_id[:8]}...")
        print(f"  Params logged: n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"{'-'*45}")
        print(f"\n[OK]  Training complete!")
        print(f"    Run:  mlflow ui   ->   http://localhost:5000")
        print(f"    Look for experiment: '{MLFLOW_EXPERIMENT}'\n")

        return accuracy, run_id


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max tree depth (default: unlimited)")
    args = parser.parse_args()

    train(n_estimators=args.n_estimators, max_depth=args.max_depth)
