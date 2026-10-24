"""
train.py
========
Full training pipeline for the Iris continuous training system.
Every training run is tracked in MLflow — params, metrics, model artefact,
feature importances, and a confusion-matrix PNG are all logged automatically.

MLflow tracks:
  - Hyperparameters       (log_params)
  - CV + validation metrics (log_metrics)
  - Per-class metrics     (log_metrics with class prefix)
  - Feature importances   (log_metrics)
  - Trained model         (mlflow.sklearn.log_model → Model Registry)
  - Confusion matrix PNG  (log_artifact)
  - Data source tag       (set_tags)

Usage:
    python src/train.py                  # train on reference data
    python src/train.py --combine        # combine reference + production (retraining)
    python src/train.py --data path.csv  # use a custom CSV
"""

import argparse
import json
import os
import pickle
import time
import warnings
from datetime import datetime, timezone
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import yaml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FEATURE_COLS = CFG["features"]["names"]
TARGET_COL = CFG["features"]["target"]
MODEL_CFG = CFG["model"]
PATHS = CFG["paths"]
MLFLOW_CFG = CFG["mlflow"]
MLFLOW_CFG["tracking_uri"] = os.getenv(
    "MLFLOW_TRACKING_URI", MLFLOW_CFG["tracking_uri"]
)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


# ── JSON-safe helper ──────────────────────────────────────────────────────────
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
# Data loading
# ══════════════════════════════════════════════════════════════════════════════


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python src/data_generator.py")
    df = pd.read_csv(path)
    print(f"Loaded {path}  ({len(df)} rows)")
    return df


def combine_datasets() -> pd.DataFrame:
    ref = pd.read_csv(PATHS["reference_data"])
    prod = pd.read_csv(PATHS["production_data"])
    combined = pd.concat([ref, prod], ignore_index=True)
    print(f"Combined: {len(ref)} ref + {len(prod)} prod = {len(combined)} total")
    return combined


def split_data(df: pd.DataFrame) -> Tuple:
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return train_test_split(
        X,
        y,
        test_size=MODEL_CFG["test_size"],
        random_state=MODEL_CFG["random_state"],
        stratify=y,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════


def train_model(X_train, y_train) -> RandomForestClassifier:
    params = MODEL_CFG["hyperparameters"]
    model = RandomForestClassifier(**params)
    print(f"\nTraining RandomForestClassifier  params={params}")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.2f}s")
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV: {cv.mean():.4f} ± {cv.std():.4f}")
    return model, float(cv.mean()), float(cv.std())


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_model(model, X_val, y_val) -> Dict:
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": float(round(accuracy_score(y_val, y_pred), 4)),
        "f1_macro": float(round(f1_score(y_val, y_pred, average="macro"), 4)),
        "f1_weighted": float(round(f1_score(y_val, y_pred, average="weighted"), 4)),
        "precision_macro": float(
            round(precision_score(y_val, y_pred, average="macro", zero_division=0), 4)
        ),
        "recall_macro": float(
            round(recall_score(y_val, y_pred, average="macro", zero_division=0), 4)
        ),
    }
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_val == i
        if mask.sum() > 0:
            metrics[f"precision_{cls}"] = float(
                round(precision_score(y_val == i, y_pred == i, zero_division=0), 4)
            )
            metrics[f"recall_{cls}"] = float(
                round(recall_score(y_val == i, y_pred == i, zero_division=0), 4)
            )
            metrics[f"f1_{cls}"] = float(
                round(f1_score(y_val == i, y_pred == i, zero_division=0), 4)
            )

    for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
        metrics[f"importance_{feat}"] = float(round(imp, 4))

    metrics["confusion_matrix"] = confusion_matrix(y_val, y_pred).tolist()

    print(f"\nValidation — accuracy={metrics['accuracy']} | f1={metrics['f1_macro']}")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))
    return metrics


def plot_confusion_matrix(cm: list, path: str) -> str:
    """Save a confusion matrix PNG and return the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center",
                color="white" if cm[i][j] > max(max(r) for r in cm) / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Model persistence (local pkl — MLflow is the primary artefact store)
# ══════════════════════════════════════════════════════════════════════════════


def save_local_candidate(model, metrics: Dict) -> None:
    """Save candidate pkl locally so evaluate.py / pipeline.py can load it."""
    os.makedirs(os.path.dirname(PATHS["candidate_model"]), exist_ok=True)
    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(PATHS["candidate_model"], "wb") as f:
        pickle.dump(bundle, f)
    print(f"Candidate saved → {PATHS['candidate_model']}")


def log_metrics_json(metrics: Dict, source: str) -> None:
    os.makedirs("logs", exist_ok=True)
    path = PATHS["metrics_log"]
    history = json.load(open(path)) if os.path.exists(path) else []
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_data": source,
            "metrics": {
                k: v for k, v in metrics.items() if k not in ("confusion_matrix",)
            },
        }
    )
    json.dump(_safe(history), open(path, "w"), indent=2)
    print(f"Metrics logged → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main training run — MLflow tracking is the centrepiece
# ══════════════════════════════════════════════════════════════════════════════


def run_training(data_path: str = None, combine: bool = False) -> Dict:
    print("=" * 55)
    print("  Iris CT — Training Pipeline  (MLflow tracking ON)")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 55)

    # ── Set up MLflow experiment ──────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    # Load data
    if combine:
        df, source_label = combine_datasets(), "reference+production"
    elif data_path:
        df, source_label = load_data(data_path), data_path
    else:
        df, source_label = load_data(PATHS["reference_data"]), "reference"

    X_train, X_val, y_train, y_val = split_data(df)

    # ── MLflow run ────────────────────────────────────────────────────────────
    with mlflow.start_run(
        nested=bool(mlflow.active_run()),
        run_name=f"train_{source_label.replace('/','_')}",
    ) as run:
        run_id = run.info.run_id
        print(f"\nMLflow run_id: {run_id}")
        print("MLflow UI:     mlflow ui  (then open http://localhost:5000)")

        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "source_data": source_label,
                "model_type": MODEL_CFG["type"],
                "triggered_by": "manual" if not combine else "pipeline",
                "train_samples": str(len(X_train)),
                "val_samples": str(len(X_val)),
                "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null")
                .read()
                .strip()
                or "N/A",
            }
        )

        # ── Log hyperparameters ───────────────────────────────────────────────
        mlflow.log_params(MODEL_CFG["hyperparameters"])
        mlflow.log_params(
            {
                "test_size": MODEL_CFG["test_size"],
                "random_state": MODEL_CFG["random_state"],
                "n_features": len(FEATURE_COLS),
                "n_classes": 3,
                "source_data": source_label,
            }
        )

        # ── Train ─────────────────────────────────────────────────────────────
        model, cv_mean, cv_std = train_model(X_train, y_train)
        mlflow.log_metrics({"cv_accuracy_mean": cv_mean, "cv_accuracy_std": cv_std})

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics = evaluate_model(model, X_val, y_val)
        cm = metrics.pop("confusion_matrix")

        # Log all scalar metrics to MLflow
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
        mlflow.log_metrics(scalar_metrics)

        # ── Log confusion matrix as artefact ──────────────────────────────────
        cm_path = "reports/confusion_matrix.png"
        plot_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        print("Confusion matrix → MLflow artefact (plots/confusion_matrix.png)")

        # ── Log model to MLflow Model Registry ────────────────────────────────
        # mlflow.sklearn.log_model logs the model AND creates a model signature
        # (input/output schema) automatically from sample data.
        input_example = pd.DataFrame(X_val[:3], columns=FEATURE_COLS)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_CFG["registered_model"],
            input_example=input_example,
        )
        print(f"Model logged to Registry as '{MLFLOW_CFG['registered_model']}'")
        print(f"Model URI: {model_info.model_uri}")

        # Store run_id in metrics for downstream use
        metrics["confusion_matrix"] = cm
        metrics["mlflow_run_id"] = run_id
        metrics["mlflow_model_uri"] = model_info.model_uri

    # ── Local artefacts (for evaluate.py / pipeline.py) ───────────────────────
    save_local_candidate(model, metrics)
    log_metrics_json(metrics, source_label)

    print("\nTraining complete.")
    print(f"  MLflow experiment: {MLFLOW_CFG['experiment_name']}")
    print(f"  Run ID:            {run_id}")
    print(
        f"  View UI:           mlflow ui --backend-store-uri {MLFLOW_CFG['tracking_uri']}"
    )
    print("Next step: python src/evaluate.py --promote")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    run_training(data_path=args.data, combine=args.combine)
