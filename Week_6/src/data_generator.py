"""
data_generator.py
=================
Creates all datasets needed for the continuous training pipeline:
  - data/reference.csv    → baseline training distribution (clean Iris)
  - data/production.csv   → simulated production data (with injected drift)
  - data/golden.csv       → curated golden dataset with known correct labels

Run once at project setup:
    python src/data_generator.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

# ── Column names ──────────────────────────────────────────────────────────────
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "species"


def load_iris_as_dataframe() -> pd.DataFrame:
    """Load the Iris dataset as a named DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_COLS)
    df[TARGET_COL] = iris.target
    return df


def generate_reference_data(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Reference data = the distribution the model was trained on.
    Uses 80% of clean Iris data.
    """
    train_df, _ = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COL]
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_df.reset_index(drop=True).to_csv(save_path, index=False)
    print(f"Reference data saved → {save_path}  ({len(train_df)} rows)")
    return train_df


def generate_production_data(
    df: pd.DataFrame, save_path: str, drift_type: str = "covariate"
) -> pd.DataFrame:
    """
    Production data = what arrives after deployment.
    Injects realistic drift to simulate a changing real-world distribution.

    drift_type options:
      "none"      → clean data (no drift)
      "covariate" → feature distribution shifts (input drift)
      "prior"     → class balance shifts (label drift)
      "combined"  → both covariate + prior drift
    """
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COL]
    )
    prod_df = test_df.copy().reset_index(drop=True)

    if drift_type == "covariate":
        # Simulate a real-world covariate shift:
        # sepal_length increases (e.g. different growing season)
        # petal_width has more variance (e.g. new subspecies mix)
        print("Injecting covariate drift: sepal_length +0.4, petal_width +noise")
        prod_df["sepal_length"] = prod_df["sepal_length"] + 0.4
        prod_df["petal_width"] = prod_df["petal_width"] + np.random.normal(
            0, 0.3, len(prod_df)
        )
        prod_df["petal_width"] = prod_df["petal_width"].clip(lower=0.0)

    elif drift_type == "prior":
        # Class balance shifts: oversample virginica (class 2)
        print("Injecting prior drift: oversample class 2 (virginica)")
        virginica = prod_df[prod_df[TARGET_COL] == 2]
        prod_df = pd.concat([prod_df, virginica, virginica], ignore_index=True)

    elif drift_type == "combined":
        print("Injecting combined covariate + prior drift")
        prod_df["sepal_length"] = prod_df["sepal_length"] + 0.4
        prod_df["petal_width"] = prod_df["petal_width"] + np.random.normal(
            0, 0.3, len(prod_df)
        )
        virginica = prod_df[prod_df[TARGET_COL] == 2]
        prod_df = pd.concat([prod_df, virginica], ignore_index=True)

    else:  # "none"
        print("No drift injected (clean production data)")

    prod_df = prod_df.sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prod_df.to_csv(save_path, index=False)
    print(
        f"Production data saved → {save_path}  ({len(prod_df)} rows, drift='{drift_type}')"
    )
    return prod_df


def generate_golden_dataset(
    df: pd.DataFrame, save_path: str, n_per_class: int = 5
) -> pd.DataFrame:
    """
    Golden dataset = curated set of representative examples with KNOWN correct labels.
    Used as a hard evaluation gate before any new model is promoted.

    Selects examples closest to each class centroid (most "typical" examples).
    These are the predictions that must NEVER regress.
    """
    golden_rows = []
    for cls in sorted(df[TARGET_COL].unique()):
        cls_df = df[df[TARGET_COL] == cls].copy()
        centroid = cls_df[FEATURE_COLS].mean().values
        distances = np.linalg.norm(cls_df[FEATURE_COLS].values - centroid, axis=1)
        cls_df["_dist"] = distances
        # Take the n closest to centroid (most representative)
        top_n = cls_df.nsmallest(n_per_class, "_dist").drop(columns=["_dist"])
        golden_rows.append(top_n)

    golden_df = pd.concat(golden_rows, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    golden_df.to_csv(save_path, index=False)
    print(
        f"Golden dataset saved  → {save_path}  ({len(golden_df)} rows, {n_per_class}/class)"
    )
    return golden_df


if __name__ == "__main__":
    print("=" * 55)
    print("  Iris CT — Dataset Generator")
    print("=" * 55)

    df = load_iris_as_dataframe()
    print(
        f"\nFull Iris dataset: {len(df)} rows, {df[TARGET_COL].value_counts().to_dict()}"
    )

    # Create all three datasets
    ref_df = generate_reference_data(df, "data/reference.csv")
    prod_df = generate_production_data(
        df, "data/production.csv", drift_type="covariate"
    )
    golden_df = generate_golden_dataset(df, "data/golden.csv", n_per_class=5)

    print("\nSummary:")
    print(f"  Reference:   {len(ref_df)} rows")
    print(f"  Production:  {len(prod_df)} rows (with covariate drift)")
    print(f"  Golden:      {len(golden_df)} rows (5 per class)")
    print("\nAll datasets generated. Run train.py next.")
