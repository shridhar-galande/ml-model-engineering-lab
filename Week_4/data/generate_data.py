"""
generate_data.py — creates iris_with_issues.csv for the demo.

The "issues" are intentional so students can see the data_check.py gate
catch them live:
  - 3 rows with missing sepal_length values
  - 2 rows with out-of-range petal_width (-1)
  - 1 row with a typo in the species column

Run once:  python data/generate_data.py
"""

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()

# Rename columns to the names our pipeline expects
df.columns = [
    "sepal_length", "sepal_width",
    "petal_length", "petal_width", "species"
]

# Map numeric target to string labels (more readable)
label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
df["species"] = df["species"].map(label_map)

# ── Inject intentional issues ──────────────────────────────
# Issue 1: 3 missing sepal_length values
df.loc[[10, 55, 100], "sepal_length"] = None

# Issue 2: 2 out-of-range petal_width values (must be > 0)
df.loc[[20, 75], "petal_width"] = -1.0

# Issue 3: 1 typo in species label
df.loc[130, "species"] = "virginicaa"   # extra 'a'

# Save
df.to_csv("data/iris_with_issues.csv", index=False)
print("✅  Saved data/iris_with_issues.csv")
print(f"    Shape: {df.shape}")
print(f"    Nulls: {df.isnull().sum().to_dict()}")
