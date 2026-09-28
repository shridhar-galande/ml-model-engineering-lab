"""Train two versions of a wine classifier (toy dataset) for the canary lab."""
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Hold out the same stratified test set for both model versions.
data = load_wine()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

# Ensure the artifact directory exists before saving either model.
Path("models").mkdir(parents=True, exist_ok=True)

# Train a smaller stable model and a larger candidate on the same split.
for version, n_trees in [("v1", 30), ("v2", 200)]:
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42).fit(X_tr, y_tr)
    # Store class names beside the estimator for use in API responses.
    joblib.dump({"model": model, "classes": list(data.target_names)}, f"models/model_{version}.joblib")
    print(f"{version}: trees={n_trees}  test accuracy={model.score(X_te, y_te):.3f}")
