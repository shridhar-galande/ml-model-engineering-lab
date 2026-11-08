# Feast Feature Store Demos

This repository contains Jupyter notebooks demonstrating how to build and use local [Feast](https://feast.dev/) feature stores with scikit-learn. The examples cover feature engineering, historical and online feature retrieval, model training, and (in one notebook) experiment tracking with MLflow.

## Notebooks

- `Feast_Iris_Beginner_Walkthrough.ipynb` — step-by-step Iris classification example, from preparing feature data through Feast retrieval and inference.
- `Feast_MLflow_Iris_Feature_Store.ipynb` — Iris feature-store workflow with MLflow experiment tracking.
- `Feast_Regression_Diabetes_Feature_Store.ipynb` — regression workflow using the diabetes dataset.
- `Feast_Feature_Store_Exercise.ipynb` — hands-on exercise for practicing Feast concepts.
- `Fraud Detection Using Feast.ipynb` — fraud-detection example using Feast.

## Setup

From the repository root, create and activate a virtual environment, then install the common dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open a notebook in VS Code or Jupyter and run its cells in order. Some notebooks also install their specific dependencies in an initial notebook cell.

## Feast feature store

Feast (Feature Store) is an open-source tool for organizing and serving the input data—called **features**—used by machine-learning models. A feature store helps teams define features once and retrieve consistent values for both model training and predictions. It supports historical retrieval from an offline source and fast, low-latency retrieval from an online store.

The beginner walkthrough's local Feast feature store is set up in `feature_repo/`. Its `feature_store.yaml` configures Feast's local provider, a registry at `feature_repo/data/registry.db`, and a SQLite online store at `feature_repo/data/online_store.db`. The Iris feature data is written to Parquet and used as the offline source.

The notebook then defines an Entity and Feature View, registers them with `FeatureStore.apply(...)` (the Python equivalent of `feast apply`), and materializes feature values into the online store. It demonstrates both:

- **Historical retrieval** with `get_historical_features(...)` for point-in-time-correct training data.
- **Online retrieval** with `get_online_features(...)` for low-latency inference lookups by entity ID.

Run the walkthrough from its first cell so it creates the project configuration and data before registering or querying the store. Feast's registry, SQLite database, and generated Parquet files are local runtime data and are ignored by Git.

## MLflow tracking

MLflow is an open-source platform for managing the machine-learning lifecycle. It can track experiments, parameters, metrics, and model artifacts, making it easier to compare training runs and keep a record of how a model was produced.

The MLflow Iris notebook uses a local SQLite tracking database. To browse tracked runs, start the MLflow server from the repository root:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Then open <http://127.0.0.1:5000>.

## Generated files

Running the notebooks may create local datasets (`data.csv`, `data.parquet`), Feast runtime files (`feature_repo/feature_store.yaml` and `feature_repo/data/`), MLflow databases and artifacts, and notebook checkpoint files. These outputs are excluded by `.gitignore`; the notebooks and `requirements.txt` are the project sources.
