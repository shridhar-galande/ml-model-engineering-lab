# MLflow Cheat Sheet

A practical quick reference for tracking experiments, comparing runs, storing
models, and using the MLflow UI. Commands use PowerShell syntax and work from
the project root unless noted.

> **Version note:** This project allows `mlflow>=2.10`. MLflow 3.x may print
> deprecation warnings for older API arguments that remain necessary for
> compatibility with MLflow 2.x. Check `mlflow --version` and the installed
> command's `--help` output when options differ.

## 1. Install and verify

```powershell
python -m pip install mlflow
mlflow --version
```

If the project virtual environment is active, `python` and `mlflow` should
resolve from that environment. Otherwise, call the executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install mlflow
.\.venv\Scripts\mlflow.exe --version
```

## 2. Start tracking locally

From the project root, this project uses SQLite for the tracking database:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://127.0.0.1:5000>. To use another port:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Or run the tracking server (UI plus REST API):

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Keep the tracking server running in its terminal. Press **Ctrl+C** there to
stop it. If the port is already occupied, stop the existing server or choose a
different `--port`. Binding to `127.0.0.1` keeps the service local; do not
expose a tracking server publicly without configuring authentication, network
access, and allowed hosts/origins appropriately.

### Tracking URI

The tracking URI tells the MLflow client where to save and find experiments:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
```

The environment variable applies to the current PowerShell session. To clear
it:

```powershell
Remove-Item Env:MLFLOW_TRACKING_URI
```

Common tracking URI forms:

| URI | Use |
|---|---|
| `sqlite:///mlflow.db` | Local SQLite database relative to the working directory |
| `sqlite:///C:/path/to/mlflow.db` | SQLite database at an absolute Windows path; use forward slashes in the URI |
| `http://127.0.0.1:5000` | MLflow tracking server |
| `file:///C:/path/to/mlruns` | Local file store; legacy/simple local experiments |

Use the same tracking URI when training and when opening the UI; otherwise,
they may show different experiments. SQLite is the recommended local choice
for this project and avoids MLflow versions that no longer allow the legacy
file-store backend by default.

## 3. Run this project

From the repository root:

```powershell
python src/data_check.py
python src/train.py
python src/evaluate.py
python src/register.py
```

Run all pipeline stages together:

```powershell
python src/pipeline.py
```

Try another training configuration:

```powershell
python src/train.py --n-estimators 50 --max-depth 4
```

The training script sets the experiment name to `iris-classifier` and logs
parameters, metrics, environment versions, the model, and
`models/latest_metrics.json`. Its model artifact path is `model`, so a run's
model URI looks like:

```text
runs:/<run-id>/model
```

`src/register.py` in this demo writes a timestamped local pickle and model
card. That is a custom registration step; it does **not** create an MLflow
Model Registry entry.

## 4. Core tracking API

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classifier")

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_params({"max_depth": 4, "random_seed": 42})

    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metrics({"precision": 0.94, "recall": 0.96})

    mlflow.set_tag("dataset", "iris")
    mlflow.set_tags({"stage": "baseline", "owner": "ml-team"})

    mlflow.log_artifact("models/latest_metrics.json")
    mlflow.log_artifacts("reports", artifact_path="reports")

print(run.info.run_id)
```

### Run context helpers

```python
with mlflow.start_run(run_name="random-forest-baseline") as run:
    ...
```

Nested runs are useful for grouping a parent workflow with individual
experiments:

```python
with mlflow.start_run(run_name="parameter-search"):
    for depth in (2, 4, 6):
        with mlflow.start_run(nested=True, run_name=f"depth-{depth}"):
            mlflow.log_param("max_depth", depth)
```

Mark an experiment as active before creating a run:

```python
mlflow.set_experiment("iris-classifier")
```

MLflow creates the experiment if it does not exist.

## 5. Scikit-learn models

Log a fitted model during an active run:

```python
import mlflow.sklearn

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, artifact_path="model")
```

Load it later using the run ID:

```python
model = mlflow.sklearn.load_model("runs:/<run-id>/model")
predictions = model.predict(X_test)
```

For MLflow versions that support the newer naming argument, `name="model"` is
preferred over the deprecated `artifact_path="model"`. Use the argument
supported by the installed MLflow version and project compatibility needs.

### Safe model serialization with `skops`

Some MLflow/scikit-learn configurations use `skops` to serialize models and
require an explicit allowlist for types that are needed when loading them.
For a Random Forest trained by your own code, the tree storage type reported
by the trust check can be allowed explicitly:

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
```

Only allow types you have reviewed and trust. Do not automatically trust every
type reported by a serialized model from an unknown source. Treat pickle-based
model artifacts as executable trusted data too.

### Autologging

Autologging can capture supported framework parameters, metrics, and models:

```python
import mlflow

mlflow.sklearn.autolog()
with mlflow.start_run():
    model.fit(X_train, y_train)
```

Use explicit logging instead when you need stable, carefully chosen metric
names or want to avoid duplicate automatically and manually logged values.

## 6. Find and compare runs

In the UI:

1. Select the `iris-classifier` experiment.
2. Compare runs by parameters and metrics.
3. Open a run to inspect its tags, source, parameters, metrics, and artifacts.
4. Download the model or artifacts from the run's artifact view.

Search runs programmatically:

```python
import mlflow

experiment = mlflow.get_experiment_by_name("iris-classifier")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy >= 0.90",
    order_by=["metrics.accuracy DESC"],
    max_results=20,
)
print(runs[["run_id", "metrics.accuracy", "params.n_estimators"]])
```

Common filter examples:

```text
metrics.accuracy > 0.90
params.n_estimators = '100'
tags.stage = 'baseline'
attributes.status = 'FINISHED'
```

Parameters and tags are stored as strings in run search filters. Use the UI or
`mlflow.search_runs()` to filter and sort; record consistent parameter, metric,
and tag names across runs to make comparisons easier.

## 7. MLflow Model Registry

The Model Registry versions named models and supports stages/aliases and
lineage. Use a database-backed tracking store (such as the project's SQLite
store) or a properly configured tracking server; a legacy file store is not
the recommended registry backend.

Register a model already logged to a run:

```python
import mlflow

model_uri = "runs:/<run-id>/model"
registered = mlflow.register_model(model_uri, "iris-classifier")
print(registered.version)
```

Load a specific registered version:

```python
model = mlflow.pyfunc.load_model("models:/iris-classifier/1")
```

In MLflow versions with Model Registry aliases, point an alias such as
`champion` at a model version and load it by alias:

```python
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("iris-classifier", "champion", "1")
model = mlflow.pyfunc.load_model("models:/iris-classifier@champion")
```

The Model Registry is different from this demo's `models/` directory. The
directory holds ordinary local files; a registered model is tracked by MLflow
with model versions and registry metadata.

## 8. Serve a logged model locally

Serve a model from a run:

```powershell
mlflow models serve --model-uri "runs:/<run-id>/model" --host 127.0.0.1 --port 5001 --env-manager local
```

Or serve a registered model:

```powershell
mlflow models serve --model-uri "models:/iris-classifier/1" --host 127.0.0.1 --port 5001 --env-manager local
```

Send a prediction request to the local scoring endpoint:

```powershell
$body = '{"dataframe_split":{"columns":["sepal_length","sepal_width","petal_length","petal_width"],"data":[[5.1,3.5,1.4,0.2]]}}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5001/invocations -ContentType "application/json" -Body $body
```

The model must accept the input format sent to the scoring endpoint. Serving
may need to install the model's recorded environment; `--env-manager local`
uses the current Python environment and is convenient for local tests.

## 9. Useful CLI commands

```powershell
mlflow --version
mlflow --help
mlflow ui --help
mlflow server --help
mlflow models --help
mlflow models serve --help
```

With a tracking server configured, common inspection commands include:

```powershell
mlflow experiments search
mlflow runs list --experiment-id 0
```

CLI subcommands and options can vary between MLflow versions; consult the
installed command's `--help` output.

## 10. Common problems

| Symptom | Likely cause | What to do |
|---|---|---|
| UI shows no runs | UI and training use different tracking stores | Start both with `sqlite:///mlflow.db`, or set `MLFLOW_TRACKING_URI` for both |
| `Address already in use` / socket bind traceback | Another process is listening on the port | Stop the earlier server with **Ctrl+C**, or choose another `--port` |
| `UntrustedTypesFoundException` for `sklearn.tree._tree.Tree` | `skops` did not allow the Random Forest tree type | Add that specific reviewed type to `skops_trusted_types` when logging |
| `artifact_path` deprecation warning | Newer MLflow prefers `name` | It is a warning; use `name` when the project's minimum MLflow version supports it |
| Warning that `pip` version could not be resolved | MLflow could not determine environment metadata | Usually non-fatal; verify that model logging completes and the artifact appears in the run |
| Job execution unsupported on Windows | MLflow's job backend is not available on Windows | Local experiment tracking and the UI can still work |
| Registry URI falls back to backend URI | No separate model-registry store was configured | Expected for a local SQLite setup |
| Unicode `UnicodeEncodeError` in PowerShell | Console encoding cannot display a Unicode character in output | Use ASCII console labels or configure the console for UTF-8 |

## 11. Reproducibility checklist

- Log the important training parameters, evaluation metrics, and dataset
  identity or version.
- Fix and log random seeds where the algorithm supports them.
- Log the model and any required plots, reports, or preprocessing artifacts.
- Record Python and key dependency versions.
- Use a consistent experiment name and tracking URI.
- Keep model input schema and feature ordering documented.
- Compare a candidate against explicit validation criteria before deployment.
- Do not load model files from untrusted sources.
