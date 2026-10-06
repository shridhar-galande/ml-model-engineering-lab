# MLOps Foundations — Hands-On Demo
### Iris Classification 

---

## What You'll Build

A **fully reproducible ML pipeline** that trains a classifier on the Iris dataset,
validates data quality, tracks experiments, and "deploys" a model — all triggered
from the command line like a real CI/CD system.

```
your_code_push
      │
      ▼
[1] data_check.py   ← Schema + quality gates (does data look right?)
      │
      ▼
[2] train.py        ← Train model, log everything to MLflow
      │
      ▼
[3] evaluate.py     ← Compare new model vs saved champion
      │
      ▼
[4] register.py     ← Save model artifact + metadata (reproducibility)
      │
      ▼
[5] pipeline.py     ← Run all steps in one command (your "CI/CD trigger")
```

---

## Setup 

```bash
# 1. Install dependencies
pip install scikit-learn mlflow pandas

# 2. Go into the project folder
cd project

# 3. Check everything works
python src/data_check.py
```

You should see: `✅ All data quality checks passed!`

---

## Session Outline 

| Step | File | Concept |
|------|------|---------|
| Look at the data | `data/` | What are we working with? |
| Data quality gate | `src/data_check.py` | Schema enforcement |
| Training + tracking | `src/train.py` | Artifacts + MLflow |
| Evaluation gate | `src/evaluate.py` | Model validation |
| Run the pipeline | `src/pipeline.py` | CI/CD trigger |
| View MLflow UI | `mlflow ui` | Lineage + reproducibility |

---

## Run Each Step Individually

```bash
# Step 1 — Data quality check
python src/data_check.py

# Step 2 — Train and log experiment
python src/train.py

# Step 3 — Evaluate against champion
python src/evaluate.py

# Step 4 — Register the winning model
python src/register.py

# Step 5 — OR run everything at once (the "pipeline")
python src/pipeline.py
```

---

## View Your Experiments

```bash
mlflow ui
```
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://localhost:5000 in your browser.
You'll see every run with its parameters, metrics, and saved model.

**This is lineage** — you can click any past run and replay it exactly.
- **Experiments tab** — every run listed with params + metrics
- **Run detail** — accuracy, F1, seed, dataset, pipeline version
- **Artifacts tab** — model files, scaler, confusion matrix plot
- **Models tab** — `iris-classifier` v1 with `champion` alias

For tracking, model logging, registry, serving, and troubleshooting commands,
see the [MLflow Cheat Sheet](./MLFLOW_CHEATSHEET.md).

---

## Project Structure

```
mlops_demo/
├── README.md               ← You are here
├── data/
│   └── iris_with_issues.csv   ← Intentionally messy data (for the demo)
├── src/
│   ├── data_check.py       ← Step 1: Data quality gates
│   ├── train.py            ← Step 2: Train + log to MLflow
│   ├── evaluate.py         ← Step 3: Compare vs champion
│   ├── register.py         ← Step 4: Save model artifact
│   └── pipeline.py         ← Step 5: Run all steps
├── tests/
│   └── test_data_check.py  ← Simple unit test
└── models/
    └── (created automatically when you run register.py)
```

---

## Key Concepts This Demo Covers

| Concept | Where You See It |
|---------|-----------------|
| **Schema enforcement** | `data_check.py` — rejects wrong column names/types |
| **Data quality gate** | `data_check.py` — fails if nulls or out-of-range values |
| **Artifact versioning** | `train.py` — MLflow logs model + params + metrics |
| **Model validation** | `evaluate.py` — only accepts model if accuracy ≥ threshold |
| **Champion/challenger** | `evaluate.py` — compares new model vs saved best |
| **Experiment lineage** | MLflow UI — every run is reproducible |
| **CI/CD trigger** | `pipeline.py` — one command runs the whole thing |
| **Reproducibility** | Fixed random seeds + logged library versions |

---

## Teaching Tips

- **Run `data_check.py` with the bad data first** — show it failing, then explain why gates matter.
- **Open MLflow UI while `train.py` runs** — live updates make it tangible.
- **Change `n_estimators` in `train.py`** and run again — show two runs side-by-side in MLflow.
- **Set accuracy threshold to 0.99 in `evaluate.py`** — show the gate blocking a "bad" model.

## Common MLOps Workflow Commands

### 1. Start MLflow Tracking Server

```bash
mlflow server
```

---

### 2. Train Machine Learning Model

```bash
python train.py
```

---

### 3. View MLflow Experiments

```bash
mlflow ui
```

---

### 4. Register Model in MLflow Model Registry

```bash
mlflow models register
```

---

### 5. Serve Model Locally

```bash
mlflow models serve
```

---

### 6. Build Docker Image for Deployment

```bash
docker build .
```

---

## Pull, Run, and Test the Model Image Locally

The GitHub Actions workflow builds the MLflow serving image after the data and
model quality checks pass, then publishes it to GitHub Container Registry
(GHCR) for pushes to `main`. Pull requests run CI but do not publish images.
The image is tagged `latest` and with the commit SHA. First, make sure Docker
Desktop is installed and running, and that the GitHub Actions image-publish job
has completed successfully.

Open PowerShell and set the image name:

```powershell
$image = "ghcr.io/galandeshridhar0-a11y/mlops-ci-pipeline:latest"
```

If the GHCR package is public, pull the image directly:

```powershell
docker pull $image
```

If the package is private, first authenticate with a GitHub personal access
token that has `read:packages` permission, then pull it:

```powershell
$env:CR_PAT = "<your-read-packages-token>"
$env:CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
Remove-Item Env:CR_PAT
docker pull $image
```

Start the model server in this PowerShell terminal:

```powershell
docker run --rm --name iris-mlflow -p 5001:5000 ghcr.io/shridhar-galande/mlops-ci-pipeline:latest
```

Wait until the container reports that the MLflow server is listening. Keep
this terminal open. In a **second** PowerShell terminal, send one Iris sample
to the model:

```powershell
$body = '{"dataframe_split":{"columns":["sepal_length","sepal_width","petal_length","petal_width"],"data":[[5.1,3.5,1.4,0.2]]}}'

$result = Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:5001/invocations `
    -ContentType "application/json" `
    -Body $body

$result | ConvertTo-Json -Depth 5
```

### Test with `curl` (macOS, Linux, or Git Bash)

With the model server still running, open another terminal and run:

```bash
curl --request POST \
  --url http://127.0.0.1:5001/invocations \
  --header "Content-Type: application/json" \
  --data '{
    "dataframe_split": {
      "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
      "data": [[5.1, 3.5, 1.4, 0.2]]
    }
  }'
```

The response should look like this:

```json
{"predictions":["setosa"]}
```

Stop the local server with **Ctrl+C** in the first terminal. The `--rm` option
removes the stopped container; the pulled image remains available locally.

No GitHub deployment environment or hosted server is required for this local
run. A GitHub Environment is only needed later if you want deployment approvals,
environment-specific secrets, or an automated deployment target.
