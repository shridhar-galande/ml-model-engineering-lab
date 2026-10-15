# MLOps Foundations — Hands-On Demo
### Iris Classification 

---

## Model Monitoring & Observability Demo

This repository has a CI-to-image-release pipeline and a manual local run
step. GitHub Actions tests the code and notebooks, trains and packages the Iris
model, runs the monitored API with Prometheus and Grafana, then publishes a
versioned Docker image to GHCR from `main`. It does **not** deploy or start
anything on your machine. You pull the tested image and run the stack yourself.

The monitoring notebooks use the Iris dataset and controlled, synthetic
feature shifts. They complement the Iris model served by the Docker image;
they are educational simulations, not live production data.

### Recommended demo flow

1. Follow [GitHub CI and image release](#github-ci-and-image-release). Push
   your changes and merge them to `main`; wait for the `CI` workflow to finish
   successfully.
2. Follow [Pull and run the released image](#pull-and-run-the-released-image)
   to pull that workflow's image tag and start the API, Prometheus, and Grafana
   on your computer.
3. Open [model_monitoring_observability.ipynb](./notebooks/model_monitoring_observability.ipynb)
   and run from top to bottom for the Iris drift case study. The shifted
   feature windows are simulated; live API metrics are available from the
   locally running monitoring stack.
4. Use [prometheus_integration_guide.ipynb](./notebooks/prometheus_integration_guide.ipynb)
   as a guide to the Iris API's Prometheus metrics and Grafana dashboard.

### Topic-to-tool map

| Topic | Demo coverage |
|-------|---------------|
| Statistical, window-based, and domain-based drift | PSI, KS, and Chi-Squared; sliding windows show when feature changes begin, and species segments expose localized drift |
| Model performance | Accuracy, weighted precision, recall, F1, and per-species results on labeled Iris holdouts |
| System performance | The API measures real request latency and success/error rates; host CPU/memory need an exporter in deployment |
| Observability | Prometheus metrics, alert rules, and dashboards run in the local stack; the notebook illustrates trace spans, which need OpenTelemetry instrumentation for production tracing |
| Prometheus and Grafana | The Docker image exposes real request, latency, and PSI metrics at `/metrics`; `docker-compose.yml` starts Prometheus and a provisioned Grafana dashboard |
| EvidentlyAI | Automated data-quality and drift report in the monitoring notebook |
| Fiddler and Arize | Examples of managed model-observability platforms to compare in discussion; they are not required to run this demo |

All Python and notebook dependencies are in [requirements.txt](./requirements.txt).
Run `python -m pip install -r requirements.txt` from the repository root before
using the notebooks.

### GitHub CI and image release

The [CI workflow](./.github/workflows/ci.yml) has two jobs:

1. **CI** runs unit and notebook-format tests, the data-quality gate, the
   train/evaluate/register pipeline, builds the monitored image, then starts
   the model + Prometheus + Grafana and checks predictions, drift metrics,
   scraping, alert rules, and the dashboard. It saves this tested image for
   the release job.
2. **Image release** runs only after CI succeeds on a push or manual dispatch
   from `main`. It publishes that same tested image as
   `ghcr.io/<owner>/<repository>:latest` and
   `ghcr.io/<owner>/<repository>:sha-<commit>`.

The Actions run summary shows the tested image's build tag, image ID, and size.
For successful runs from `main`, it also shows the GHCR image tags and a
`docker pull` command.

Pull requests run CI but do not publish images. Nothing is deployed to a
server or started on your local machine by GitHub Actions. You do not need to
add repository secrets for publishing; the workflow uses `GITHUB_TOKEN`.

### Pull and run the released image

After your push to `main`, open **GitHub → Actions → CI**, select the newest
successful run, and copy its commit SHA. Make sure Docker Desktop is running.
In PowerShell, replace `OWNER/REPOSITORY` with your GitHub owner and repository
(the GHCR image name is lowercase), then use the matching short commit SHA:

```powershell
$image = "ghcr.io/OWNER/REPOSITORY:sha-<commit>"
```

If the GHCR package is public, skip login. If it is private, create a GitHub
token with `read:packages` and log in (do not save the token in the repository):

```powershell
$env:CR_PAT = "<your-read-packages-token>"
$env:CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
Remove-Item Env:CR_PAT
```

Pull the exact tested image and start the stack:

```powershell
docker pull $image
$env:MODEL_IMAGE = $image
docker compose up -d --wait
```

To run the latest published image directly:

```powershell
$env:MODEL_IMAGE = "ghcr.io/shridhar-galande/mlops-monitoring-observability:latest"
docker compose up -d --wait
```

Verify the model health and make a prediction:

```powershell
Invoke-RestMethod http://localhost:8000/healthz

$body = '{"dataframe_split":{"columns":["sepal_length","sepal_width","petal_length","petal_width"],"data":[[5.1,3.5,1.4,0.2]]}}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/invocations `
  -ContentType "application/json" -Body $body
```

Send 100 shifted rows to fill the monitoring window and trigger feature PSI:

```powershell
$shiftedRows = 1..100 | ForEach-Object { ,@(7.5,4.8,6.8,2.5) }
$driftBody = @{
  dataframe_split = @{
    columns = @("sepal_length","sepal_width","petal_length","petal_width")
    data = $shiftedRows
  }
} | ConvertTo-Json -Depth 5 -Compress
Invoke-RestMethod -Method Post -Uri http://localhost:8000/invocations `
  -ContentType "application/json" -Body $driftBody

Invoke-WebRequest http://localhost:8000/metrics | Select-Object -ExpandProperty Content
```

Open Prometheus at <http://localhost:9090> or Grafana at
<http://localhost:3000> (`admin` / `admin`). The provisioned **Model Monitoring**
dashboard displays request rate, error ratio, p95 latency, and PSI by feature.
Prometheus evaluates the rules in `alert_rules.yml`. The API exposes request
outcomes, latency, predictions, and feature PSI; host CPU/memory require an
exporter in a real deployment. The companion notebooks demonstrate Iris
feature drift and compare model quality on a controlled shifted holdout.

Stop the stack when finished:

```powershell
docker compose down
```

---

## What You'll Build

A **reproducible Iris ML pipeline** that validates data, trains a classifier,
tracks experiments, evaluates the model, packages a monitored API image, and
releases that image to GHCR after CI passes. Running the released image locally
is a manual step described above.

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
[5] pipeline.py     ← Run the model steps in one command
```

GitHub Actions runs this model pipeline, tests the monitoring stack, and
publishes the tested image from `main`; see
[GitHub CI and image release](#github-ci-and-image-release).

---

## Setup 

Run from the repository root in PowerShell. For the notebooks, open the
repository folder in VS Code, select the `.venv` Python interpreter as the
notebook kernel, and open a notebook under `notebooks/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python src/data_check.py --fix
python src/pipeline.py --n-estimators 10 --max-depth 2
```

---

## Session Outline 

| Step | File | Concept |
|------|------|---------|
| Look at the data | `data/` | What are we working with? |
| Data quality gate | `src/data_check.py` | Schema enforcement |
| Training + tracking | `src/train.py` | Artifacts + MLflow |
| Evaluation gate | `src/evaluate.py` | Model validation |
| Run the pipeline | `src/pipeline.py` | Data check, training, evaluation, registration |
| CI and image release | `.github/workflows/ci.yml` | Test image and publish to GHCR from `main` |
| View MLflow UI | `mlflow ui` | Lineage + reproducibility |

---

## Run Each Step Individually

```bash
# Step 1 — See the intentionally bad input fail
python src/data_check.py

# Clean the demo data before training
python src/data_check.py --fix

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
.
├── .github/workflows/ci.yml       # Tests, trains, container-smokes, publishes on main
├── notebooks/                     # Monitoring lessons, Prometheus guide and slides
├── src/                           # Iris checks, training, evaluation and monitored API
├── tests/                         # API, monitoring stack, notebook and data tests
├── grafana/                       # Provisioned datasource and dashboard
├── Dockerfile                     # Model API image built by CI
├── docker-compose.yml             # Model API + Prometheus + Grafana
├── prometheus.yml                 # Scrape configuration
├── alert_rules.yml                # Request, latency and feature-drift alerts
└── requirements.txt               # Application, test and notebook dependencies
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
| **Training pipeline** | `pipeline.py` — runs data checks, training, evaluation, and registration |
| **CI and image release** | `.github/workflows/ci.yml` — tests the image and publishes it to GHCR from `main` |
| **Reproducibility** | Fixed random seeds + logged library versions |

---

## Teaching Tips

- **Run `data_check.py` with the bad data first** — show it failing, then explain why gates matter.
- **Open MLflow UI while `train.py` runs** — live updates make it tangible.
- **Change `n_estimators` in `train.py`** and run again — show two runs side-by-side in MLflow.
- **Set accuracy threshold to 0.99 in `evaluate.py`** — show the gate blocking a "bad" model.

For the full local monitoring run, use
[Pull and run the released image](#pull-and-run-the-released-image). The
companion [MLflow cheat sheet](./MLFLOW_CHEATSHEET.md) has experiment-tracking
commands.

---
