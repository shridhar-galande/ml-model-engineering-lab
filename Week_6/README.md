# Iris Continuous Training System

End-to-end MLOps pipeline on the Iris dataset — drift detection, automated
retraining, evaluation gates, and MLflow tracking throughout.

---

## Files at a Glance

| File | Purpose |
|------|---------|
| `config.yaml` | All thresholds, paths, SLOs, MLflow settings — edit here, not in code |
| `src/data_generator.py` | Creates `reference.csv`, `production.csv` (with drift), `golden.csv` |
| `src/train.py` | Trains RandomForest → logs to MLflow → saves candidate model |
| `src/evaluate.py` | Runs 5 gates → promotes candidate to MLflow Model Registry |
| `src/monitor.py` | Computes PSI / KS / JSD drift + EvidentlyAI report → MLflow run |
| `src/retrain_trigger.py` | Evaluates all 4 trigger types → logs decision to MLflow |
| `src/pipeline.py` | **One command to run everything** — monitor → trigger → train → evaluate → promote |
| `src/monitoring_service.py` | Runs the scheduled pipeline and exposes Prometheus metrics |
| `Dockerfile` / `compose.yaml` | Runs the GHCR image with local monitoring services |
| `monitoring/` | Prometheus configuration and provisioned Grafana dashboard |
| `requirements.txt` | All Python dependencies |
| `src/tests/test_pipeline.py` | 17 unit tests |
| `.github/workflows/continuous_training.yml` | GitHub Actions CI/CD |

---

## Execution Order

### Step 1 — First-time setup

```bash
pip install -r requirements.txt
python src/data_generator.py        # creates data/reference.csv, production.csv, golden.csv
```

### Step 2 — Initial training and deployment

```bash
python src/train.py                 # trains model, logs to MLflow, saves models/candidate_model.pkl
python src/evaluate.py --promote    # runs 5 gates, promotes to MLflow Registry as Production
```

At this point your model is live in the MLflow Model Registry (stage = Production).

### Step 3 — Check drift and monitoring

```bash
python src/monitor.py               # runs PSI / KS / JSD on all features, saves drift_report.html
```

Open `reports/drift_report.html` in a browser for the interactive EvidentlyAI report.

### Step 4 — Check whether retraining is needed

```bash
python src/retrain_trigger.py       # evaluates time / performance / drift / active-learning triggers
```

Prints `DECISION: RETRAIN 🚨` or `DECISION: NO ACTION ✅`.

### Step 5 — Retrain (if triggered)

```bash
python src/train.py --combine       # retrains on reference + production data combined
python src/evaluate.py --promote    # runs gates again, promotes new version if it passes
```

---

## One-Command Automated Pipeline

Instead of running steps 3–5 manually, run the full pipeline:

```bash
python src/pipeline.py              # monitor → trigger → retrain (if needed) → evaluate → promote
python src/pipeline.py --retrain    # skip trigger check, force a retrain
python src/pipeline.py --dry-run    # run everything but do NOT promote
python src/pipeline.py --monitor-only  # check drift and stop, no retrain
```

The GitHub Actions workflow runs on qualifying pushes to `main`, on its weekly
Sunday 00:00 UTC schedule, or when started manually from the Actions tab.

The Actions workflow keeps model building separate from inference monitoring:
it generates data, trains/evaluates and caches an initial Production model if
one does not yet exist, then runs drift/performance monitoring against that
model. Retraining runs only when the inference trigger fires (or you force it
from `workflow_dispatch`), followed by all five evaluation gates and promotion.
Model files and MLflow tracking/registry state are transferred between jobs and
cached for subsequent workflow runs.

---

## View Results in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Every training run, monitoring cycle, trigger decision, and pipeline execution
is logged here with full metrics, artefacts, and tags.

---

## What Gets Logged to MLflow

| Script | Experiment | Logged |
|--------|-----------|--------|
| `src/train.py` | `iris_ct` | Hyperparams, CV scores, accuracy/F1/recall per class, feature importances, confusion matrix PNG, model → Registry |
| `src/evaluate.py` | `iris_ct` (same run) | Gate pass/fail, Registry transition None→Staging→Production |
| `src/monitor.py` | `iris_ct_monitoring` | PSI/KS/JSD per feature, class shift, EvidentlyAI HTML report |
| `src/retrain_trigger.py` | `iris_ct_monitoring` | Per-trigger metrics, final retrain=true/false tag |
| `src/pipeline.py` | `iris_ct` | Parent run with all child runs nested inside, pipeline duration, outcome |

Load the Production model at any time:
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/iris_rf_model/Production")
```

---

## The Four Retraining Triggers

| # | Trigger | Fires when | Config |
|---|---------|-----------|--------|
| 1 | **Time-based** | Last retrain > 7 days ago | `triggers.time_based.interval_days` |
| 2 | **Performance** | Accuracy < 0.85 or F1 < 0.82 on labelled production data | `slo.accuracy.critical` |
| 3 | **Drift-based** | PSI > 0.20 or KS p-value < 0.05 on any feature | `drift.psi.trigger` |
| 4 | **Active learning** | > 20% of predictions below 70% confidence | `src/retrain_trigger.py` |

## The Five Evaluation Gates

All must pass before a candidate is promoted to Production in the Registry:

| # | Gate | What it checks |
|---|------|---------------|
| 1 | Baseline comparison | New model ≥ current Production model accuracy (−0.5% margin) |
| 2 | Minimum thresholds | Accuracy ≥ 0.85, F1 ≥ 0.82 — hard floors |
| 3 | Golden dataset | 100% correct on 15 curated representative examples |
| 4 | Per-class recall | Every class ≥ 0.80 recall — no hidden class failures |
| 5 | Distribution check | No missing values, class balance ≥ 5%, at least 30 samples |

## Run Tests

```bash
pytest src/tests/ -v    # 17 tests
```

---

## Local Docker Demo: Prometheus + Grafana

This demo runs three local containers: the MLOps pipeline/metrics exporter,
Prometheus, and Grafana. GitHub Actions builds the application image and
publishes it to GitHub Container Registry (GHCR) after the lint and unit-test
job passes. The build context, Dockerfile, and platform are configured in
the **Build & Publish Docker Image** job in
`.github/workflows/continuous_training.yml`.

### Pull the image built by GitHub Actions

After a workflow run completes, open **Actions → Iris — Continuous Training
Pipeline → the run**. The **Build & Publish Docker Image** job and the workflow
run summary include a link to the GHCR package and a copyable command similar
to `docker pull ghcr.io/shridhar-galande/mlops-continuous-training-retraining-automation:run-123456789`.
The unique `run-<run-id>` tag identifies that exact build. The commit tag
(`sha-<commit>`) is also pushed, and successful builds from `main` update
`:latest`.

For a private GHCR package, authenticate locally with a GitHub personal access
token that has `read:packages`:

```powershell
$env:GHCR_TOKEN = Read-Host "GitHub token with read:packages"
$env:GHCR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
Remove-Item Env:GHCR_TOKEN
```

The default Compose configuration pulls the published `:latest` image each
time it starts. To use the exact image from a particular Actions run instead,
pull that run's tag and set `MLOPS_IMAGE`:

```powershell
$env:MLOPS_IMAGE = "ghcr.io/shridhar-galande/mlops-continuous-training-retraining-automation:run-RUN_ID"
docker compose -f compose.yaml -f compose.image.yaml up -d
docker compose -f compose.yaml -f compose.image.yaml ps
```

Replace `run-RUN_ID` with the run tag shown in the Actions summary.
`compose.image.yaml` lets you select that run-specific image. Compose still
uses this repository for Prometheus and Grafana configuration and pulls their
pinned upstream images.

If you are running a fork, Actions publishes to that fork's GHCR package; use
the package/image URL shown in that workflow run rather than the example above.

### Start the demo (Windows PowerShell)

Run these commands from the repository root:

```powershell
docker compose pull
docker compose up -d
docker compose ps
docker compose logs -f mlops-demo
```

With new/empty Docker volumes, startup creates demo data and trains/evaluates
the initial model before it starts the monitoring cycles. Later starts reuse
the model, data, MLflow state, and history persisted in Docker volumes. Allow a
few minutes for the first-time setup. Leave the logs command running to watch
each pipeline stage; press `Ctrl+C` to stop following logs (the containers
continue running).

To build and run the application from source instead of pulling the GitHub
Actions image:

```powershell
docker pull prom/prometheus:v2.54.1
docker pull grafana/grafana:11.2.0
docker build -t iris-mlops-demo:local .
$env:MLOPS_IMAGE = "iris-mlops-demo:local"
docker compose -f compose.yaml -f compose.image.yaml up --pull never -d
```

### See the retraining trigger

The default Compose configuration enables `DEMO_MODE`. It writes a reproducible
labelled production batch of 120 rows with a deliberately shifted feature; this
exceeds the configured drift sample minimum and makes the drift trigger fire.
The service starts a pipeline run immediately and checks again every five
minutes. To see the trigger without waiting for the next cycle, follow the
first run's logs:

```powershell
docker compose logs -f mlops-demo
```

Wait until the logs say `Demo record endpoint available at
http://0.0.0.0:8001/demo/records` before posting records; the app trains its
baseline model before enabling the ingestion endpoint.

Look for the `MONITOR — Drift & Performance` section reporting a drift trigger,
then `TRIGGER — Should we retrain?` with `RETRAIN`, followed by the `TRAIN —
Retrain on combined data` and `EVALUATE — All 5 gates` sections. A trigger means
the pipeline attempted retraining; the evaluation gates can still reject the
candidate and prevent model promotion.

### Open Grafana and Prometheus

| Service | URL | Login |
|---------|-----|-------|
| Grafana dashboard | <http://localhost:3000/d/iris-mlops-demo/iris-mlops-retraining-demo> | `admin` / `admin` |
| Prometheus | <http://localhost:9090> | — |
| Application metrics | <http://localhost:8000/metrics> | — |
| Demo record ingestion API | <http://localhost:8001/demo/records> | — |

Grafana's provisioned **Iris MLOps Retraining Demo** dashboard displays feature
PSI, the monitoring drift signal, the retraining decision, individual trigger
reasons, model accuracy/F1, and pipeline duration. Metrics appear after the
first pipeline cycle completes. The trigger panels are latest-cycle gauges and
may return to `0` after a later cycle with no trigger; **Retraining decisions
(total)** is the cumulative counter. In Prometheus, open **Graph**, enter
`mlops_monitor_drift_trigger`, `mlops_retrain_trigger`, or
`mlops_retrain_decisions_total`, then select **Execute**. A value of `1` for
either trigger gauge means that the most recent completed cycle detected drift
or decided to retrain, respectively; the counter increases each time retraining
is triggered.

### Prometheus alert rules

Prometheus loads [monitoring/alert_rules.yml](./monitoring/alert_rules.yml)
on startup. After updating an already-running stack, apply the rules with
`docker compose restart prometheus`. To view current and pending alerts, open
<http://localhost:9090/alerts>. The rules cover a retraining decision, PSI above
the `0.20` drift threshold, pipeline failure, an unreachable metrics exporter,
and no successful pipeline run for 15 minutes. Rules evaluate after the
configured `for` duration; for example, a retraining alert should appear
shortly after the triggered run finishes.

These rules make alerts visible in Prometheus; they do not send email, Slack, or
push notifications. A notification receiver (typically Alertmanager) must be
configured separately if you need alerts delivered outside the Prometheus UI.

### Push these test samples to trigger retraining now

Post the three labelled samples below (one for each Iris class) to the local
demo API. The API repeats the samples to meet the configured minimum drift
sample count, applies a controlled `sepal_width` shift, saves them as production
data, and immediately runs the local monitoring/retraining pipeline. This
starts retraining in the running Compose app; it does not trigger a GitHub
Actions run. The request waits for training and evaluation to finish, so allow
several minutes on the first request. The endpoint is enabled only when
`DEMO_MODE=true` (the Compose default).

After starting Compose and waiting for the demo record endpoint to become
available, copy-paste this in a second Windows PowerShell window:

```powershell
$payload = @{
  records = @(
    @{ sepal_length = 5.1; sepal_width = 3.5; petal_length = 1.4; petal_width = 0.2; species = 0 },
    @{ sepal_length = 6.4; sepal_width = 3.2; petal_length = 4.5; petal_width = 1.5; species = 1 },
    @{ sepal_length = 6.3; sepal_width = 3.3; petal_length = 6.0; petal_width = 2.5; species = 2 }
  )
} | ConvertTo-Json -Depth 5

$result = Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8001/demo/records" `
  -ContentType "application/json" `
  -TimeoutSec 600 `
  -Body $payload

$result | Format-List
```

The response should report `records_received: 3`, at least 50
`production_rows`, and `monitor_trigger: True`, `retrain: True`, with `drift`
in `triggered_by`. This smaller example demonstrates that the trigger fires.
The candidate can still fail evaluation and not be promoted; for a successful
promotion example, use the balanced six-sample batch below.

### Demo a successful retraining and promotion

To demonstrate a candidate that passes all five evaluation gates, post this
balanced set of six labelled Iris samples. The API repeats them to the minimum
batch size and applies the same drift shift, so the drift trigger still fires.
This balanced batch was verified with the default model settings and demo
dataset. The baseline-comparison gate also compares with the Production model
already stored in your Docker volume, so a different model history can affect
that gate. It replaces the previous demo production batch and runs a new local
pipeline cycle:

```powershell
$payload = @{
  records = @(
    @{ sepal_length = 4.8; sepal_width = 3.0; petal_length = 1.4; petal_width = 0.1; species = 0 },
    @{ sepal_length = 4.8; sepal_width = 3.0; petal_length = 1.4; petal_width = 0.3; species = 0 },
    @{ sepal_length = 5.9; sepal_width = 3.2; petal_length = 4.8; petal_width = 1.8; species = 1 },
    @{ sepal_length = 6.7; sepal_width = 3.0; petal_length = 5.0; petal_width = 1.7; species = 1 },
    @{ sepal_length = 6.7; sepal_width = 3.0; petal_length = 5.2; petal_width = 2.3; species = 2 },
    @{ sepal_length = 6.4; sepal_width = 2.8; petal_length = 5.6; petal_width = 2.2; species = 2 }
  )
} | ConvertTo-Json -Depth 5

$result = Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8001/demo/records" `
  -ContentType "application/json" `
  -TimeoutSec 600 `
  -Body $payload

$result | Format-List
```

For a successful promotion, the response should show `retrain: True`,
`outcome: promoted`, and `promoted: True`. Confirm the `EVALUATE — All 5 gates`
and `Model promoted to Production` messages in `docker compose logs -f
mlops-demo`. The Grafana retrain-trigger gauge shows that retraining was
triggered; it does not indicate promotion. If this candidate is rejected,
inspect the reported evaluation gates and confirm the current Production model
and demo data are from the expected default setup.

To verify Prometheus directly after the request:

```powershell
(Invoke-RestMethod "http://localhost:9090/api/v1/query?query=mlops_retrain_trigger").data.result
```

For a faster repeat demo, set a shorter interval before starting Compose:

```powershell
$env:PIPELINE_INTERVAL_SECONDS = "60"
docker compose up -d
```

This interval controls how often another complete monitoring/retraining
decision runs; the first run is still immediate. Clear the PowerShell override
after stopping the stack with `Remove-Item Env:PIPELINE_INTERVAL_SECONDS`.

View recent logs, then stop the stack:

```powershell
docker compose logs --tail 100 mlops-demo
docker compose down
```

Data, models, MLflow state, and monitoring history persist in Docker volumes.
To irreversibly delete the persisted demo data, models, MLflow database, and
monitoring history and start over with clean volumes:

```powershell
docker compose down -v
```

To build the application image without starting the full stack:

```powershell
docker build -t iris-mlops-demo:local .
```
