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