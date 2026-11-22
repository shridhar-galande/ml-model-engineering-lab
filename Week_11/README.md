# Security, Compliance & Responsible AI in Production
### Deliverables

This package contains the slide deck and companion demo notebook for the session.

## Contents

| File | Description |
|---|---|
| `Security_Compliance_Responsible_AI.pptx` | 45-slide deck covering all 5 session topics, with "Live Demo" transition slides pointing to the notebook |
| `Security_Compliance_Responsible_AI_Demo.ipynb` | Jupyter notebook with 4 independent, fully executed demo blocks — one per deck section |

## Session Structure

| # | Section | Notebook Block |
|---|---|---|
| 1 | Model Security — why it matters, the four threat buckets (data, input, model & privacy, infrastructure & access), defensive design | Block 1 |
| 2 | Data Privacy — PII, minimisation, anonymisation, access control, governance | Block 2 |
| 3 | Responsible AI — fairness (group-wise metrics + limits), explainability (local & global), audit trails | Block 3 |
| 4 | Compliance — SOC2, GDPR, HIPAA in ML systems | — (conceptual only) |
| 5 | Case Study — drift vs. fairness trade-offs | Block 4 |

Plus intro and closing sections.

Each concept section in the deck is followed by a "Live Demo" slide that names the matching
notebook block, its dataset, and what to run live.

## Notebook Demo Blocks

| Block | Section | Dataset | What It Shows |
|---|---|---|---|
| 1 | Model Security | Breast Cancer Wisconsin (`sklearn`) | One sub-demo per threat bucket: a black-box extraction attack (~99% fidelity from query access alone), a membership inference score, a data-poisoning sweep (accuracy drop as label noise increases), a Mahalanobis-distance OOD detector (100% detection on synthetic OOD samples), and a query-rate anomaly detector that catches the extraction attack's traffic pattern |
| 2 | Data Privacy | Adult Income (UCI, via `shap`) | Quantifies re-identification risk from quasi-identifiers, applies generalization/masking, and compares risk reduction against the accuracy cost |
| 3 | Responsible AI | NHANES-I Health Survey (UCI, via `shap`) | Trains a mortality-risk classifier, slices accuracy/false-positive rate by sex, computes demographic parity and equalized odds gaps, and explains predictions with SHAP both locally (one record) and globally (overall feature importance) |
| 4 | Case Study | Diabetes (`sklearn`), split by age cohort | Simulates population drift, detects it with PSI, shows a fairness gap emerging under drift, and compares an accuracy-only retrain against a fairness-constrained retrain — plus an interactive threshold slider |

### Dataset substitution note

The original plan called for the **German Credit Data** set in Block 3. That dataset is hosted on
the UCI repository / OpenML, which this build environment's network allowlist can't reach.
**NHANES-I** (bundled with the `shap` library, no external fetch required) is used instead — a real
health survey with age and sex attributes that supports the same fairness + explainability workflow.
If you have network access to UCI/OpenML in your own environment, swapping in German Credit Data
would need: reloading the dataset, adjusting `feature_cols`, and re-pointing the protected attribute
from `sex_isFemale` to the credit dataset's `age`/`foreign_worker` field.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib joblib shap fairlearn ipywidgets
```

Everything else (Breast Cancer Wisconsin, Diabetes) ships with `scikit-learn`; Adult Income and
NHANES-I are fetched automatically by `shap.datasets` on first run (small one-time download).

Run the notebook top to bottom, or jump directly to any single block — each is self-contained.

## Design Conventions

- **Deck:** dark navy/teal palette (`#0B2545` / `#1C7293`), Cambria headings, Calibri body, no
  emojis, no slide timestamps
- **Notebook:** matching navy/teal/coral palette for all charts, no timestamps, real datasets
  throughout (no synthetic data)
