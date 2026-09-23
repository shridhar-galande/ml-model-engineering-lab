# Lab 2 — Batch vs. Online Inference

## Overview

This lab compares two inference patterns for the same trained machine learning model:

- Batch inference: score all records in one pass
- Online inference: score one record at a time, as a live request

The goal is to measure and understand the trade-off between throughput and latency in a simple, realistic ML serving scenario.

## Learning Objectives

By the end of this lab, you should be able to:

- explain why batch and online inference have different latency profiles
- compute percentile-based latency metrics such as p50, p95, and p99
- interpret the throughput-versus-latency trade-off using real measurements
- compare the practical implications of batch jobs and online serving systems

## Dataset and Model

- Dataset: `load_breast_cancer` from scikit-learn
- Task: binary classification (malignant vs. benign)
- Model: `RandomForestClassifier`
- Why this lab works well: the data is small enough to run quickly, while still representing a realistic risk-scoring use case

## Files in This Project

- [Lab2_Batch_vs_Online_Inference.ipynb](Lab2_Batch_vs_Online_Inference.ipynb) — main notebook for the experiment
- [requirements.txt](requirements.txt) — Python dependencies
- [README.md](README.md) — project overview and instructions

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Lab

Open the notebook and run all cells in order:

```bash
jupyter notebook Lab2_Batch_vs_Online_Inference.ipynb
```

The notebook will:

1. load the dataset
2. split it into training and test sets
3. train a Random Forest model
4. run batch inference over the full test set
5. run online inference one record at a time
6. measure latency distributions
7. compare batch and online performance visually and numerically

## Expected Output

The notebook produces the following artifacts:

- `batch_predictions.csv` — predictions generated in the batch scoring job
- `latency_histogram.png` — distribution of per-request online latencies
- `latency_percentiles.png` — percentile plot for p50, p95, p99, and max latency
- summary comparison of batch vs. online inference metrics

## Key Insight

A batch job is usually more efficient when many predictions can be processed together, because the inference engine handles a large vector of data in one call. An online endpoint, however, must respond to each request quickly and individually, so per-request overhead becomes much more important.

## Discussion Questions

1. Why is batch inference cheaper per record than single-record online inference?
2. Why do p95 and p99 latencies matter more for live endpoints than for nightly batch jobs?
3. If an online endpoint had a strict p95 SLA, what changes might you consider to reduce latency?
4. Besides model inference, what else differs between a batch service and a production online endpoint?

## Notes

This lab is intentionally designed to focus on serving behavior rather than model training quality. The objective is to understand how inference architecture affects performance, not to optimize the model itself.

