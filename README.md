# ML Model Engineering Lab

This repository contains weekly labs for learning how to build, deploy, monitor, optimize, and scale production machine learning systems.

## About ML Model Engineering

ML model engineering is the practice of turning machine learning models into reliable, maintainable, and useful production systems. It connects data science with software engineering, data engineering, cloud infrastructure, and operations.

A production ML system usually moves through the following lifecycle:

1. Collect and validate data and features.
2. Train, evaluate, and version the model.
3. Package and deploy the model for batch, online, or streaming inference.
4. Monitor model quality, data drift, latency, availability, and cost.
5. Retrain, roll back, or retire the model as its conditions change.

The goal is not only high prediction accuracy. A well-engineered ML system must also be reproducible, scalable, observable, secure, cost-aware, and responsible. These labs explore the engineering decisions and trade-offs required to operate models reliably in real-world environments.

## Modules

The course covers:

- **M1-M3:** ML model engineering fundamentals, inference, and serving strategies
- **M4-M6:** Deployment pipelines, MLOps, monitoring, observability, and retraining
- **M7-M8:** Model optimization and production trade-offs
- **M9-M10:** Feature stores, data engineering, and real-time ML pipelines
- **M11-M12:** Security, responsible AI, and multi-model systems
- **M13:** Large-scale ML system design and the capstone project

## Repository Structure

Each week's work is stored in its own folder with the relevant notebooks, source code, dependencies, and instructions.

```text
.
├── README.md
├── Week_1/
├── Week_2/
│   ├── Lab2_Batch_vs_Online_Inference.ipynb
│   ├── README.md
│   └── requirements.txt
├── Week_3/
└── ...
```
---