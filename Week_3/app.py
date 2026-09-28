"""FastAPI model server for the wine classifier."""
import os

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the local defaults; variables already set in the shell take precedence.
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v1.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
N_FEATURES = 13

# Load the selected model once when the API process starts.
artifact = joblib.load(MODEL_PATH)
model, classes = artifact["model"], artifact["classes"]
app = FastAPI(title="Wine Classifier", version=MODEL_VERSION)


class PredictRequest(BaseModel):
    # Pydantic validates that each input feature is a number.
    features: list[float]


@app.get("/health")
def health():
    # Used by local checks and Kubernetes readiness probes.
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != N_FEATURES:
        raise HTTPException(status_code=422, detail=f"expected {N_FEATURES} features, got {len(req.features)}")

    # Reshape one request into the 2D array expected by scikit-learn.
    x = np.array(req.features).reshape(1, -1)
    probs = model.predict_proba(x)[0]
    idx = int(np.argmax(probs))
    return {"model_version": MODEL_VERSION, "prediction": idx, "class_name": classes[idx],
            "probabilities": [round(float(p), 4) for p in probs]}
