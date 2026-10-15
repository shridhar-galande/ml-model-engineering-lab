import json
import logging
import os
import time
import uuid
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from sklearn.datasets import load_iris


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LOGGER = logging.getLogger("model_monitoring")


def create_app(model: Any, drift_window_size: int = 100) -> FastAPI:
    if drift_window_size < 2:
        raise ValueError("drift_window_size must be at least 2")

    registry = CollectorRegistry()
    requests_total = Counter(
        "model_requests_total",
        "Prediction requests handled by outcome",
        ["outcome"],
        registry=registry,
    )
    predictions_total = Counter(
        "model_predictions_total",
        "Predictions returned by class",
        ["prediction"],
        registry=registry,
    )
    request_latency = Histogram(
        "model_request_duration_seconds",
        "Prediction request latency in seconds",
        registry=registry,
    )
    feature_psi = Gauge(
        "model_feature_psi",
        "Population Stability Index for recent inputs versus the Iris reference",
        ["feature"],
        registry=registry,
    )

    reference = load_iris(as_frame=True).data
    reference.columns = FEATURES
    edges: dict[str, np.ndarray] = {}
    reference_shares: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        bins = np.unique(np.quantile(reference[feature], np.linspace(0, 1, 11)))
        if len(bins) < 2:
            raise ValueError(f"Reference feature {feature} must have at least two distinct values")
        edges[feature] = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))
        counts, _ = np.histogram(reference[feature], bins=edges[feature])
        reference_shares[feature] = counts / counts.sum()
        feature_psi.labels(feature=feature).set(0)

    recent_rows: deque[np.ndarray] = deque(maxlen=drift_window_size)
    app = FastAPI(title="Monitored Iris Model", version=os.getenv("MODEL_VERSION", "local"))
    app.state.model = model
    app.state.registry = registry

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        if isinstance(app.state.model, LazyModel):
            app.state.model.load()
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/invocations")
    async def invoke(request: Request) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        outcome = "success"
        status_code = 200
        try:
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
            try:
                split = payload["dataframe_split"]
                columns = split["columns"]
                rows = split["data"]
                if columns != FEATURES or not rows:
                    raise ValueError(
                        f"Expected non-empty dataframe_split with columns {FEATURES}"
                    )
                frame = pd.DataFrame(rows, columns=columns).astype(float)
                if not np.isfinite(frame.to_numpy()).all():
                    raise ValueError("Input features must all be finite numbers")
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            predictions = np.asarray(app.state.model.predict(frame)).reshape(-1)
            if len(predictions) != len(frame):
                raise RuntimeError("Model returned a different number of predictions than input rows")

            for row in frame.to_numpy():
                recent_rows.append(row)
            if len(recent_rows) >= drift_window_size:
                window = np.asarray(recent_rows)
                for index, feature in enumerate(FEATURES):
                    counts, _ = np.histogram(window[:, index], bins=edges[feature])
                    current_shares = counts / counts.sum()
                    epsilon = 1e-6
                    psi = np.sum(
                        (current_shares - reference_shares[feature])
                        * np.log(
                            (current_shares + epsilon)
                            / (reference_shares[feature] + epsilon)
                        )
                    )
                    feature_psi.labels(feature=feature).set(float(psi))

            for prediction in predictions:
                predictions_total.labels(prediction=str(prediction)).inc()
            return Response(
                content=json.dumps({"predictions": predictions.tolist()}),
                media_type="application/json",
                headers={"X-Request-ID": request_id},
            )
        except HTTPException as exc:
            outcome = "error"
            status_code = exc.status_code
            raise
        except Exception as exc:
            outcome = "error"
            status_code = 500
            LOGGER.exception("Prediction failed for request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="Model prediction failed") from exc
        finally:
            requests_total.labels(outcome=outcome).inc()
            request_latency.observe(time.perf_counter() - started)
            LOGGER.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status_code": status_code,
                        "outcome": outcome,
                    }
                )
            )

    return app


class LazyModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def load(self) -> None:
        if self._model is None:
            import mlflow.pyfunc

            self._model = mlflow.pyfunc.load_model(self.model_path)

    def predict(self, frame: pd.DataFrame) -> Any:
        self.load()
        return self._model.predict(frame)


def load_model_app() -> FastAPI:
    model_path = os.getenv("MODEL_PATH", "/model")
    return create_app(LazyModel(model_path))


app = load_model_app()
