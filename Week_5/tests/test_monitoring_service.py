import numpy as np
from fastapi.testclient import TestClient

from src.monitoring_service import FEATURES, create_app


class StubModel:
    def predict(self, frame):
        return np.array(["setosa"] * len(frame))


def make_client(drift_window_size=4):
    return TestClient(create_app(StubModel(), drift_window_size=drift_window_size))


def test_health_endpoint():
    response = make_client().get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predictions_are_served_and_instrumented():
    client = make_client()
    response = client.post(
        "/invocations",
        headers={"X-Request-ID": "demo-123"},
        json={
            "dataframe_split": {
                "columns": FEATURES,
                "data": [[5.1, 3.5, 1.4, 0.2]],
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": ["setosa"]}
    assert response.headers["X-Request-ID"] == "demo-123"
    metrics = client.get("/metrics").text
    assert 'model_requests_total{outcome="success"} 1.0' in metrics
    assert 'model_predictions_total{prediction="setosa"} 1.0' in metrics
    assert "model_request_duration_seconds_count 1.0" in metrics


def test_invalid_input_is_counted_as_error():
    client = make_client()
    response = client.post(
        "/invocations",
        json={"dataframe_split": {"columns": FEATURES, "data": []}},
    )

    assert response.status_code == 422
    assert 'model_requests_total{outcome="error"} 1.0' in client.get("/metrics").text


def test_shifted_window_updates_feature_drift_metrics():
    client = make_client(drift_window_size=4)
    response = client.post(
        "/invocations",
        json={
            "dataframe_split": {
                "columns": FEATURES,
                "data": [[7.5, 4.8, 6.8, 2.5]] * 4,
            }
        },
    )

    assert response.status_code == 200
    metrics = client.get("/metrics").text
    assert 'model_feature_psi{feature="petal_length"}' in metrics
    assert 'model_feature_psi{feature="petal_length"} 0.0' not in metrics
