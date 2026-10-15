import json
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def post_predictions(rows, request_id):
    payload = json.dumps(
        {"dataframe_split": {"columns": FEATURES, "data": rows}}
    ).encode()
    request = urllib.request.Request(
        f"{BASE_URL}/invocations",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        assert response.headers["X-Request-ID"] == request_id
        return json.load(response)


healthy = post_predictions([[5.1, 3.5, 1.4, 0.2]], "ci-healthy")
assert healthy["predictions"] == ["setosa"], healthy

shifted = post_predictions([[7.5, 4.8, 6.8, 2.5]] * 100, "ci-drift")
assert len(shifted["predictions"]) == 100, shifted

with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10) as response:
    metrics = response.read().decode()

assert 'model_requests_total{outcome="success"} 2.0' in metrics
assert 'model_predictions_total{prediction="setosa"} 1.0' in metrics
assert 'model_feature_psi{feature="petal_length"}' in metrics
assert 'model_feature_psi{feature="petal_length"} 0.0' not in metrics
print("Container smoke test passed: prediction, request metrics, and feature drift.")
