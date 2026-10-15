import base64
import json
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROMETHEUS_URL = "http://127.0.0.1:9090"
GRAFANA_URL = "http://127.0.0.1:3000"


def get_json(url, headers=None):
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def wait_for(check, description, timeout=90):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, URLError, ValueError) as exc:
            last_error = exc
        time.sleep(3)
    raise AssertionError(f"Timed out waiting for {description}: {last_error}")


def prometheus_query(expression):
    url = f"{PROMETHEUS_URL}/api/v1/query?{urlencode({'query': expression})}"
    response = get_json(url)
    assert response["status"] == "success", response
    return response["data"]["result"]


def prometheus_ready():
    with urlopen(f"{PROMETHEUS_URL}/-/ready", timeout=5) as response:
        return response.status == 200


wait_for(
    prometheus_ready,
    "Prometheus readiness",
)
wait_for(
    lambda: any(
        sample["value"][1] == "1"
        for sample in prometheus_query('up{job="monitored-model"}')
    ),
    "Prometheus scraping the model API",
)
wait_for(
    lambda: any(
        float(sample["value"][1]) > 0
        for sample in prometheus_query('model_feature_psi{feature="petal_length"}')
    ),
    "non-zero feature PSI in Prometheus",
)
rules = wait_for(
    lambda: get_json(f"{PROMETHEUS_URL}/api/v1/rules")["data"]["groups"],
    "Prometheus alert rules",
)
assert any(group["name"] == "model-monitoring" for group in rules), rules

authorization = base64.b64encode(b"admin:admin").decode()
grafana_headers = {"Authorization": f"Basic {authorization}"}
dashboard = wait_for(
    lambda: get_json(
        f"{GRAFANA_URL}/api/dashboards/uid/model-monitoring",
        headers=grafana_headers,
    ),
    "Grafana provisioned monitoring dashboard",
)
assert dashboard["dashboard"]["title"] == "Model Monitoring", dashboard
print("Monitoring stack smoke test passed: Prometheus scrape/PSI/alerts and Grafana dashboard.")
