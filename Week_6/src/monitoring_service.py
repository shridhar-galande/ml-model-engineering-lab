"""Run the retraining pipeline periodically and expose its metrics to Prometheus."""

import argparse
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pandas as pd
import yaml
from prometheus_client import Counter, Gauge, start_http_server

with open("config.yaml") as config_file:
    CFG = yaml.safe_load(config_file)

PATHS = CFG["paths"]
FEATURE_COLS = CFG["features"]["names"]
TARGET_COL = CFG["features"]["target"]
LOGGER = logging.getLogger("mlops_monitoring")
PIPELINE_LOCK = threading.Lock()
DEMO_MODE_ENABLED = False

PIPELINE_RUNS = Counter(
    "mlops_pipeline_runs_total",
    "Completed pipeline runs, grouped by outcome.",
    ["outcome"],
)
RETRAIN_DECISIONS = Counter(
    "mlops_retrain_decisions_total", "Pipeline cycles that triggered retraining."
)
PIPELINE_RUNNING = Gauge(
    "mlops_pipeline_running", "Whether a pipeline cycle is currently running."
)
PIPELINE_FAILED = Gauge(
    "mlops_pipeline_failed", "Whether the most recent pipeline cycle failed."
)
PIPELINE_DURATION = Gauge(
    "mlops_pipeline_duration_seconds", "Duration of the most recent pipeline cycle."
)
LAST_PIPELINE_SUCCESS = Gauge(
    "mlops_last_successful_pipeline_timestamp_seconds",
    "Unix timestamp of the last successful pipeline cycle.",
)
MONITOR_TRIGGER = Gauge(
    "mlops_monitor_drift_trigger", "Whether monitoring detected a retrain trigger."
)
RETRAIN_TRIGGER = Gauge(
    "mlops_retrain_trigger", "Whether trigger evaluation decided to retrain."
)
TRIGGER_REASON = Gauge(
    "mlops_retrain_trigger_reason",
    "Whether a named retraining trigger fired in the latest cycle.",
    ["trigger"],
)
FEATURE_PSI = Gauge(
    "mlops_feature_drift_psi", "Population Stability Index by feature.", ["feature"]
)
MODEL_ACCURACY = Gauge(
    "mlops_model_accuracy", "Most recently observed model accuracy."
)
MODEL_F1 = Gauge("mlops_model_f1_macro", "Most recently observed macro F1 score.")


def build_trigger_batch(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate submitted demo records and expand them into a drift-sized batch."""
    if not isinstance(records, list) or not records or len(records) > 100:
        raise ValueError("records must contain between 1 and 100 items")

    columns = FEATURE_COLS + [TARGET_COL]
    expected = set(columns)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError(
                f"record {index} must contain exactly these fields: "
                f"{', '.join(columns)}"
            )

    batch = pd.DataFrame.from_records(records, columns=columns)
    try:
        for column in columns:
            batch[column] = pd.to_numeric(batch[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("all feature and species values must be numeric") from exc

    if not np.isfinite(batch[columns].to_numpy(dtype=float)).all():
        raise ValueError("feature and species values must be finite")
    species = batch[TARGET_COL]
    if not (species % 1 == 0).all() or not species.isin(
        CFG["features"]["classes"]
    ).all():
        raise ValueError("species must be a configured integer class label")
    batch[TARGET_COL] = species.astype(int)

    min_samples = max(1, CFG["triggers"]["drift_based"]["min_samples_for_drift"])
    batch = batch.iloc[[index % len(batch) for index in range(min_samples)]].copy()
    batch["sepal_width"] = batch["sepal_width"] + 1.0
    return batch


def build_demo_production(
    source_df: pd.DataFrame, sample_count: int = 120
) -> pd.DataFrame:
    """Create a labelled, reproducible production batch with detectable drift."""
    production = source_df.sample(
        n=sample_count, replace=True, random_state=42
    ).reset_index(drop=True)
    production["sepal_width"] = production["sepal_width"] + 1.0
    return production[FEATURE_COLS + [TARGET_COL]]


def prepare_demo_assets(demo_mode: bool) -> None:
    from data_generator import (
        generate_golden_dataset,
        generate_production_data,
        generate_reference_data,
        load_iris_as_dataframe,
    )

    source_df = load_iris_as_dataframe()
    if not all(
        os.path.isfile(PATHS[key])
        for key in ("reference_data", "production_data", "golden_dataset")
    ):
        generate_reference_data(source_df, PATHS["reference_data"])
        generate_golden_dataset(source_df, PATHS["golden_dataset"])
        generate_production_data(source_df, PATHS["production_data"], "none")

    if demo_mode:
        production = build_demo_production(source_df)
        os.makedirs(os.path.dirname(PATHS["production_data"]) or ".", exist_ok=True)
        production.to_csv(PATHS["production_data"], index=False)
        LOGGER.info(
            "Demo data ready: %s labelled production rows with sepal_width drift",
            len(production),
        )


def bootstrap_model() -> None:
    if os.path.isfile(PATHS["current_model"]):
        return

    LOGGER.info("No current model found; training and evaluating the initial model.")
    from evaluate import run_evaluation
    from train import run_training

    run_training()
    result = run_evaluation(auto_promote=True)
    if not result["all_passed"] or not os.path.isfile(PATHS["current_model"]):
        raise RuntimeError("Initial model failed evaluation; monitoring cannot start.")


def update_metrics(record: dict) -> None:
    monitor = record.get("monitor", {})
    trigger = record.get("trigger", {})
    trigger_results = record.get("trigger_results", {})
    monitor_triggered = bool(monitor.get("trigger_retraining", False))
    retrain_triggered = bool(trigger.get("retrain", False))
    outcome = record.get("outcome", "error")

    MONITOR_TRIGGER.set(float(monitor_triggered))
    RETRAIN_TRIGGER.set(float(retrain_triggered))
    PIPELINE_FAILED.set(float(str(outcome).startswith("error")))
    PIPELINE_DURATION.set(float(record.get("duration_secs", 0.0)))
    PIPELINE_RUNS.labels(outcome=_metric_outcome(outcome)).inc()
    if retrain_triggered:
        RETRAIN_DECISIONS.inc()
    if not str(outcome).startswith("error"):
        LAST_PIPELINE_SUCCESS.set(time.time())

    for name in ("time", "performance", "drift", "active_learning"):
        TRIGGER_REASON.labels(trigger=name).set(
            float(trigger_results.get(name, {}).get("fire", False))
        )

    features = monitor.get("feature_drift", {}).get("per_feature", {})
    for feature in FEATURE_COLS:
        FEATURE_PSI.labels(feature=feature).set(
            float(features.get(feature, {}).get("psi", 0.0))
        )

    performance = monitor.get("performance", {})
    train_metrics = record.get("train_metrics", {})
    accuracy = performance.get("accuracy", train_metrics.get("accuracy"))
    f1_macro = performance.get("f1_macro", train_metrics.get("f1_macro"))
    if accuracy is not None:
        MODEL_ACCURACY.set(float(accuracy))
    if f1_macro is not None:
        MODEL_F1.set(float(f1_macro))


def _metric_outcome(outcome) -> str:
    value = str(outcome)
    if value.startswith("error"):
        return "error"
    return value


def _execute_pipeline_cycle() -> dict:
    PIPELINE_RUNNING.set(1)
    metrics_recorded = False
    try:
        from pipeline import run_pipeline

        record = run_pipeline()
        update_metrics(record)
        metrics_recorded = True
        if str(record.get("outcome", "")).startswith("error"):
            raise RuntimeError(f"Pipeline completed with error: {record['outcome']}")
        LOGGER.info("Pipeline outcome: %s", record["outcome"])
        return record
    except Exception:
        PIPELINE_FAILED.set(1)
        if not metrics_recorded:
            PIPELINE_RUNS.labels(outcome="error").inc()
        LOGGER.exception("Monitoring pipeline cycle failed")
        raise
    finally:
        PIPELINE_RUNNING.set(0)


class DemoIngestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/demo/records":
            self._send_json(404, {"error": "use POST /demo/records"})
            return
        if not DEMO_MODE_ENABLED:
            self._send_json(403, {"error": "demo record ingestion is disabled"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if content_length < 1:
            self._send_json(400, {"error": "request body is required"})
            return
        if content_length > 1_000_000:
            self._send_json(413, {"error": "request body exceeds 1 MB"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "request body must be valid JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return

        try:
            records = payload.get("records")
            production = build_trigger_batch(records)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        try:
            with PIPELINE_LOCK:
                os.makedirs(os.path.dirname(PATHS["production_data"]) or ".", exist_ok=True)
                production.to_csv(PATHS["production_data"], index=False)
                LOGGER.info(
                    "Received %d demo records; running pipeline on %d drifted rows",
                    len(records),
                    len(production),
                )
                result = _execute_pipeline_cycle()
        except Exception as exc:
            self._send_json(
                500, {"error": "pipeline execution failed", "detail": str(exc)}
            )
            return

        self._send_json(
            200,
            {
                "records_received": len(records),
                "production_rows": len(production),
                "monitor_trigger": result.get("monitor", {}).get(
                    "trigger_retraining", False
                ),
                "retrain": result.get("trigger", {}).get("retrain", False),
                "triggered_by": result.get("trigger", {}).get("triggered_by", []),
                "outcome": result.get("outcome"),
                "promoted": result.get("promoted", False),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("Demo ingest: %s", format % args)


def run_service(interval_seconds: int, port: int, demo_mode: bool) -> None:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be at least 1")

    global DEMO_MODE_ENABLED
    DEMO_MODE_ENABLED = demo_mode
    start_http_server(port, addr="0.0.0.0")
    LOGGER.info("Prometheus metrics available on http://0.0.0.0:%s/metrics", port)
    prepare_demo_assets(demo_mode)
    bootstrap_model()
    ingest_server = ThreadingHTTPServer(("0.0.0.0", 8001), DemoIngestHandler)
    threading.Thread(target=ingest_server.serve_forever, daemon=True).start()
    LOGGER.info(
        "Demo record endpoint available at http://0.0.0.0:8001/demo/records"
    )

    while True:
        try:
            with PIPELINE_LOCK:
                _execute_pipeline_cycle()
        except Exception:
            LOGGER.error("Scheduled pipeline cycle failed; service will retry.")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.getenv("PIPELINE_INTERVAL_SECONDS", "300")),
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("METRICS_PORT", "8000"))
    )
    parser.add_argument(
        "--demo-mode",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("DEMO_MODE", "true").lower() == "true",
    )
    args = parser.parse_args()
    run_service(args.interval_seconds, args.port, args.demo_mode)
