"""
src/tests/test_pipeline.py
======================
Unit tests for the Iris CT pipeline components.
Run with:  pytest src/tests/ -v
"""

import json
import os
import pickle
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "species"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def iris_df():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_COLS)
    df[TARGET_COL] = iris.target
    return df


@pytest.fixture
def trained_model(iris_df):
    X = iris_df[FEATURE_COLS].values
    y = iris_df[TARGET_COL].values
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X, y)
    return m


# ── data_generator ────────────────────────────────────────────────────────────


def test_generate_reference_data(iris_df, tmp_path):
    from data_generator import generate_reference_data

    path = str(tmp_path / "ref.csv")
    ref = generate_reference_data(iris_df, path)
    assert os.path.exists(path)
    assert len(ref) == int(len(iris_df) * 0.8)
    assert set(ref.columns) == set(FEATURE_COLS + [TARGET_COL])


def test_generate_production_data_covariate(iris_df, tmp_path):
    from data_generator import generate_production_data

    path = str(tmp_path / "prod.csv")
    prod = generate_production_data(iris_df, path, drift_type="covariate")
    assert os.path.exists(path)
    # sepal_length should be higher than original
    orig_mean = iris_df["sepal_length"].mean()
    assert prod["sepal_length"].mean() > orig_mean


def test_generate_golden_dataset(iris_df, tmp_path):
    from data_generator import generate_golden_dataset

    path = str(tmp_path / "golden.csv")
    golden = generate_golden_dataset(iris_df, path, n_per_class=3)
    assert len(golden) == 9  # 3 classes × 3 per class
    assert golden[TARGET_COL].nunique() == 3


# ── monitor (statistical tests) ───────────────────────────────────────────────


def test_psi_no_drift(iris_df):
    from monitor import compute_psi

    col = iris_df["sepal_length"]
    psi = compute_psi(col, col)
    assert psi is not None
    assert psi < 0.10, f"PSI={psi} should be near 0 for identical distributions"


def test_psi_with_drift(iris_df):
    from monitor import compute_psi

    ref = iris_df["sepal_length"]
    prod = ref + 0.8  # clear shift
    psi = compute_psi(ref, prod)
    assert psi > 0.10, f"PSI={psi} should detect shift"


def test_demo_production_batch_triggers_feature_drift(iris_df):
    from monitor import run_feature_drift
    from monitoring_service import build_demo_production

    production = build_demo_production(iris_df)
    result = run_feature_drift(iris_df, production, FEATURE_COLS)

    assert len(production) >= 50
    assert result["trigger_retraining"]


def test_submitted_demo_records_expand_and_trigger_drift(iris_df):
    from monitoring_service import build_trigger_batch

    records = iris_df.iloc[:3].to_dict(orient="records")
    production = build_trigger_batch(records)

    assert len(production) >= 50
    assert set(production[TARGET_COL]) == set(iris_df[TARGET_COL].iloc[:3])
    assert np.allclose(
        production["sepal_width"].iloc[:3].values,
        iris_df["sepal_width"].iloc[:3].values + 1.0,
    )


def test_demo_ingest_endpoint_runs_pipeline(iris_df, monkeypatch, tmp_path):
    import monitoring_service as service

    records = iris_df.iloc[:3].to_dict(orient="records")
    production_path = tmp_path / "production.csv"
    monkeypatch.setattr(service, "DEMO_MODE_ENABLED", True)
    monkeypatch.setitem(service.PATHS, "production_data", str(production_path))
    monkeypatch.setattr(
        service,
        "_execute_pipeline_cycle",
        lambda: {
            "monitor": {"trigger_retraining": True},
            "trigger": {"retrain": True, "triggered_by": ["drift"]},
            "outcome": "evaluation_failed",
            "promoted": False,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.DemoIngestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/demo/records",
            data=json.dumps({"records": records}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read())
        assert result["records_received"] == 3
        assert result["production_rows"] >= 50
        assert result["monitor_trigger"] is True
        assert result["retrain"] is True
        assert result["outcome"] == "evaluation_failed"
        assert len(pd.read_csv(production_path)) >= 50
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ks_no_drift(iris_df):
    from monitor import compute_ks

    col = iris_df["sepal_length"]
    ks = compute_ks(col, col)
    assert ks["p_value"] > 0.05
    assert ks["drifted"] is False


def test_ks_with_drift(iris_df):
    from monitor import compute_ks

    ref = iris_df["sepal_length"]
    prod = pd.Series(ref.values + 1.5)
    ks = compute_ks(ref, prod)
    assert ks["drifted"] is True


def test_jsd_identical_returns_zero(iris_df):
    from monitor import compute_jsd

    col = iris_df["sepal_length"]
    jsd = compute_jsd(col, col)
    assert jsd < 0.05, f"JSD={jsd} for identical distributions should be ~0"


# ── evaluate (gates) ──────────────────────────────────────────────────────────


def test_gate_minimum_thresholds_pass(iris_df, trained_model, tmp_path):
    """A well-trained model on clean data should pass the minimum thresholds."""
    bundle = {
        "model": trained_model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": {},
    }
    # Write bundle to temp path
    pkl = tmp_path / "cand.pkl"
    pickle.dump(bundle, open(pkl, "wb"))

    from evaluate import gate_minimum_thresholds

    passed, result = gate_minimum_thresholds(bundle, iris_df)
    assert passed, f"Expected pass but got: {result}"
    assert result["accuracy"] >= 0.85


def test_gate_golden_dataset_pass(iris_df, trained_model, tmp_path):
    """Model trained on full Iris should predict all golden examples correctly."""
    from data_generator import generate_golden_dataset

    golden_path = str(tmp_path / "golden.csv")
    golden_df = generate_golden_dataset(iris_df, golden_path, n_per_class=5)
    bundle = {
        "model": trained_model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": {},
    }

    from evaluate import gate_golden_dataset

    passed, result = gate_golden_dataset(bundle, golden_df)
    assert passed, f"Expected golden pass but n_wrong={result['n_wrong']}"


def test_gate_distribution_check_clean(iris_df):
    from evaluate import gate_distribution_check

    passed, result = gate_distribution_check(iris_df)
    assert passed, f"Clean Iris should pass distribution check: {result['issues']}"


def test_gate_distribution_check_missing(iris_df):
    from evaluate import gate_distribution_check

    df_with_na = iris_df.copy()
    df_with_na.loc[0, "sepal_length"] = np.nan
    passed, result = gate_distribution_check(df_with_na)
    assert not passed
    assert any("missing" in i for i in result["issues"])


# ── retrain_trigger ────────────────────────────────────────────────────────────


def test_psi_trigger_fires_on_large_drift(iris_df):
    from retrain_trigger import _psi

    ref = iris_df["sepal_length"]
    prod = ref + 2.0
    psi = _psi(ref, prod)
    assert psi > 0.20, f"PSI={psi} should trigger on large drift"


def test_active_learning_trigger_high_confidence(iris_df, trained_model, tmp_path):
    """A well-trained model should NOT fire the uncertainty trigger."""
    # Write bundle
    bundle_path = tmp_path / "current_model.pkl"
    bundle = {
        "model": trained_model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": {},
    }
    pickle.dump(bundle, open(bundle_path, "wb"))

    # Patch PATHS temporarily
    import retrain_trigger as rt

    orig = rt.PATHS["current_model"]
    rt.PATHS["current_model"] = str(bundle_path)

    metrics = {}
    result = rt.check_active_learning(iris_df, metrics)
    rt.PATHS["current_model"] = orig

    # Clean Iris model should be confident
    assert (
        result["uncertain_fraction"] < 0.20
    ), f"Well-trained model should not fire AL trigger: {result}"
