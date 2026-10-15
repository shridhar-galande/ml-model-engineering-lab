# ============================================================
#  BONUS — UNIT TESTS
#  tests/test_data_check.py
#
#  WHAT THIS DOES:
#    Tests the data quality checks in isolation.
#    In real CI/CD these run automatically on every git push
#    BEFORE the data check even touches real data.
#
#  RUN IT:
#    python -m pytest tests/ -v
# ============================================================

import sys
import os
import pandas as pd
import pytest

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_check import EXPECTED_COLUMNS, EXPECTED_SPECIES, COLUMN_RANGES


# ── Fixtures — reusable test data ─────────────────────────────

def good_df():
    """A perfectly clean Iris dataframe."""
    return pd.DataFrame({
        "sepal_length": [5.1, 6.3, 4.9],
        "sepal_width":  [3.5, 2.9, 3.0],
        "petal_length": [1.4, 4.7, 1.4],
        "petal_width":  [0.2, 1.6, 0.2],
        "species":      ["setosa", "versicolor", "setosa"],
    })


# ── Tests ──────────────────────────────────────────────────────

def test_correct_columns():
    df = good_df()
    assert list(df.columns) == EXPECTED_COLUMNS, \
        "Column names must match schema"


def test_no_nulls():
    df = good_df()
    assert df.isnull().sum().sum() == 0, \
        "Clean data should have no nulls"


def test_null_detected():
    df = good_df()
    df.loc[0, "sepal_length"] = None
    assert df.isnull().sum().sum() > 0, \
        "Should detect injected null"


def test_valid_species_labels():
    df = good_df()
    unexpected = set(df["species"].unique()) - EXPECTED_SPECIES
    assert len(unexpected) == 0, \
        f"Unexpected species labels: {unexpected}"


def test_bad_species_detected():
    df = good_df()
    df.loc[0, "species"] = "virginicaa"   # typo
    unexpected = set(df["species"].unique()) - EXPECTED_SPECIES
    assert len(unexpected) == 1, \
        "Should detect the typo label"


def test_values_in_range():
    df = good_df()
    for col, (lo, hi) in COLUMN_RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        assert len(bad) == 0, \
            f"All values in '{col}' should be within [{lo}, {hi}]"


def test_out_of_range_detected():
    df = good_df()
    df.loc[0, "petal_width"] = -1.0    # invalid negative value
    lo, hi = COLUMN_RANGES["petal_width"]
    bad = df[(df["petal_width"] < lo) | (df["petal_width"] > hi)]
    assert len(bad) == 1, \
        "Should detect the out-of-range petal_width"


def test_minimum_row_count():
    df = good_df()
    assert len(df) >= 3, "Test fixture has at least 3 rows"


def test_wrong_column_name_detected():
    df = good_df()
    df.rename(columns={"sepal_length": "Sepal_Length"}, inplace=True)
    assert list(df.columns) != EXPECTED_COLUMNS, \
        "Wrong column name should not match schema"
