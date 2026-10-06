# ============================================================
#  STEP 1 — DATA QUALITY GATE
#  src/data_check.py
#
#  WHAT THIS DOES:
#    Before we train anything, we check that the data
#    actually looks the way we expect.
#    If it doesn't → we STOP and tell the user why.
#    This is called a "quality gate" in MLOps.
#
#  RUN IT:
#    python src/data_check.py
# ============================================================

import pandas as pd
import sys
import os

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH = "data/iris_with_issues.csv"

# These are our "schema" rules — what we EXPECT the data to look like.
EXPECTED_COLUMNS = ["sepal_length", "sepal_width",
                    "petal_length", "petal_width", "species"]

EXPECTED_SPECIES = {"setosa", "versicolor", "virginica"}

# Numeric columns must stay within these real-world bounds
COLUMN_RANGES = {
    "sepal_length": (4.0, 8.0),
    "sepal_width":  (2.0, 5.0),
    "petal_length": (1.0, 7.0),
    "petal_width":  (0.1, 2.6),
}


# ── HELPER ───────────────────────────────────────────────────
def check(condition: bool, message: str):
    """
    Like an assert, but prints a clear PASS/FAIL message.
    If condition is False we print the message and exit.
    """
    if condition:
        print(f"  ✅  PASS  — {message}")
    else:
        print(f"  ❌  FAIL  — {message}")
        print("\n⛔  Pipeline stopped. Fix the data before training.")
        sys.exit(1)   # exit code 1 = failure (CI/CD sees this)


# ── MAIN ─────────────────────────────────────────────────────
def run_checks(path: str = DATA_PATH, fix: bool = False):
    """
    Run all data quality checks.

    Parameters
    ----------
    path : str   — path to the CSV file
    fix  : bool  — if True, auto-fix issues and save a clean version
                   (used by pipeline.py so training can continue)
    """
    print("=" * 55)
    print("  DATA QUALITY GATE")
    print(f"  File: {path}")
    print("=" * 55)

    # ── Load ───────────────────────────────────────────────
    df = pd.read_csv(path)
    print(f"\n📂 Loaded {len(df)} rows, {len(df.columns)} columns\n")

    # ── CHECK 1: Column names ──────────────────────────────
    print("[ Check 1 ] Schema — correct column names?")
    actual_cols = list(df.columns)
    check(
        actual_cols == EXPECTED_COLUMNS,
        f"Columns match schema: {actual_cols}"
    )

    # ── CHECK 2: No missing values ──────────────────────────
    print("\n[ Check 2 ] Completeness — any missing values?")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if fix and total_nulls > 0:
        # For demo: fill missing numerics with column median
        for col in EXPECTED_COLUMNS[:-1]:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  🔧  Auto-fixed nulls in '{col}' → filled with median ({median_val:.2f})")
        total_nulls = df.isnull().sum().sum()

    check(
        total_nulls == 0,
        f"No missing values (found {null_counts[null_counts > 0].to_dict() or 'none'})"
    )

    # ── CHECK 3: Value ranges ──────────────────────────────
    print("\n[ Check 3 ] Validity — numeric columns in expected range?")
    for col, (lo, hi) in COLUMN_RANGES.items():
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]

        if fix and len(out_of_range) > 0:
            # Clip to valid range
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"  🔧  Auto-fixed {len(out_of_range)} out-of-range values in '{col}' → clipped to [{lo}, {hi}]")
            out_of_range = df[(df[col] < lo) | (df[col] > hi)]  # recheck

        check(
            len(out_of_range) == 0,
            f"'{col}' in range [{lo}, {hi}]  (bad rows: {len(out_of_range)})"
        )

    # ── CHECK 4: Valid class labels ────────────────────────
    print("\n[ Check 4 ] Labels — only known species values?")
    actual_species = set(df["species"].unique())
    unexpected = actual_species - EXPECTED_SPECIES

    if fix and unexpected:
        # Drop rows with unknown labels
        bad_rows = df[~df["species"].isin(EXPECTED_SPECIES)]
        df = df[df["species"].isin(EXPECTED_SPECIES)]
        print(f"  🔧  Removed {len(bad_rows)} rows with unknown label(s): {unexpected}")
        unexpected = set()

    check(
        len(unexpected) == 0,
        f"Species labels valid  (unexpected: {unexpected or 'none'})"
    )

    # ── CHECK 5: Row count sanity ──────────────────────────
    print("\n[ Check 5 ] Volume — at least 50 rows?")
    check(len(df) >= 50, f"Row count = {len(df)} (need ≥ 50)")

    # ── SAVE CLEAN VERSION ──────────────────────────────────
    if fix:
        clean_path = path.replace(".csv", "_clean.csv")
        df.to_csv(clean_path, index=False)
        print(f"\n💾  Saved clean data → {clean_path}")
        return clean_path

    print("\n✅  All data quality checks passed!")
    return path


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    #
    # ⬇  TEACHING MOMENT ⬇
    #
    # By default we run with fix=False so students SEE the failures.
    # Switch fix=True (or pass --fix) to auto-clean and continue.
    #
    fix_mode = "--fix" in sys.argv
    run_checks(fix=fix_mode)
