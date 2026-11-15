"""
consumer.py
-----------
Reads transaction events from Kafka and turns raw events into the things a
real-time ML pipeline actually needs: quality-checked, windowed, model-ready
features -- plus a simple live anomaly score.

The code is organized into the same steps as the lab guide, so you can read
(and modify) it section by section:

  STEP A - Consume  : connect to Kafka and read the topic
  STEP B - Data quality checks : freshness, completeness/correctness, duplicates
  STEP C - Windowing : sliding 60-second window of features, per account
  STEP D - Scoring   : a simple rule-based "anomaly" score over the window
  STEP E - Schema handling : new fields are read if present, ignored if not

Run:
    python consumer.py
"""

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from kafka import KafkaConsumer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "transactions"

WINDOW_SECONDS = 60          # sliding window length for rolling features
FRESHNESS_SLA_SECONDS = 5    # events "should" be processed within this long
BURST_COUNT_THRESHOLD = 5    # more than this many txns in the window -> flag
AMOUNT_SPIKE_MULTIPLIER = 4  # amount > 4x the account's own rolling avg -> flag

# ---- STEP B state: data quality tracking -----------------------------------
seen_transaction_ids = set()
dq_counts = {
    "total": 0,
    "duplicates": 0,
    "missing_fields": 0,
    "invalid_values": 0,
    "out_of_order": 0,
    "stale_over_sla": 0,
}
last_event_time_per_account = {}  # account_id -> most recent event_time seen

# ---- STEP C state: sliding window per account ------------------------------
# each entry: (event_time, amount)
window_by_account = defaultdict(deque)

REQUIRED_FIELDS = ["transaction_id", "account_id", "amount", "event_time"]


def check_data_quality(event: dict) -> dict:
    """
    STEP B - Freshness, completeness, correctness, duplicates.
    Returns a dict of flags for this event; also updates the running dq_counts.
    """
    dq_counts["total"] += 1
    flags = {"duplicate": False, "missing_fields": False, "invalid_values": False,
              "out_of_order": False, "stale": False}

    # Completeness: are the fields we need actually present?
    missing = [f for f in REQUIRED_FIELDS if f not in event]
    if missing:
        dq_counts["missing_fields"] += 1
        flags["missing_fields"] = True
        return flags  # can't check anything else without these fields

    # Correctness: is the value sane?
    if event["amount"] is None or event["amount"] <= 0:
        dq_counts["invalid_values"] += 1
        flags["invalid_values"] = True

    # Duplicates: have we already processed this transaction_id?
    if event["transaction_id"] in seen_transaction_ids:
        dq_counts["duplicates"] += 1
        flags["duplicate"] = True
    seen_transaction_ids.add(event["transaction_id"])

    # Out-of-order: did an older event arrive after a newer one, per account?
    acct = event["account_id"]
    evt_time = datetime.fromisoformat(event["event_time"])
    if acct in last_event_time_per_account and evt_time < last_event_time_per_account[acct]:
        dq_counts["out_of_order"] += 1
        flags["out_of_order"] = True
    else:
        last_event_time_per_account[acct] = evt_time

    # Freshness: how long between the event happening and us processing it?
    lag = (datetime.now(timezone.utc) - evt_time).total_seconds()
    if lag > FRESHNESS_SLA_SECONDS:
        dq_counts["stale_over_sla"] += 1
        flags["stale"] = True

    return flags


def update_window(event: dict):
    """STEP C - Maintain a rolling WINDOW_SECONDS window of (time, amount) per account."""
    acct = event["account_id"]
    evt_time = datetime.fromisoformat(event["event_time"])
    dq = window_by_account[acct]
    dq.append((evt_time, event["amount"]))

    cutoff = evt_time.timestamp() - WINDOW_SECONDS
    while dq and dq[0][0].timestamp() < cutoff:
        dq.popleft()


def window_features(account_id: str) -> dict:
    """STEP C - Compute count / avg / max over the current window for an account."""
    entries = window_by_account[account_id]
    if not entries:
        return {"count": 0, "avg_amount": 0.0, "max_amount": 0.0}
    amounts = [a for _, a in entries]
    return {
        "count": len(amounts),
        "avg_amount": sum(amounts) / len(amounts),
        "max_amount": max(amounts),
    }


def score_event(event: dict, features: dict) -> dict:
    """
    STEP D - A simple, explainable "anomaly" rule over the rolling features.
    (Swap this for a real incremental model -- e.g. river or sklearn
    partial_fit -- once you're comfortable with the pipeline mechanics.)
    """
    reasons = []
    if features["count"] > BURST_COUNT_THRESHOLD:
        reasons.append(f"{features['count']} txns in last {WINDOW_SECONDS}s (burst)")
    if features["avg_amount"] > 0 and event.get("amount", 0) > AMOUNT_SPIKE_MULTIPLIER * features["avg_amount"]:
        reasons.append(f"amount {event['amount']} is >{AMOUNT_SPIKE_MULTIPLIER}x rolling avg "
                        f"({features['avg_amount']:.2f})")
    return {"flagged": bool(reasons), "reasons": reasons}


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )

    print(f"Consuming from topic '{TOPIC}' on {BOOTSTRAP_SERVERS}. Ctrl+C to stop.\n")
    last_summary = time.time()

    try:
        for message in consumer:
            event = message.value

            # STEP B
            flags = check_data_quality(event)
            if flags["missing_fields"]:
                print(f"[DQ] missing required field(s) -- skipping event")
                continue

            # STEP E - schema handling: just read the field if it's there.
            # No branching needed for old events -- this is what backward
            # compatibility buys you.
            schema_tag = "v2" if "device_fingerprint" in event else "v1"

            # STEP C
            update_window(event)
            features = window_features(event["account_id"])

            # STEP D
            score = score_event(event, features)

            flag_str = ", ".join(
                k for k, v in flags.items() if v and k not in ("missing_fields",)
            )
            status = "FLAGGED" if score["flagged"] else "ok"
            print(f"[{status:7s}] {event['transaction_id'][:8]} "
                  f"acct={event['account_id']:9s} amount={event.get('amount'):>8} "
                  f"window(count={features['count']}, avg={features['avg_amount']:.2f}) "
                  f"schema={schema_tag} "
                  f"{'dq=[' + flag_str + ']' if flag_str else ''} "
                  f"{score['reasons']}")

            # Periodic data-quality summary, like a simple monitoring dashboard.
            if time.time() - last_summary > 10:
                print("\n--- Data Quality Summary (last 10s window of totals) ---")
                print(json.dumps(dq_counts, indent=2))
                print("---------------------------------------------------------\n")
                last_summary = time.time()

    except KeyboardInterrupt:
        print("\nStopping consumer.")
        print("\nFinal data quality summary:")
        print(json.dumps(dq_counts, indent=2))
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
