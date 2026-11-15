"""
producer.py
-----------
Simulates a real-time stream of card-transaction events and publishes them
to a Kafka topic. This plays the role of the "producer" from the streaming
architecture section: an upstream application writing events as they happen.

To make the lab useful for the Data Quality section, this producer
deliberately injects the messy realities of real event streams:
  - out-of-order events   (network delay -> arrives after later events)
  - duplicate events       (retry logic re-sends the same event)
  - corrupted/invalid values (missing or negative amount)
  - a schema change partway through (adds an optional new field)

Run:
    python producer.py
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "transactions"

ACCOUNTS = [f"acct_{i:03d}" for i in range(1, 21)]
MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "dining", "fuel", "online"]

# After this many events, start including the new optional field.
# This simulates a producer (app) being upgraded to schema v2 mid-stream.
SCHEMA_V2_AFTER_EVENT = 40

# A small pool of recently sent events, used to occasionally resend a
# duplicate and to occasionally send a late/out-of-order event.
recent_events = []


def make_event(event_num: int) -> dict:
    """Build one transaction event. event_time is when it 'really happened'."""
    account_id = random.choice(ACCOUNTS)
    amount = round(random.uniform(3, 250), 2)

    # ~8% of the time, spike the amount and fire a burst of transactions on
    # the same account in quick succession -- this is the pattern the
    # windowed features in consumer.py are designed to catch.
    if random.random() < 0.08:
        amount = round(random.uniform(400, 1500), 2)

    event = {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account_id,
        "amount": amount,
        "event_time": datetime.now(timezone.utc).isoformat(),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
    }

    # Schema evolution: v2 adds an optional field. Backward-compatible --
    # old consumer logic that doesn't know about this field should still work.
    if event_num >= SCHEMA_V2_AFTER_EVENT:
        event["device_fingerprint"] = f"dev_{random.randint(1000, 9999)}"

    return event


def maybe_corrupt(event: dict) -> dict:
    """~4% chance: simulate a corrupted/invalid record."""
    if random.random() < 0.04:
        corruption = random.choice(["missing_amount", "negative_amount", "missing_account"])
        event = dict(event)  # copy
        if corruption == "missing_amount":
            del event["amount"]
        elif corruption == "negative_amount":
            event["amount"] = -abs(event["amount"])
        elif corruption == "missing_account":
            del event["account_id"]
    return event


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Producing to topic '{TOPIC}' on {BOOTSTRAP_SERVERS}. Ctrl+C to stop.\n")

    event_num = 0
    try:
        while True:
            event_num += 1
            event = make_event(event_num)
            event = maybe_corrupt(event)

            # ~5% chance: resend a duplicate of a recent event instead.
            if recent_events and random.random() < 0.05:
                event = random.choice(recent_events)
                print(f"[DUPLICATE]     {event['transaction_id']}")
            # ~5% chance: hold this event back and send an OLDER one now,
            # simulating a late/out-of-order arrival.
            elif recent_events and random.random() < 0.05:
                late_event = random.choice(recent_events)
                producer.send(TOPIC, value=late_event)
                print(f"[LATE/OUT-OF-ORDER] {late_event['transaction_id']}  "
                      f"(event_time={late_event['event_time']})")

            producer.send(TOPIC, value=event)
            recent_events.append(event)
            if len(recent_events) > 30:
                recent_events.pop(0)

            tag = "amount" not in event and "[MISSING AMOUNT]" or ""
            print(f"[SENT #{event_num:04d}] {event.get('transaction_id', '???')[:8]} "
                  f"acct={event.get('account_id', '???')} "
                  f"amount={event.get('amount', 'N/A')} "
                  f"{'schema=v2' if 'device_fingerprint' in event else 'schema=v1'} {tag}")

            producer.flush()
            time.sleep(random.uniform(0.1, 0.4))

    except KeyboardInterrupt:
        print("\nStopping producer.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
