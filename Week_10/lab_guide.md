# Lab: Real-Time Data Pipeline with Kafka

## What you're building

A small but real streaming pipeline: a **producer** publishes simulated
transaction events to **Kafka**, and a **consumer** reads them, checks their
quality, computes rolling (windowed) features per account, and applies a
simple real-time anomaly rule. This is the same shape as the "Streaming for
ML" and "Data Quality in Real-Time ML" sections, just small enough to run on
your laptop.

## Objectives

- Stand up a real Kafka broker and see producers/consumers/topics in action
- Watch batch-style thinking break down once data is continuous
- Implement freshness, completeness, and correctness checks on live data
- Turn a raw event stream into windowed, model-ready features
- See how a schema change (a new optional field) does — and doesn't — break
  a running consumer

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+
- The 5 files from this lab: `docker-compose.yml`, `requirements.txt`,
  `producer.py`, `consumer.py`, and this guide

## Setup

```bash
pip install -r requirements.txt
docker compose up -d
```

Give the broker about 10–15 seconds to finish starting before moving on. You
can check it's healthy with `docker compose ps` (status should say
`healthy`), or watch its logs with `docker compose logs -f kafka`.

> **Note:** this uses the official `apache/kafka` image plus
> [Kafka UI](https://github.com/provectus/kafka-ui) for a visual view of
> topics and messages. If you previously pulled an older Bitnami-based
> version of this file, re-copy `docker-compose.yml` — older Bitnami rolling
> tags like `bitnami/kafka:3.7` have since been removed from Docker Hub.

## Step 0 — Open the Kafka UI

Once both containers are up, open **http://localhost:8080** in your browser.

You should see one cluster (`local`) with status `Online`. This is a good
place to keep a tab open for the rest of the lab:

- **Topics** — once the producer creates `transactions`, click into it to
  see partitions, message count, and consumer group lag update live
- **Messages** tab on the topic — browse individual events, see their raw
  JSON, and jump to a specific offset or timestamp
- **Consumers** tab — watch the consumer group from `consumer.py` and its
  lag (how far behind the latest message it is) — a direct, visual way to
  see the freshness/latency ideas from the slides
- Later, when the producer starts sending `schema=v2` events, you can spot
  the new `device_fingerprint` field directly in a message's JSON here

Keep this tab open alongside your two terminals for the rest of the lab.

## Step 1 — Start the consumer

Open a terminal and run:

```bash
python consumer.py
```

It will sit and wait — there's no data yet. This is expected: unlike a
notebook, the consumer doesn't need data to already exist, it waits for it
to arrive.

## Step 2 — Start the producer

In a **second** terminal:

```bash
python producer.py
```

You'll see a steady stream of `[SENT #...]` lines, and occasional
`[DUPLICATE]` and `[LATE/OUT-OF-ORDER]` lines — the producer injects these on
purpose. Switch back to the consumer terminal and watch events arrive and get
scored in near real time.

## Step 3 — Read the data quality signal

Every ~10 seconds, the consumer prints a **Data Quality Summary**:

```json
{
  "total": 214,
  "duplicates": 9,
  "missing_fields": 6,
  "invalid_values": 8,
  "out_of_order": 11,
  "stale_over_sla": 2
}
```

Connect each number back to the session:

- `duplicates` — the same `transaction_id` seen more than once (a retry, or
  a consumer that reprocessed a message)
- `missing_fields` / `invalid_values` — completeness and correctness issues
- `out_of_order` — an event arriving after a *later* event from the same
  account already arrived
- `stale_over_sla` — events that blew past the freshness SLA defined at the
  top of `consumer.py` (`FRESHNESS_SLA_SECONDS`)

**Try it:** lower `FRESHNESS_SLA_SECONDS` to `1` and restart the consumer.
Watch `stale_over_sla` climb — this is exactly the freshness-vs-latency
trade-off from the slides: a tighter SLA catches more staleness, but you pay
for it in false positives if the pipeline just has ordinary network jitter.

## Step 4 — Watch the windowed features

Each line also prints the current rolling window for that account:

```
window(count=3, avg=612.40)
```

This is computed in `update_window()` / `window_features()` in
`consumer.py` — a sliding `WINDOW_SECONDS`-second window per `account_id`,
recomputed as each new event arrives. This is the "state" the session
discussed: a single event tells you almost nothing on its own, but the
window turns it into a feature a model can use.

**Try it:** change `WINDOW_SECONDS` from `60` to `20` and restart the
consumer. Notice how much noisier `count`/`avg` become with a shorter
window — narrower windows react faster but are less stable.

## Step 5 — Watch the live anomaly flags

Occasionally you'll see `[FLAGGED]` instead of `[ok  ]`, with a reason like:

```
[FLAGGED] a1b2c3d4 acct=acct_007  amount=1204.50 window(count=6, avg=210.11) ['amount 1204.50 is >4x rolling avg (210.11)']
```

`score_event()` is a deliberately simple, explainable rule — exactly the
kind of rule you'd reach for before investing in a full online model. The
comment in the code shows where you'd swap in an incremental classifier
(e.g. `river` or scikit-learn's `partial_fit`) once the pipeline mechanics
are solid.

## Step 6 — Observe schema evolution live

Leave both scripts running. After ~40 events, the producer starts adding an
optional `device_fingerprint` field (look for `schema=v2` in the consumer
output, versus `schema=v1` earlier in the run).

Notice the consumer **did not crash or need a restart** — it just reads the
field if present (`STEP E` in `consumer.py`). This is backward compatibility
in practice: the new schema is additive and optional, so old consumer logic
keeps working unmodified.

**Try it:** open `consumer.py` and add a line that uses
`event["device_fingerprint"]` unconditionally (no `.get()` or `in` check).
Restart the consumer and watch it crash on every `v1` event — this is what
happens when a consumer assumes a field will always be there instead of
treating the schema as something that evolves.

## Step 7 — Shut down

```bash
# Ctrl+C in both terminals, then:
docker compose down -v
```

## Discussion questions

1. Which data quality issue in the summary would you treat as **drop the
   event**, and which would you treat as **fix and keep**? Why?
2. If `stale_over_sla` stayed high no matter how the pipeline was tuned,
   would you invest in Flink-style event-at-a-time processing, or accept the
   current microbatch-like latency? What would you need to know to decide?
3. The schema change here was backward compatible (an optional field). Sketch
   an example of a *breaking* schema change for this event, and describe how
   a data contract would have caught it before it reached this consumer.
