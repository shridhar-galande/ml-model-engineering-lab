"""Send requests to the service; report which model version answered and latency percentiles."""
import argparse
import collections
import time

import numpy as np
import requests
from sklearn.datasets import load_wine

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

# Reuse wine samples as valid requests without generating random input data.
X = load_wine().data
versions, latencies = collections.Counter(), []
# Print about 20 progress updates, even for smaller test runs.
progress_interval = max(1, args.n // 20)
start_time = time.perf_counter()
print(f"Starting {args.n} requests to {args.url}/predict", flush=True)

for index in range(args.n):
    payload = {"features": X[index % len(X)].tolist()}
    t0 = time.perf_counter()
    # Close each connection so repeated requests can be balanced across pods.
    try:
        response = requests.post(
            f"{args.url}/predict", json=payload, headers={"Connection": "close"}, timeout=5
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Request {index + 1}/{args.n} failed: {error}", flush=True)
        raise SystemExit(1) from error

    latencies.append((time.perf_counter() - t0) * 1000)
    versions[response.json()["model_version"]] += 1

    completed = index + 1
    if completed % progress_interval == 0 or completed == args.n:
        elapsed = time.perf_counter() - start_time
        print(
            f"Progress: {completed}/{args.n} requests "
            f"({100 * completed / args.n:.0f}%) in {elapsed:.1f}s",
            flush=True,
        )

# Summarize version distribution and latency after all requests complete.
print(f"\nRequests: {args.n}")
for v, c in sorted(versions.items()):
    print(f"  {v}: {c} responses ({100 * c / args.n:.1f}%)")
p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
print(f"Latency ms  p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}")
