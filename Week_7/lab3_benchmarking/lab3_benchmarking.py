# %% [markdown]
# # Lab 3 — Benchmarking Dashboard
#
# **Week 7: Model Standardization and Optimization**
#
# ## What you will do in this lab
# 1. Auto-detect all model artifacts from Labs 1 & 2, regenerating any
#    that are missing so this lab runs standalone.
# 2. Run **proper latency benchmarking** with warmup and p50 / p95 / p99
#    percentile reporting — not just a mean.
# 3. Sweep **throughput** across batch sizes [1, 4, 8, 16] to understand
#    how each model scales under concurrent load.
# 4. Profile **peak memory usage** with Python's `tracemalloc`.
# 5. Produce a **matplotlib dashboard** (4 panels: latency percentiles,
#    model size, throughput, summary table).
# 6. Print a **SLO-based production recommendation** that maps each model
#    to real deployment guidance.
#
# ## Why percentiles, not means?
# In production, we care about *tail latency* — the slowest requests.
# If your p99 latency is 500 ms but your mean is 50 ms, 1% of users are
# seeing a terrible experience. SLOs (service-level objectives) are almost
# always written as "p95 < 100 ms" or "p99 < 200 ms", not "mean < X".
# This lab uses the same metric that production systems actually track.
#
# ## Lab 3 models benchmarked
# | # | Name          | Source                         |
# |---|---------------|--------------------------------|
# | 1 | FP32 Baseline | Lab 1 ONNX export              |
# | 2 | INT8 PTQ ONNX | Lab 2 `quantize_dynamic`       |
# | 3 | INT8 TFLite   | Lab 2 TFLite INT8 converter    |
# | 4 | Pruned ONNX   | Lab 2 structured filter pruning|

# %% [markdown]
# ## Step 0 — Imports and setup

# %%
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts and notebooks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

# ── Artifact paths ─────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
LAB1_ART  = os.path.join(BASE, "..", "lab1_onnx_conversion", "artifacts")
LAB2_ART  = os.path.join(BASE, "..", "lab2_compression",     "artifacts")
LAB3_ART  = os.path.join(BASE, "artifacts")
os.makedirs(LAB1_ART, exist_ok=True)
os.makedirs(LAB2_ART, exist_ok=True)
os.makedirs(LAB3_ART, exist_ok=True)

BASELINE_ONNX = os.path.join(LAB1_ART, "mobilenetv2.onnx")
INT8_ONNX     = os.path.join(LAB2_ART, "mobilenetv2_int8.onnx")
PRUNED_ONNX   = os.path.join(LAB2_ART, "mobilenetv2_pruned.onnx")
TFLITE_PATH   = os.path.join(LAB2_ART, "mobilenetv2_int8.tflite")
DASHBOARD_PNG = os.path.join(LAB3_ART, "benchmark_dashboard.png")

print("Setup complete.")
print(f"TensorFlow  : {tf.__version__}")
print(f"ONNX Runtime: {ort.__version__}")

# %% [markdown]
# ## Step 1 — Auto-detect and regenerate missing artifacts
#
# Labs 3 can be run standalone without having completed Labs 1 and 2
# first. We check which model files exist and regenerate any that are
# missing using the same logic from Labs 1 and 2.

# %%
def build_baseline_onnx(path: str) -> None:
    """Export MobileNetV2 FP32 ONNX (Lab 1 logic)."""
    print(f"  Generating FP32 baseline ONNX → {path}")
    m   = keras.applications.MobileNetV2(weights="imagenet")
    sig = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")]
    tf2onnx.convert.from_keras(m, input_signature=sig, opset=17, output_path=path)


def build_int8_onnx(baseline_path: str, out_path: str) -> None:
    """Apply ONNX dynamic INT8 quantization (Lab 2 logic)."""
    print(f"  Generating INT8 PTQ ONNX → {out_path}")
    quantize_dynamic(baseline_path, out_path, weight_type=QuantType.QUInt8)


def build_tflite(out_path: str) -> None:
    """Export TFLite INT8 model (Lab 2 logic)."""
    print(f"  Generating TFLite INT8 → {out_path}")
    m = keras.applications.MobileNetV2(weights="imagenet")

    def rep_ds():
        for _ in range(50):
            yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations             = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset    = rep_ds
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type      = tf.float32
    c.inference_output_type     = tf.float32
    with open(out_path, "wb") as f:
        f.write(c.convert())


def prune_and_export(out_onnx: str) -> None:
    """Apply 30% filter pruning and export to ONNX (Lab 2 logic)."""
    print(f"  Generating Pruned ONNX → {out_onnx}")
    m = keras.applications.MobileNetV2(weights="imagenet")
    for layer in m.layers:
        if not isinstance(layer, (keras.layers.Conv2D,
                                  keras.layers.DepthwiseConv2D)):
            continue
        ws = layer.get_weights()
        if not ws:
            continue
        k       = ws[0]
        n_f     = k.shape[-1]
        n_prune = max(1, int(n_f * 0.3))
        flat    = k.reshape(-1, n_f)
        norms   = np.linalg.norm(flat, axis=0)
        k[..., np.argsort(norms)[:n_prune]] = 0.0
        ws[0] = k
        layer.set_weights(ws)
    sig = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")]
    tf2onnx.convert.from_keras(m, input_signature=sig, opset=17, output_path=out_onnx)


print("Checking for Lab 1 & 2 artifacts...")
missing = []
if not os.path.exists(BASELINE_ONNX): missing.append("baseline")
if not os.path.exists(INT8_ONNX):     missing.append("int8_onnx")
if not os.path.exists(TFLITE_PATH):   missing.append("tflite")
if not os.path.exists(PRUNED_ONNX):   missing.append("pruned")

if not missing:
    print("All artifacts found — skipping regeneration.")
else:
    print(f"Missing: {missing}. Regenerating...")
    if "baseline"  in missing: build_baseline_onnx(BASELINE_ONNX)
    if "int8_onnx" in missing: build_int8_onnx(BASELINE_ONNX, INT8_ONNX)
    if "tflite"    in missing: build_tflite(TFLITE_PATH)
    if "pruned"    in missing: prune_and_export(PRUNED_ONNX)
    print("Regeneration complete.")

# %% [markdown]
# ## Step 2 — Latency benchmarking: warmup + p50 / p95 / p99
#
# ### Why 200 runs?
# With 200 timed samples per model, p99 is estimated from the top 2
# slowest runs — statistically meaningful without being prohibitively slow.
#
# ### Why separate warmup?
# The first few inference calls are always slower: memory pages need to
# be loaded, JIT compilation may happen inside the runtime, CPU caches are
# cold. Warmup runs let the runtime reach its "steady state" before we
# start timing.

# %%
N_WARMUP = 10
N_RUNS   = 200   # 200 samples gives meaningful p99 from only 2 outliers

# We benchmark at batch size 1 for the latency percentile analysis,
# then sweep multiple batch sizes in Step 3 for throughput.
DUMMY_B1 = np.random.randn(1, 224, 224, 3).astype(np.float32)


def latency_percentiles(times_ms: list) -> dict:
    """Compute p50, p95, p99 from a list of per-call latencies in ms."""
    arr = np.array(times_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


def run_ort_latency(onnx_path: str, input_np: np.ndarray) -> dict:
    """Benchmark an ONNX Runtime model, return percentile latencies in ms."""
    sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for _ in range(N_WARMUP):
        sess.run([out_name], {in_name: input_np})

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run([out_name], {in_name: input_np})
        times.append((time.perf_counter() - t0) * 1000.0)

    return latency_percentiles(times)


def run_tflite_latency(tflite_path: str, input_np: np.ndarray) -> dict:
    """Benchmark a TFLite model, return percentile latencies in ms."""
    interp  = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    for _ in range(N_WARMUP):
        interp.set_tensor(in_idx, input_np)
        interp.invoke()

    times = []
    for _ in range(N_RUNS):
        interp.set_tensor(in_idx, input_np)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)

    return latency_percentiles(times)


print("Running latency benchmarks (200 runs each, batch size 1)...")
models_meta = [
    ("FP32 Baseline",   "ort",     BASELINE_ONNX),
    ("INT8 PTQ (ONNX)", "ort",     INT8_ONNX),
    ("INT8 TFLite",     "tflite",  TFLITE_PATH),
    ("Pruned ONNX",     "ort",     PRUNED_ONNX),
]

latency_results = {}
for name, runtime, path in models_meta:
    print(f"  {name}...", end=" ", flush=True)
    if runtime == "ort":
        perc = run_ort_latency(path, DUMMY_B1)
    else:
        perc = run_tflite_latency(path, DUMMY_B1)
    latency_results[name] = perc
    print(f"p50={perc['p50']:.1f}ms  p95={perc['p95']:.1f}ms  p99={perc['p99']:.1f}ms")

# %% [markdown]
# ## Step 3 — Throughput sweep across batch sizes
#
# Throughput = (batch_size) / (total time for that batch in seconds)
# expressed in **images per second**.
#
# A faster runtime will handle larger batches proportionally better.
# This sweep shows how each model scales as concurrent traffic increases —
# critical for deciding how many inference replicas you need at peak load.
#
# We skip TFLite here: TFLite's Interpreter is designed for single-sample
# mobile inference and does not natively accept variable batch sizes.

# %%
BATCH_SIZES = [1, 4, 8, 16]
N_WARMUP_TP = 5
N_RUNS_TP   = 20   # fewer runs here — large batches are slow on CPU


def throughput_ort(onnx_path: str, batch_size: int) -> float:
    """Return images/second for an ONNX Runtime model at a given batch size."""
    x        = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
    sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for _ in range(N_WARMUP_TP):
        sess.run([out_name], {in_name: x})

    t0 = time.perf_counter()
    for _ in range(N_RUNS_TP):
        sess.run([out_name], {in_name: x})
    elapsed = time.perf_counter() - t0

    # total images processed = batch_size * N_RUNS_TP
    return (batch_size * N_RUNS_TP) / elapsed


print("\nRunning throughput sweep (batch sizes: 1, 4, 8, 16)...")
throughput_results = {}   # {model_name: {batch_size: imgs/sec}}

ort_models = [
    ("FP32 Baseline", BASELINE_ONNX),
    ("INT8 PTQ (ONNX)", INT8_ONNX),
    ("Pruned ONNX", PRUNED_ONNX),
]

for name, path in ort_models:
    throughput_results[name] = {}
    print(f"  {name}...")
    for bs in BATCH_SIZES:
        tp = throughput_ort(path, bs)
        throughput_results[name][bs] = tp
        print(f"    batch={bs:>2d}: {tp:6.1f} img/s")

# %% [markdown]
# ## Step 4 — Memory profiling with `tracemalloc`
#
# `tracemalloc` is Python's built-in memory tracer. It records every
# allocation made during a block of code and reports the peak usage.
#
# We measure peak memory used *while running inference* (not just model
# load): this is what the inference server's container needs at runtime.
#
# Note: `tracemalloc` measures Python-side allocations. Native C++
# allocations inside the ONNX Runtime or TFLite C library are not
# captured. These numbers are therefore a lower bound on real usage, but
# they are still useful for comparing models relative to each other.

# %%
def measure_peak_memory_ort(onnx_path: str,
                              input_np: np.ndarray,
                              n_runs: int = 10) -> float:
    """Return peak Python-side memory in MB during ORT inference."""
    sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    tracemalloc.start()
    for _ in range(n_runs):
        sess.run([out_name], {in_name: input_np})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)   # bytes → MB


def measure_peak_memory_tflite(tflite_path: str,
                                 input_np: np.ndarray,
                                 n_runs: int = 10) -> float:
    """Return peak Python-side memory in MB during TFLite inference."""
    interp  = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    tracemalloc.start()
    for _ in range(n_runs):
        interp.set_tensor(in_idx, input_np)
        interp.invoke()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


print("\nProfiling peak inference memory...")
memory_results = {}
for name, runtime, path in models_meta:
    if runtime == "ort":
        mem = measure_peak_memory_ort(path, DUMMY_B1)
    else:
        mem = measure_peak_memory_tflite(path, DUMMY_B1)
    memory_results[name] = mem
    print(f"  {name:<22}: {mem:.2f} MB peak (Python-side)")

# %% [markdown]
# ## Step 5 — Matplotlib benchmarking dashboard
#
# A 4-panel figure that gives an at-a-glance overview of all models:
# - **Panel 1**: p50 / p95 / p99 latency per model (grouped bar chart).
# - **Panel 2**: model size on disk (horizontal bar chart).
# - **Panel 3**: throughput vs batch size (line chart, ONNX models only).
# - **Panel 4**: summary table (latency, size, memory, speedup).

# %%
MODEL_NAMES  = [n for n, _, _ in models_meta]
COLORS       = ["#2a6496", "#3aafa9", "#d9534f", "#5cb85c"]
COLOR_MAP    = dict(zip(MODEL_NAMES, COLORS))
PERCENTILE_C = {"p50": "#2a6496", "p95": "#f0ad4e", "p99": "#d9534f"}

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#f7f9fa")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel 1: latency percentiles ───────────────────────────────────────────
ax1  = fig.add_subplot(gs[0, 0])
pcts = ["p50", "p95", "p99"]
x    = np.arange(len(MODEL_NAMES))
w    = 0.24
for i, pct in enumerate(pcts):
    vals = [latency_results[n][pct] for n in MODEL_NAMES]
    bars = ax1.bar(x + i * w, vals, width=w, label=pct,
                   color=PERCENTILE_C[pct], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#333")

ax1.set_title("Latency Percentiles (batch=1)", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Latency (ms)")
ax1.set_xticks(x + w)
ax1.set_xticklabels(MODEL_NAMES, rotation=18, ha="right", fontsize=9)
ax1.legend(title="Percentile", fontsize=9)
ax1.set_facecolor("#ffffff")
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel 2: model size on disk ────────────────────────────────────────────
ax2      = fig.add_subplot(gs[0, 1])
sizes_mb = {
    "FP32 Baseline":   os.path.getsize(BASELINE_ONNX) / 1e6,
    "INT8 PTQ (ONNX)": os.path.getsize(INT8_ONNX)     / 1e6,
    "INT8 TFLite":     os.path.getsize(TFLITE_PATH)   / 1e6,
    "Pruned ONNX":     os.path.getsize(PRUNED_ONNX)   / 1e6,
}
bar_colors = [COLOR_MAP[n] for n in MODEL_NAMES]
bars2 = ax2.barh(MODEL_NAMES,
                 [sizes_mb[n] for n in MODEL_NAMES],
                 color=bar_colors, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars2, [sizes_mb[n] for n in MODEL_NAMES]):
    ax2.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f} MB", va="center", fontsize=9, color="#333")
ax2.set_title("Model Size on Disk", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Size (MB)")
ax2.set_facecolor("#ffffff")
ax2.grid(axis="x", linestyle="--", alpha=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Panel 3: throughput vs batch size ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
tp_colors = {
    "FP32 Baseline":   COLORS[0],
    "INT8 PTQ (ONNX)": COLORS[1],
    "Pruned ONNX":     COLORS[3],
}
for name, tp_dict in throughput_results.items():
    xs = sorted(tp_dict.keys())
    ys = [tp_dict[b] for b in xs]
    ax3.plot(xs, ys, marker="o", label=name, color=tp_colors[name],
             linewidth=2, markersize=7)
    for b, y in zip(xs, ys):
        ax3.annotate(f"{y:.0f}", (b, y),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=8, color=tp_colors[name])

ax3.set_title("Throughput vs Batch Size\n(ONNX models, CPU)",
              fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Batch size")
ax3.set_ylabel("Throughput (images / sec)")
ax3.set_xticks(BATCH_SIZES)
ax3.legend(fontsize=9)
ax3.set_facecolor("#ffffff")
ax3.grid(linestyle="--", alpha=0.4)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel 4: summary table ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

fp32_lat   = latency_results["FP32 Baseline"]["p50"]
table_data = []
col_labels = ["Variant", "Size (MB)", "p50 (ms)", "p95 (ms)", "p99 (ms)",
              "Mem (MB)", "Speedup"]

for name in MODEL_NAMES:
    lr   = latency_results[name]
    mem  = memory_results[name]
    size = sizes_mb[name]
    spd  = fp32_lat / lr["p50"]
    table_data.append([
        name,
        f"{size:.1f}",
        f"{lr['p50']:.1f}",
        f"{lr['p95']:.1f}",
        f"{lr['p99']:.1f}",
        f"{mem:.1f}",
        f"{spd:.2f}x",
    ])

tbl = ax4.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.85)

# Style header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2a6496")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Alternating row shading
for i in range(1, len(MODEL_NAMES) + 1):
    bg = "#eaf3fb" if i % 2 == 0 else "#ffffff"
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(bg)

ax4.set_title("Full Comparison Summary", fontsize=13, fontweight="bold", pad=10)

# ── Main title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Week 7 · Model Standardization & Optimization — Benchmark Dashboard\n"
    "MobileNetV2  ·  CPU Inference  ·  Batch=1 (latency/memory)  ·  Batch=[1,4,8,16] (throughput)",
    fontsize=12, fontweight="bold", y=0.99, color="#1a1a2e"
)

plt.savefig(DASHBOARD_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nDashboard saved: {DASHBOARD_PNG}")

# %% [markdown]
# ## Step 6 — SLO-based production recommendation
#
# In the real world, the choice between model variants is driven by your
# SLO (service-level objective). Below we evaluate each model against
# two common thresholds:
# - **Latency SLO**: p95 latency must be under 50 ms (a reasonable
#   real-time API budget for a CPU-only deployment).
# - **Size SLO**: model must fit in under 10 MB (a common edge/mobile
#   constraint).
#
# The recommendation block maps each model to a concrete deployment
# context based on these checks.

# %%
LATENCY_SLO_MS   = 50.0   # p95 < 50 ms — real-time API, CPU deployment
SIZE_SLO_MB      = 10.0   # < 10 MB on disk — edge/mobile deployment

print()
print("=" * 70)
print("LAB 3 — PRODUCTION RECOMMENDATION (SLO-BASED)")
print(f"  SLO targets: p95 latency < {LATENCY_SLO_MS} ms | size < {SIZE_SLO_MB} MB")
print("=" * 70)

recommendations = {
    "FP32 Baseline":
        "Best for: development baseline and accuracy reference. Not recommended "
        "for cost-sensitive production — use one of the compressed variants instead.",
    "INT8 PTQ (ONNX)":
        "Best for: CPU deployments where quantization is applied without retraining. "
        "Lowest disk footprint of the ONNX variants; good first compression step.",
    "INT8 TFLite":
        "Best for: mobile and edge devices (Android, IoT, embedded systems). "
        "Smallest size and XNNPACK-accelerated latency. Use the TFLite runtime.",
    "Pruned ONNX":
        "Best for: cloud CPU deployments where you want to reduce compute without "
        "changing the runtime stack. Combine with quantization for additive gains.",
}

for name in MODEL_NAMES:
    lr      = latency_results[name]
    size    = sizes_mb[name]
    lat_ok  = lr["p95"] <= LATENCY_SLO_MS
    size_ok = size       <= SIZE_SLO_MB

    lat_flag  = "✓ PASS" if lat_ok  else "✗ FAIL"
    size_flag = "✓ PASS" if size_ok else "✗ FAIL"

    print(f"\n  {name}")
    print(f"    p95 latency: {lr['p95']:>6.1f} ms  [{lat_flag}]  (SLO: < {LATENCY_SLO_MS} ms)")
    print(f"    Size:        {size:>6.1f} MB  [{size_flag}]  (SLO: < {SIZE_SLO_MB} MB)")
    print(f"    → {recommendations[name]}")

print()
print("=" * 70)
print("\nArtifacts written:")
print(f"  - {DASHBOARD_PNG}")
print()
print("Lab 3 complete. All three labs together cover the full pipeline:")
print("  Train → Compress → Export to ONNX → Optimized Runtime → Production")
