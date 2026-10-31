# %% [markdown]
# # Lab 2 — Compression: Quantization & Pruning
#
# **Week 7: Model Standardization and Optimization**
#
# ## What you will do in this lab
# 1. Load (or regenerate) the baseline MobileNetV2 ONNX model from Lab 1.
# 2. Apply **ONNX INT8 dynamic quantization** (`quantize_dynamic`) — the
#    fastest path to a quantized model: no retraining, no calibration data.
# 3. Apply **TensorFlow Lite INT8 quantization** — TF's native compression
#    format, with a representative-dataset calibration step for better
#    accuracy than purely dynamic quantization.
# 4. Apply **structured pruning** on the Keras MobileNetV2 — zero out the
#    30% of filters in each Conv2D layer with the lowest L2 norm, then
#    export the pruned model to ONNX.
# 5. Benchmark and compare all four variants side-by-side: size on disk,
#    inference latency, and top-1 class consistency with the FP32 baseline.
#
# ## Why this matters
# Quantization and pruning are the "Compression" lever from the Week 7
# lecture. Each technique trades a small, controlled accuracy drop for
# a meaningful gain in speed and a reduction in model size:
#
#   FP32 baseline  →  INT8 quantization (~4× size reduction, faster math)
#   FP32 baseline  →  Pruning (zeros out unimportant filters, sparser ops)
#
# You will see how each approach behaves differently and why they are
# often combined in a real MLOps pipeline.
#
# ## Lab 2 compression variants
# | # | Name          | Method                                  |
# |---|---------------|-----------------------------------------|
# | 1 | FP32 Baseline | Lab 1 ONNX export, no compression       |
# | 2 | INT8 PTQ ONNX | ONNX Runtime `quantize_dynamic`         |
# | 3 | INT8 TFLite   | TF Lite INT8 with representative data   |
# | 4 | Pruned ONNX   | 30% structured filter pruning → ONNX   |

# %% [markdown]
# ## Step 0 — Imports and setup

# %%
import os
import sys
import time
import warnings
import tracemalloc

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf2onnx
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
LAB1_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "lab1_onnx_conversion", "artifacts")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE_ONNX   = os.path.join(LAB1_DIR,   "mobilenetv2.onnx")
INT8_ONNX       = os.path.join(OUTPUT_DIR,  "mobilenetv2_int8.onnx")
PRUNED_KERAS    = os.path.join(OUTPUT_DIR,  "mobilenetv2_pruned.keras")
PRUNED_ONNX     = os.path.join(OUTPUT_DIR,  "mobilenetv2_pruned.onnx")
TFLITE_PATH     = os.path.join(OUTPUT_DIR,  "mobilenetv2_int8.tflite")

# A standard ImageNet-style random input (used everywhere in this lab).
DUMMY_NP = np.random.randn(1, 224, 224, 3).astype(np.float32)
DUMMY_TF = tf.constant(DUMMY_NP)

print("Setup complete.")
print(f"TensorFlow  : {tf.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"Artifacts   : {OUTPUT_DIR}")

# %% [markdown]
# ## Step 1 — Load or regenerate the baseline ONNX model
#
# Lab 2 picks up exactly where Lab 1 finished. We try to load
# `mobilenetv2.onnx` from Lab 1's `artifacts/` folder. If it is not
# there yet (e.g. you are running Lab 2 standalone), we regenerate it
# automatically so this lab is self-contained.

# %%
def regenerate_baseline_onnx(path: str) -> None:
    """Export a fresh MobileNetV2 ONNX model to `path`.

    Called automatically when the Lab 1 artifact is missing.
    """
    print("  Generating MobileNetV2 ONNX model (Lab 1 artifact not found)...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = keras.applications.MobileNetV2(weights="imagenet")
    sig = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")]
    tf2onnx.convert.from_keras(m, input_signature=sig, opset=17, output_path=path)
    print(f"  Saved to {path}")


if not os.path.exists(BASELINE_ONNX):
    print("Lab 1 artifact not found — regenerating...")
    regenerate_baseline_onnx(BASELINE_ONNX)
else:
    print(f"Baseline ONNX loaded from: {BASELINE_ONNX}")

baseline_size_mb = os.path.getsize(BASELINE_ONNX) / (1024 * 1024)
print(f"Baseline FP32 ONNX size: {baseline_size_mb:.2f} MB")

# %% [markdown]
# ## Step 2 — ONNX INT8 dynamic quantization
#
# ### What is dynamic quantization?
# The model's **weights** are permanently converted from FP32 to INT8.
# **Activations** are quantized on-the-fly during each inference call —
# no calibration dataset is needed.
#
# ### Trade-offs
# - ✅ No retraining, no calibration data — fast to apply.
# - ✅ Weights shrink ~4× in memory.
# - ⚠️  Activation quantization at runtime adds a small overhead per call.
# - ⚠️  Accuracy drop is slightly larger than QAT, but usually acceptable.
#
# `quantize_dynamic` from `onnxruntime.quantization` does all of this in
# one function call. `QuantType.QUInt8` tells it to target unsigned 8-bit
# integers (the format ONNX Runtime's CPU kernels are optimized for).

# %%
print("Applying ONNX dynamic INT8 quantization...")

quantize_dynamic(
    model_input=BASELINE_ONNX,
    model_output=INT8_ONNX,
    weight_type=QuantType.QUInt8,   # quantize weights to unsigned 8-bit
)

int8_size_mb = os.path.getsize(INT8_ONNX) / (1024 * 1024)
size_reduction = (1 - int8_size_mb / baseline_size_mb) * 100
print(f"INT8 ONNX saved to : {INT8_ONNX}")
print(f"Size: {int8_size_mb:.2f} MB  (reduced {size_reduction:.1f}% from {baseline_size_mb:.2f} MB)")

# %% [markdown]
# ## Step 3 — TensorFlow Lite INT8 quantization
#
# ### What is TF Lite INT8 quantization?
# TF Lite's quantization takes the full Keras model and converts it to
# a `.tflite` file with both weights **and** activations in INT8. Unlike
# ONNX dynamic quantization, it uses a small **representative dataset**
# to calibrate the activation ranges before freezing them. This makes the
# quantized model more accurate at runtime because activation scaling
# factors are pre-tuned, not computed on-the-fly.
#
# ### Why compare both?
# They represent two different points in the ONNX/TFLite ecosystem:
# - ONNX INT8 `quantize_dynamic` = portable, easy, good enough for most cases.
# - TFLite INT8 with calibration = more accurate, targets mobile/edge devices.

# %%
print("Loading Keras MobileNetV2 for TFLite conversion...")
keras_model = keras.applications.MobileNetV2(weights="imagenet")


def representative_dataset():
    """Yield small batches of random FP32 inputs for TFLite INT8 calibration.

    In a real deployment you would use samples from your actual data
    distribution here. Random data is fine for a demo — it shows the
    mechanism without requiring a real dataset.
    We yield 50 samples, which is the commonly recommended minimum.
    """
    for _ in range(50):
        sample = np.random.randn(1, 224, 224, 3).astype(np.float32)
        yield [sample]


print("Running TFLite INT8 quantization (with representative-dataset calibration)...")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Tell the converter we want full-integer quantization (weights + activations).
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide the calibration data so activation ranges can be pre-computed.
converter.representative_dataset = representative_dataset

# Require that all ops can be quantized to INT8.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Force input and output tensors to stay as float32 so we can feed the
# same dummy_np input we use everywhere else without re-casting.
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

tflite_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
size_reduction_tfl = (1 - tflite_size_mb / baseline_size_mb) * 100
print(f"TFLite INT8 saved to: {TFLITE_PATH}")
print(f"Size: {tflite_size_mb:.2f} MB  (reduced {size_reduction_tfl:.1f}% from {baseline_size_mb:.2f} MB)")

# %% [markdown]
# ## Step 4 — Structured pruning on the Keras model
#
# ### What is structured pruning?
# Pruning removes weights that contribute little to the model's output.
# **Structured** pruning removes entire filters (3D weight tensors) from
# Conv2D layers, rather than individual scalar weights. This produces a
# sparse model that runs faster on standard hardware without needing
# special sparse-kernel support.
#
# ### Our approach: L2-norm filter pruning
# For each Conv2D and DepthwiseConv2D layer we:
# 1. Compute the L2 norm of each filter (a scalar measure of its "magnitude"
#    and therefore its importance to the output).
# 2. Rank filters from smallest (least important) to largest (most important).
# 3. Zero out the bottom 30% of filters by setting their weights to zero.
#
# Zeroing weights is called **soft** structured pruning: the architecture
# is unchanged (same layer shapes), but 30% of filters are now all-zero
# and contribute nothing to the output. This is straightforward to export
# to ONNX and benchmark, and shows the concept clearly. In production you
# would follow pruning with fine-tuning to recover any accuracy drop.
#
# ### Why 30%?
# 30% is a common starting point that typically gives a meaningful size/
# speed benefit with a small accuracy impact. The right number for a
# production model depends on an accuracy-vs-efficiency sweep.

# %%
def prune_model_filters(model: keras.Model, prune_fraction: float = 0.3) -> keras.Model:
    """Return a copy of `model` with the lowest-L2-norm filters zeroed out.

    Works on Conv2D and DepthwiseConv2D layers. Other layer types are left
    unchanged. The model's architecture (layer shapes) is not altered —
    only weight values change.

    Args:
        model:          A compiled or loaded Keras model.
        prune_fraction: Fraction of filters to zero out per layer (0–1).

    Returns:
        The same model object with pruned weights set in-place (also
        returned for convenience).
    """
    layers_pruned = 0
    filters_zeroed = 0
    filters_total = 0

    for layer in model.layers:
        # Only process convolutional layers that have learnable filters.
        is_conv2d       = isinstance(layer, keras.layers.Conv2D)
        is_depthwise    = isinstance(layer, keras.layers.DepthwiseConv2D)

        if not (is_conv2d or is_depthwise):
            continue   # BatchNorm, Dense, etc. — leave untouched

        weights = layer.get_weights()
        if not weights:
            continue  # no trainable weights (e.g. a frozen layer)

        kernel = weights[0]  # shape: (H, W, in_channels, out_channels)
                              # or (H, W, in_channels, depth_multiplier) for DW

        # Number of filters = last axis of the kernel tensor.
        n_filters = kernel.shape[-1]
        n_prune   = max(1, int(n_filters * prune_fraction))

        # Compute the L2 norm of each filter.
        # We reshape each filter to a vector and compute its Euclidean norm.
        # Shape: (H*W*in_channels, n_filters) -> norm per column = per filter.
        kernel_flat = kernel.reshape(-1, n_filters)          # (D, n_filters)
        l2_norms    = np.linalg.norm(kernel_flat, axis=0)   # (n_filters,)

        # Find the indices of the n_prune smallest-norm filters.
        prune_idx = np.argsort(l2_norms)[:n_prune]

        # Zero out those filters in the kernel array.
        kernel[..., prune_idx] = 0.0

        # Write the modified kernel back to the layer.
        weights[0] = kernel
        layer.set_weights(weights)

        layers_pruned   += 1
        filters_zeroed  += n_prune
        filters_total   += n_filters

    print(f"  Pruned {layers_pruned} conv layers | "
          f"zeroed {filters_zeroed}/{filters_total} filters "
          f"({filters_zeroed/filters_total*100:.1f}% of all conv filters)")
    return model


print("Loading Keras model for pruning (this is a separate instance)...")
pruned_model = keras.applications.MobileNetV2(weights="imagenet")

print(f"Applying structured filter pruning (30% per Conv/DepthwiseConv2D layer)...")
pruned_model = prune_model_filters(pruned_model, prune_fraction=0.30)

# Save the pruned Keras model to disk so Lab 3 can reload it without
# needing to re-run the pruning step.
pruned_model.save(PRUNED_KERAS)
print(f"Pruned Keras model saved: {PRUNED_KERAS}")

# %% [markdown]
# ## Step 5 — Export the pruned Keras model to ONNX
#
# We convert the pruned Keras model to ONNX just like we did in Lab 1.
# The exported ONNX graph will reflect the zeroed-out filters, giving
# ONNX Runtime an opportunity to skip zero-multiplication at inference
# time (depending on the runtime version and CPU's sparse-math support).

# %%
print("Exporting pruned Keras model to ONNX...")

input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")]

tf2onnx.convert.from_keras(
    pruned_model,
    input_signature=input_signature,
    opset=17,
    output_path=PRUNED_ONNX,
)

pruned_size_mb = os.path.getsize(PRUNED_ONNX) / (1024 * 1024)
size_reduction_pruned = (1 - pruned_size_mb / baseline_size_mb) * 100
print(f"Pruned ONNX saved to: {PRUNED_ONNX}")
print(f"Size: {pruned_size_mb:.2f} MB  (delta {size_reduction_pruned:+.1f}% vs baseline)")

# %% [markdown]
# ## Step 6 — Benchmark all four variants
#
# Now we run the same latency benchmark (warmup + averaged runs) across
# all four model variants. For ONNX models (baseline, INT8, pruned) we
# use ONNX Runtime. For the TFLite model we use TensorFlow's own Lite
# interpreter, which is the runtime TFLite models are designed for.
#
# We also check that each compressed model agrees with the FP32 baseline
# on the predicted class for our dummy input (a "top-1 match" check).

# %%
N_WARMUP = 10
N_RUNS   = 50


def benchmark_ort_session(onnx_path: str,
                           input_array: np.ndarray,
                           n_warmup: int = N_WARMUP,
                           n_runs:   int = N_RUNS) -> float:
    """Create an ONNX Runtime session and return mean latency in ms."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for _ in range(n_warmup):
        sess.run([out_name], {in_name: input_array})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run([out_name], {in_name: input_array})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies)), sess


def benchmark_tflite(tflite_path: str,
                     input_array: np.ndarray,
                     n_warmup: int = N_WARMUP,
                     n_runs:   int = N_RUNS) -> float:
    """Load a TFLite model and return mean latency in ms."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    for _ in range(n_warmup):
        interp.set_tensor(in_idx, input_array)
        interp.invoke()

    latencies = []
    for _ in range(n_runs):
        interp.set_tensor(in_idx, input_array)
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies)), interp


print("Benchmarking all four variants — this takes a minute...\n")

# ── 1. FP32 baseline ───────────────────────────────────────────────────────
print("  [1/4] FP32 baseline ONNX...")
baseline_latency, baseline_sess = benchmark_ort_session(BASELINE_ONNX, DUMMY_NP)
baseline_out_name = baseline_sess.get_outputs()[0].name
baseline_in_name  = baseline_sess.get_inputs()[0].name
fp32_output = baseline_sess.run([baseline_out_name], {baseline_in_name: DUMMY_NP})[0]
fp32_pred   = int(np.argmax(fp32_output, axis=1)[0])
print(f"     Latency : {baseline_latency:.2f} ms | FP32 pred class: {fp32_pred}")

# ── 2. INT8 PTQ (ONNX quantize_dynamic) ────────────────────────────────────
print("  [2/4] INT8 PTQ (ONNX dynamic quantization)...")
int8_latency, int8_sess = benchmark_ort_session(INT8_ONNX, DUMMY_NP)
int8_out_name = int8_sess.get_outputs()[0].name
int8_in_name  = int8_sess.get_inputs()[0].name
int8_output = int8_sess.run([int8_out_name], {int8_in_name: DUMMY_NP})[0]
int8_pred   = int(np.argmax(int8_output, axis=1)[0])
int8_match  = (int8_pred == fp32_pred)
print(f"     Latency : {int8_latency:.2f} ms | pred class: {int8_pred} | matches FP32: {int8_match}")

# ── 3. TFLite INT8 (with calibration) ─────────────────────────────────────
print("  [3/4] TFLite INT8...")
tflite_latency, tfl_interp = benchmark_tflite(TFLITE_PATH, DUMMY_NP)
tfl_in_idx  = tfl_interp.get_input_details()[0]["index"]
tfl_out_idx = tfl_interp.get_output_details()[0]["index"]
tfl_interp.set_tensor(tfl_in_idx, DUMMY_NP)
tfl_interp.invoke()
tfl_output = tfl_interp.get_tensor(tfl_out_idx)
tfl_pred   = int(np.argmax(tfl_output, axis=1)[0])
tfl_match  = (tfl_pred == fp32_pred)
print(f"     Latency : {tflite_latency:.2f} ms | pred class: {tfl_pred} | matches FP32: {tfl_match}")

# ── 4. Pruned ONNX ─────────────────────────────────────────────────────────
print("  [4/4] Pruned ONNX (30% filter pruning)...")
pruned_latency, pruned_sess = benchmark_ort_session(PRUNED_ONNX, DUMMY_NP)
pruned_out_name = pruned_sess.get_outputs()[0].name
pruned_in_name  = pruned_sess.get_inputs()[0].name
pruned_output = pruned_sess.run([pruned_out_name], {pruned_in_name: DUMMY_NP})[0]
pruned_pred   = int(np.argmax(pruned_output, axis=1)[0])
pruned_match  = (pruned_pred == fp32_pred)
print(f"     Latency : {pruned_latency:.2f} ms | pred class: {pruned_pred} | matches FP32: {pruned_match}")

# %% [markdown]
# ## Step 7 — Summary table
#
# The four numbers that matter for a deployment decision:
# - **Size on disk** — how much storage and memory does the model need?
# - **Latency** — how fast is a single forward pass?
# - **Speedup** — improvement over the FP32 baseline.
# - **Top-1 match** — does the compressed model agree with the baseline?

# %%
results = [
    ("FP32 Baseline",    baseline_size_mb,  baseline_latency, True),
    ("INT8 PTQ (ONNX)",  int8_size_mb,      int8_latency,     int8_match),
    ("INT8 TFLite",      tflite_size_mb,    tflite_latency,   tfl_match),
    ("Pruned ONNX",      pruned_size_mb,    pruned_latency,   pruned_match),
]

print()
print("=" * 70)
print("LAB 2 SUMMARY — Compression Comparison (MobileNetV2, batch=1)")
print("=" * 70)
header = f"{'Variant':<22} {'Size (MB)':>10} {'Latency (ms)':>13} {'Speedup':>9} {'Top-1 match':>12}"
print(header)
print("-" * 70)
for name, size, lat, match in results:
    speedup = baseline_latency / lat if lat > 0 else float("nan")
    print(f"{name:<22} {size:>10.2f} {lat:>13.2f} {speedup:>9.2f}x {str(match):>12}")
print("=" * 70)

print()
print("Artifacts written:")
for path in [INT8_ONNX, TFLITE_PATH, PRUNED_KERAS, PRUNED_ONNX]:
    print(f"  - {path}")
print()
print("Next: Lab 3 loads all four artifacts to run a full benchmarking "
      "dashboard with p50/p95/p99 latency, throughput, and memory profiling.")
