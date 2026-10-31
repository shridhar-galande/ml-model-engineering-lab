# %% [markdown]
# # Lab 1 — Keras (TensorFlow) to ONNX Conversion
#
# **Week 7: Model Standardization and Optimization**
#
# ## What you will do in this lab
# 1. Load a pretrained MobileNetV2 model using `tf.keras.applications`.
# 2. Measure baseline Keras inference latency (before any conversion).
# 3. Export the model to ONNX using `tf2onnx`, with a dynamic batch axis so
#    the exported graph can accept any batch size at inference time, not
#    just the batch size used during export.
# 4. Validate the exported ONNX graph using `onnx.checker` — this confirms
#    the graph is well-formed before we try to run it anywhere.
# 5. (Optional) Apply graph-level optimizations using `onnxoptimizer`
#    (operator fusion, redundant node elimination).
# 6. Run the ONNX model using ONNX Runtime and compare its output against
#    the original Keras output, to make sure the conversion did not
#    silently change the model's behaviour.
# 7. Print a clean summary comparing Keras vs ONNX on model size and
#    inference latency.
#
# ## Why this matters
# A trained Keras model is tied to the TensorFlow runtime and Python. ONNX
# gives us a standard, framework-neutral graph format that many runtimes
# (ONNX Runtime, TensorRT, OpenVINO, mobile runtimes) can all read. This is
# the "export to a standard format" step from the Week 7 lecture pipeline:
#
#     Train -> Compress -> Export to ONNX -> Optimized Runtime -> Production
#
# This lab covers the "Export to ONNX" step, plus a first look at running
# the exported model with an optimized runtime (ONNX Runtime).
#
# ## A note on the model choice
# The lecture references "ResNet-18" as a small, easy-to-reason-about CNN.
# This lab uses **MobileNetV2** instead of a ResNet variant: it is a
# pretrained, ImageNet-trained CNN built into `tf.keras.applications` (just
# like ResNet-18 would be in `torchvision`), but it is purpose-built to be
# small and fast — a closer match to the lecture's "edge/mobile-friendly"
# footprint discussion, and lighter to export, optimize, and benchmark on a
# typical laptop. Everything else about the workflow — export, validation,
# optimization, benchmarking — is identical regardless of which CNN you
# plug in.

# %% [markdown]
# ## Step 0 — Imports and setup
#
# We import everything up front so it's easy to see exactly what this lab
# depends on. If any of these imports fail, check `requirements.txt` in the
# `w7_lab` root folder and the Troubleshooting section of the README.

# %%
import os
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf2onnx
import onnx
import onnxruntime as ort

# onnxoptimizer is optional — we guard the import so the lab still runs
# end-to-end even if it is not installed. See README for install steps.
try:
    import onnxoptimizer
    ONNXOPTIMIZER_AVAILABLE = True
except ImportError:
    ONNXOPTIMIZER_AVAILABLE = False

warnings.filterwarnings("ignore")  # keep output clean for a teaching demo
tf.get_logger().setLevel("ERROR")  # quiet TensorFlow's own info/warning logs

# A fixed random seed makes the dummy input reproducible across runs, which
# makes it easier to compare numbers between learners.
tf.random.set_seed(42)
np.random.seed(42)

# All artifacts this lab produces (the .onnx file, etc.) are written here.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
ONNX_PATH = os.path.join(OUTPUT_DIR, "mobilenetv2.onnx")
ONNX_OPTIMIZED_PATH = os.path.join(OUTPUT_DIR, "mobilenetv2_optimized.onnx")

print("Setup complete.")
print(f"TensorFlow version: {tf.__version__}")
print(f"tf2onnx version    : {tf2onnx.__version__}")
print(f"ONNX version       : {onnx.__version__}")
print(f"ONNX Runtime ver   : {ort.__version__}")
print(f"onnxoptimizer      : {'available' if ONNXOPTIMIZER_AVAILABLE else 'NOT installed (optional step will be skipped)'}")

# %% [markdown]
# ## Step 1 — Load a pretrained MobileNetV2
#
# We use `tf.keras.applications`'s pretrained MobileNetV2, a CNN that's a
# good stand-in for "a model someone trained and now needs to deploy."
# Keras models loaded this way are already in inference mode by default
# (no separate `eval()` call is needed — Keras tracks training vs inference
# behaviour per-call via the `training=False` argument, which is what we
# use throughout this lab).

# %%
print("Loading pretrained MobileNetV2...")
model = keras.applications.MobileNetV2(weights="imagenet")

# A standard ImageNet-style input: batch of 1, 224x224, 3 colour channels.
# Note Keras uses channels-last (NHWC) ordering for image inputs.
dummy_input = tf.random.normal((1, 224, 224, 3))

print("Model loaded.")
print(f"Dummy input shape: {tuple(dummy_input.shape)}")

# %% [markdown]
# ## Step 2 — Baseline Keras inference timing
#
# Before converting anything, we need a baseline number to compare against.
# Without a baseline, "ONNX Runtime is fast" is a meaningless claim — fast
# *compared to what*?
#
# Two things we do carefully here, both standard practice for any latency
# measurement:
# - **Warmup runs**: the first few forward passes are often slower due to
#   lazy graph tracing/initialization. We run a few throwaway iterations
#   first and don't count them.
# - **Multiple timed runs**: a single run is noisy. We average over many
#   runs to get a stable number.

# %%
def benchmark_keras(model, input_tensor, n_warmup=10, n_runs=50):
    """Run a simple warmup + averaged-latency benchmark for a Keras model.

    Returns the mean latency in milliseconds per forward pass.
    """
    # Warmup: let any lazy tracing/setup happen before we start timing.
    for _ in range(n_warmup):
        _ = model(input_tensor, training=False)

    # Timed runs.
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(input_tensor, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # seconds -> ms

    return float(np.mean(latencies))


keras_latency_ms = benchmark_keras(model, dummy_input)
print(f"Baseline Keras latency: {keras_latency_ms:.3f} ms per forward pass (batch size 1)")

# %% [markdown]
# ## Step 3 — Export to ONNX with a dynamic batch axis
#
# `tf2onnx.convert.from_keras()` traces the model's computation graph and
# converts it to ONNX. A few choices matter here:
#
# - **`opset=17`**: ONNX "opsets" are versioned sets of supported
#   operators, similar to a language standard version. Opset 17 is a recent,
#   widely-supported version as of this course — recent enough to support
#   modern ops, old enough to be broadly compatible with runtimes.
# - **`input_signature` with `None` batch dimension**: by default, tracing
#   would hard-code the input shape used during conversion (here, batch
#   size 1). In production we rarely want to be locked to one batch size —
#   we may batch multiple requests together for throughput. Declaring the
#   batch dimension as `None` in the input signature lets the exported
#   model accept *any* batch size at inference time.

# %%
print("Exporting model to ONNX...")

input_signature = [
    tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")
]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=17,
    output_path=ONNX_PATH,
)

onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"Exported ONNX model to: {ONNX_PATH}")
print(f"ONNX file size: {onnx_size_mb:.2f} MB")

# %% [markdown]
# ## Step 4 — Validate the exported graph with `onnx.checker`
#
# Exporting can "succeed" (no exception raised) while still producing a
# malformed graph in edge cases. `onnx.checker.check_model()` runs ONNX's
# own structural validation — checking things like: do all referenced
# tensors exist, are shapes consistent, are operator attributes valid for
# their opset. This is a cheap, fast sanity check we should always run
# right after export, before handing the model to any runtime.

# %%
print("Validating ONNX model structure with onnx.checker...")
onnx_model_loaded = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model_loaded)
print("Validation passed: the ONNX graph is well-formed.")

# %% [markdown]
# ## Step 5 — (Optional) Graph optimization with `onnxoptimizer`
#
# `onnxoptimizer` applies graph-level rewrites that don't change the
# model's outputs but can reduce graph size or improve runtime efficiency —
# for example, fusing consecutive operators or eliminating nodes that have
# no effect on the output. This step is optional: ONNX Runtime applies its
# own graph optimizations automatically when it loads a model (we'll see
# this in Step 6), so `onnxoptimizer` is mainly useful when you want an
# optimized `.onnx` file on disk itself, independent of which runtime loads
# it later.

# %%
if ONNXOPTIMIZER_AVAILABLE:
    print("Running onnxoptimizer graph fusion passes...")
    passes = [
        "eliminate_identity",
        "eliminate_nop_dropout",
        "fuse_consecutive_transposes",
        "fuse_bn_into_conv",
    ]
    optimized_model = onnxoptimizer.optimize(onnx_model_loaded, passes)
    onnx.save(optimized_model, ONNX_OPTIMIZED_PATH)

    optimized_size_mb = os.path.getsize(ONNX_OPTIMIZED_PATH) / (1024 * 1024)
    print(f"Optimized ONNX model saved to: {ONNX_OPTIMIZED_PATH}")
    print(f"Optimized file size: {optimized_size_mb:.2f} MB (original: {onnx_size_mb:.2f} MB)")
else:
    print("Skipping onnxoptimizer step (package not installed). "
          "See README.md for the optional install command.")
    ONNX_OPTIMIZED_PATH = None

# %% [markdown]
# ## Step 6 — Run the model with ONNX Runtime
#
# ONNX Runtime is an optimized runtime (the third lever from the lecture:
# Standard Formats -> Compression -> Optimized Runtimes). When it loads a
# model, it automatically applies its own graph optimizations — operator
# fusion, constant folding, layout optimization — tuned for the execution
# provider in use (here, CPU).
#
# We create an `InferenceSession`, which is ONNX Runtime's handle for
# running a loaded model repeatedly without reloading it each time.

# %%
print("Creating ONNX Runtime inference session...")
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Session input name : {input_name}")
print(f"Session output name: {output_name}")


def benchmark_onnxruntime(session, input_array, n_warmup=10, n_runs=50):
    """Same warmup + averaged-latency pattern as the Keras benchmark,
    adapted for an ONNX Runtime session."""
    # Warmup.
    for _ in range(n_warmup):
        _ = session.run([output_name], {input_name: input_array})

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = session.run([output_name], {input_name: input_array})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return float(np.mean(latencies))


# ONNX Runtime expects numpy arrays, not TF tensors.
dummy_input_np = dummy_input.numpy()

onnxruntime_latency_ms = benchmark_onnxruntime(session, dummy_input_np)
print(f"ONNX Runtime latency: {onnxruntime_latency_ms:.3f} ms per forward pass (batch size 1)")

# %% [markdown]
# ## Step 7 — Verify Keras and ONNX outputs match
#
# Speed means nothing if the converted model produces different
# predictions. We run the *same* input through both the original Keras
# model and the exported ONNX model, then compare outputs element-by-element.
# A small numerical difference is expected (different math libraries,
# floating-point operation order), but it should be tiny — we use a strict
# tolerance of `1e-4` max absolute difference as our pass/fail bar.

# %%
print("Comparing Keras and ONNX Runtime outputs on the same input...")

keras_output = model(dummy_input, training=False).numpy()
onnx_output = session.run([output_name], {input_name: dummy_input_np})[0]

max_abs_diff = float(np.max(np.abs(keras_output - onnx_output)))
outputs_match = max_abs_diff < 1e-4

print(f"Max absolute difference between Keras and ONNX outputs: {max_abs_diff:.8f}")
if outputs_match:
    print("PASS: outputs match within tolerance (max diff < 1e-4). Conversion is numerically safe.")
else:
    print("FAIL: outputs differ more than expected. Investigate the export before proceeding.")

# Also confirm both models agree on the predicted class, as a second,
# more intuitive sanity check.
keras_pred_class = int(np.argmax(keras_output, axis=1)[0])
onnx_pred_class = int(np.argmax(onnx_output, axis=1)[0])
print(f"Keras predicted class index: {keras_pred_class}")
print(f"ONNX predicted class index : {onnx_pred_class}")
print(f"Predicted classes match: {keras_pred_class == onnx_pred_class}")

# %% [markdown]
# ## Step 8 — Summary
#
# A single, readable summary table to close out the lab — this is the kind
# of before/after snapshot you'd put in a deployment readiness checklist.

# %%
# Keras parameter count -> approximate in-memory size assuming float32 (4 bytes/param).
keras_param_count = model.count_params()
keras_size_mb = (keras_param_count * 4) / (1024 * 1024)

speedup = keras_latency_ms / onnxruntime_latency_ms if onnxruntime_latency_ms > 0 else float("nan")

print("=" * 60)
print("LAB 1 SUMMARY — Keras vs ONNX Runtime (MobileNetV2, batch=1)")
print("=" * 60)
print(f"{'Metric':<28}{'Keras':<16}{'ONNX Runtime':<16}")
print(f"{'Latency (ms)':<28}{keras_latency_ms:<16.3f}{onnxruntime_latency_ms:<16.3f}")
print(f"{'Model size (MB)':<28}{keras_size_mb:<16.2f}{onnx_size_mb:<16.2f}")
print("-" * 60)
print(f"Speedup (Keras / ONNX Runtime): {speedup:.2f}x")
print(f"Output validation: {'PASSED' if outputs_match else 'FAILED'} (max diff = {max_abs_diff:.8f})")
print(f"Predicted class match: {keras_pred_class == onnx_pred_class}")
print("=" * 60)
print()
print("Artifacts written:")
print(f"  - {ONNX_PATH}")
if ONNX_OPTIMIZED_PATH:
    print(f"  - {ONNX_OPTIMIZED_PATH}")
print()
print("Next: Lab 2 reuses mobilenetv2.onnx from this lab's artifacts/ folder "
      "to apply quantization and pruning.")
