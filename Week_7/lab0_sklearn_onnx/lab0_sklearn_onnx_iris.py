# %% [markdown]
# # sklearn to ONNX — Iris Dataset
# ## Conversion · Compression · Optimized Runtime
#
# **Week 7 · Model Standardization and Optimization**
#
# ---
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
# 1. Convert any scikit-learn classifier to ONNX using `skl2onnx`.
# 2. Validate the converted graph with `onnx.checker` and inspect its
#    inputs, outputs, and nodes.
# 3. Run sklearn ONNX models with ONNX Runtime and verify predictions
#    match the original sklearn model exactly.
# 4. Apply **INT8 dynamic quantization** to an ONNX model and compare
#    file sizes before and after.
# 5. Apply **graph fusion** passes with `onnxoptimizer` and see how the
#    node count changes.
# 6. Configure **ONNX Runtime session options** to unlock built-in
#    graph-level optimizations.
# 7. Benchmark all variants — sklearn native, ONNX FP32, ONNX INT8,
#    ONNX optimized — with proper warmup and percentile latency.
# 8. Sweep throughput across batch sizes and visualise everything in a
#    single matplotlib dashboard.
#
# ## The three levers (Week 7 recap)
#
# ```
# Lever 1 — Standard formats  : skl2onnx exports sklearn → ONNX graph
# Lever 2 — Compression       : quantize_dynamic + onnxoptimizer
# Lever 3 — Optimized runtime : ONNX Runtime with ORT_ENABLE_ALL
# ```
#
# All three applied to a classic tabular dataset with three sklearn models.

# %% [markdown]
# ## Cell 2 — Imports and setup

# %%
import os
import time
import warnings
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime import SessionOptions, GraphOptimizationLevel

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import onnxoptimizer
    ONNXOPT_OK = True
except ImportError:
    ONNXOPT_OK = False

warnings.filterwarnings("ignore")
np.random.seed(42)

# All .onnx files are written here
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def artifact(name: str) -> str:
    """Return the full path for an artifact file."""
    return os.path.join(OUTPUT_DIR, name)

print("Versions")
print(f"  scikit-learn    : {__import__('sklearn').__version__}")
print(f"  skl2onnx        : {__import__('skl2onnx').__version__}")
print(f"  onnx            : {onnx.__version__}")
print(f"  onnxruntime     : {ort.__version__}")
print(f"  onnxoptimizer   : {onnxoptimizer.__version__ if ONNXOPT_OK else 'not installed'}")

# %% [markdown]
# ## Cell 3 — Load and explore the Iris dataset
#
# The Iris dataset is a classic multiclass classification problem:
# - **150 samples**, **4 numeric features**, **3 classes**
# - Features: sepal length, sepal width, petal length, petal width (all in cm)
# - Classes: Setosa, Versicolour, Virginica
#
# It is small enough that we can see every prediction the model makes,
# which makes it ideal for verifying that sklearn and ONNX outputs match
# exactly.

# %%
iris   = load_iris()
X, y   = iris.data.astype(np.float32), iris.target
names  = iris.feature_names
labels = iris.target_names

df = pd.DataFrame(X, columns=names)
df["species"] = [labels[i] for i in y]

print(f"Dataset shape  : {X.shape}")
print(f"Class names    : {labels}")
print(f"Samples/class  : {np.bincount(y)}")
print()
print(df.sample(8, random_state=42).to_string(index=False))

# %% [markdown]
# ### Quick feature distributions

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
colors = ["#2a6496", "#3aafa9", "#d9534f"]
for ax, feat in zip(axes, names):
    for cls_i, (cls_name, color) in enumerate(zip(labels, colors)):
        vals = df[df["species"] == cls_name][feat]
        ax.hist(vals, bins=12, alpha=0.65, label=cls_name, color=color)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel("cm")
    if ax == axes[0]:
        ax.set_ylabel("count")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Iris Feature Distributions by Class", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(artifact("iris_distributions.png"), dpi=130, bbox_inches="tight")
plt.show()
print("Saved: iris_distributions.png")

# %% [markdown]
# ## Cell 4 — Train / test split and scaling
#
# We apply `StandardScaler` inside a `sklearn.pipeline.Pipeline` for each
# model. Using a Pipeline matters for ONNX export: `skl2onnx` can export
# the entire Pipeline — scaler + classifier — as a single ONNX graph.
# This means at inference time we hand the runtime raw features (cm) and
# get predictions back, exactly like calling `pipeline.predict()`. No
# separate scaling step needed at deploy time.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")

# %% [markdown]
# ## Cell 5 — Train three sklearn pipelines
#
# We use three classifiers that represent different model families:
#
# | Model | Family | Notes |
# |-------|--------|-------|
# | `LogisticRegression` | Linear | Fast, simple, very small ONNX graph |
# | `RandomForestClassifier` | Ensemble (trees) | Larger ONNX graph (one node per tree split) |
# | `SVC(probability=True)` | Kernel method | Needs `probability=True` to export probabilities |
#
# All three are wrapped in a `Pipeline(StandardScaler + classifier)` so
# the scaler is baked into the ONNX export.

# %%
pipelines = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=200, random_state=42)),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=50, random_state=42)),
    ]),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(probability=True, random_state=42)),
    ]),
}

sklearn_accuracy = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    sklearn_accuracy[name] = acc
    print(f"\n{'='*50}")
    print(f"  {name}  —  Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=labels))

# %% [markdown]
# ## Cell 6 — Theory: how `skl2onnx` converts sklearn models
#
# ### The problem `skl2onnx` solves
# scikit-learn models store their learned parameters as Python/NumPy
# objects and run inference via Python method calls (`predict`,
# `predict_proba`). That's great for experimentation but it means:
# - The model is **tied to Python and scikit-learn**.
# - A Java, C++, or mobile service can't call `.predict()`.
# - Scaling to many parallel workers means scaling Python processes.
#
# `skl2onnx` walks the sklearn object tree and emits an **ONNX graph**
# whose nodes are standard mathematical operators (MatMul, Add, ArgMax,
# TreeEnsembleClassifier, …). Any ONNX-compatible runtime can then execute
# this graph without Python or scikit-learn installed.
#
# ### `initial_type` — why sklearn needs explicit types
# PyTorch and TensorFlow models carry dtype and shape information in their
# tensors. sklearn estimators do not — a fitted `LogisticRegression` has no
# record of whether it was trained on float32 or float64 data. We must
# therefore tell `skl2onnx` explicitly:
#
# ```python
# initial_type = [("float_input", FloatTensorType([None, 4]))]
# #                  ↑ name           ↑ dtype       ↑ (batch, features)
# # None in position 0 = dynamic batch size
# ```
#
# ### What the converted graph contains
# For a Pipeline(StandardScaler + LogisticRegression) the ONNX graph will
# include:
# 1. **Sub** and **Div** nodes — implement the StandardScaler transform.
# 2. **MatMul** and **Add** nodes — implement the linear model.
# 3. **Softmax** and **ArgMax** nodes — produce probabilities and the
#    predicted class label.
#
# For a RandomForest the graph contains a single
# `TreeEnsembleClassifier` node — a native ONNX operator that encodes the
# entire forest as a flat data structure. This is why tree models can run
# very fast inside ONNX Runtime.

# %% [markdown]
# ## Cell 7 — Convert all three pipelines to ONNX
#
# `convert_sklearn` takes the fitted pipeline and the `initial_type` hint
# and returns an `onnx.ModelProto` object. We then serialize it to disk
# with `onnx.save()`.

# %%
# 4 input features (sepal/petal length and width), dynamic batch axis.
INITIAL_TYPE = [("float_input", FloatTensorType([None, 4]))]

onnx_paths    = {}   # {model_name: path_to_onnx_file}
onnx_models   = {}   # {model_name: onnx.ModelProto}
onnx_sizes_mb = {}   # {model_name: file_size_in_MB}

print("Converting sklearn pipelines to ONNX...\n")
for name, pipe in pipelines.items():
    path = artifact(f"{name.lower()}_fp32.onnx")

    # convert_sklearn is the main entry point. It introspects the pipeline
    # step by step and emits the right ONNX operators for each sklearn type.
    onnx_model = convert_sklearn(
        pipe,
        initial_types=INITIAL_TYPE,
        target_opset=17,          # same opset we used in the Keras labs
    )
    onnx.save(onnx_model, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    onnx_paths[name]    = path
    onnx_models[name]   = onnx_model
    onnx_sizes_mb[name] = size_mb

    print(f"  {name}")
    print(f"    Saved to : {path}")
    print(f"    File size: {size_mb:.4f} MB")
    print()

# %% [markdown]
# ## Cell 8 — Inspect the ONNX graphs
#
# `onnx.checker.check_model()` validates the graph structure. After that
# we peek at the input/output tensor specs and count the graph's nodes —
# this shows concretely how each model family maps to operators.
#
# A key thing to notice: the RandomForest graph has far more nodes than
# Logistic Regression because each decision split in each tree becomes an
# operator in the graph.

# %%
print("ONNX graph inspection\n")
print(f"{'Model':<22} {'Inputs':<10} {'Outputs':<10} {'Nodes':>6}  {'Validation'}")
print("-" * 65)

for name, onnx_model in onnx_models.items():
    # Validate: raises an exception if the graph is malformed
    onnx.checker.check_model(onnx_model)

    graph  = onnx_model.graph
    inputs = [f"{i.name}:{list(i.type.tensor_type.shape.dim)}" for i in graph.input]
    n_out  = len(graph.output)
    n_node = len(graph.node)

    print(f"  {name:<20} {len(inputs):<10} {n_out:<10} {n_node:>6}  ✓ PASSED")

print()
print("Detailed input/output specs:")
for name, onnx_model in onnx_models.items():
    print(f"\n  {name}")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value or "?" for d in inp.type.tensor_type.shape.dim]
        print(f"    IN  {inp.name:20s}  shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value or "?" for d in out.type.tensor_type.shape.dim]
        print(f"    OUT {out.name:20s}  shape={shape}")

# %% [markdown]
# ## Cell 9 — ONNX Runtime inference
#
# We create one `InferenceSession` per model. The session loads the ONNX
# graph, applies its own internal graph optimizations, and is ready to
# run inference on numpy arrays.
#
# **Two outputs**: sklearn classifiers exported with `skl2onnx` typically
# produce two outputs:
# 1. **Label** — the predicted class index (like `predict()`).
# 2. **Probabilities** — a list of dicts or an array of class scores
#    (like `predict_proba()`).

# %%
ort_sessions = {}   # {model_name: InferenceSession}
ort_preds    = {}   # {model_name: predicted_labels_array}
ort_probs    = {}   # {model_name: probabilities_array}

print("Running ONNX Runtime inference on test set...\n")

for name, path in onnx_paths.items():
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_sessions[name] = sess

    in_name   = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    # Run inference on the full test set (X_test is already float32)
    outputs = sess.run(out_names, {in_name: X_test})

    # Output 0 = predicted labels, Output 1 = probabilities
    pred_labels = np.array(outputs[0])
    raw_probs   = outputs[1]

    # skl2onnx returns probabilities as a list of dicts: [{cls0: p, cls1: p, cls2: p}, ...]
    # We convert to a plain (n_samples, n_classes) numpy array.
    if isinstance(raw_probs[0], dict):
        probs = np.array([[d[k] for k in sorted(d)] for d in raw_probs])
    else:
        probs = np.array(raw_probs)

    ort_preds[name] = pred_labels
    ort_probs[name] = probs

    ort_acc = accuracy_score(y_test, pred_labels)
    print(f"  {name:<22}  ONNX accuracy: {ort_acc:.4f}  |  outputs: {out_names}")

# %% [markdown]
# ## Cell 10 — Verify sklearn vs ONNX predictions match exactly
#
# This is the critical correctness check. The conversion should be
# **lossless** for all three classifiers — every sample in the test set
# should get the same predicted class from sklearn and ONNX Runtime.
# Any mismatch would indicate a conversion bug.

# %%
print("Prediction match check: sklearn vs ONNX Runtime\n")
all_passed = True
for name, pipe in pipelines.items():
    sklearn_preds = pipe.predict(X_test)
    onnx_preds    = ort_preds[name]

    # Element-wise comparison
    matches       = (sklearn_preds == onnx_preds)
    n_match       = int(matches.sum())
    n_total       = len(y_test)
    passed        = (n_match == n_total)
    all_passed    = all_passed and passed

    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {name:<22}  {n_match}/{n_total} predictions match  [{status}]")

    # If there are mismatches, show exactly which samples differ
    if not passed:
        bad = np.where(~matches)[0]
        for idx in bad:
            print(f"    Sample {idx}: sklearn={sklearn_preds[idx]}  onnx={onnx_preds[idx]}")

print()
if all_passed:
    print("All models: sklearn and ONNX Runtime predictions are identical. ✓")
else:
    print("WARNING: some predictions differ. Investigate before deploying.")

# %% [markdown]
# ---
# # Section 2 — Compression
#
# ## Cell 11 — Theory: quantization on non-neural sklearn models
#
# ### What changes vs neural network quantization?
# When we quantized MobileNetV2 in Lab 2, we were quantizing millions
# of floating-point weights in Conv and Dense layers. sklearn models are
# structurally very different:
#
# | Model | What gets quantized |
# |-------|---------------------|
# | LogisticRegression | Weight matrix, bias vector → INT8 |
# | RandomForest | Threshold values at each split node → INT8 |
# | SVC | Support vector coefficients → INT8 |
#
# For **tree-based models** (RandomForest), quantization often has
# almost no accuracy impact — the model's logic is based on whether a
# feature is above or below a threshold, and rounding those thresholds
# to INT8 rarely changes the decision.
#
# For **linear models** (LogisticRegression) and **kernel methods** (SVC)
# the weight values matter more, so there can be a tiny accuracy delta —
# still usually within rounding error on clean tabular data.
#
# ### `quantize_dynamic` on sklearn ONNX graphs
# `quantize_dynamic` walks the ONNX graph and replaces FP32 MatMul and
# Gemm nodes with INT8 equivalents, storing the quantized weights
# statically. Activations are quantized dynamically per-inference-call.
# The result is a smaller file and faster matrix arithmetic on CPUs that
# have INT8 SIMD instructions.
#
# ### `onnxoptimizer` graph fusion
# `onnxoptimizer` applies algebraic rewrites to the ONNX graph:
# - **Constant folding**: evaluate sub-graphs that depend only on constants
#   and replace them with their result.
# - **Identity elimination**: remove nodes that copy a tensor unchanged.
# - **Consecutive transpose fusion**: merge two back-to-back transposes
#   into one (or eliminate them if they cancel out).
# - **BN-into-Conv fusion**: fold BatchNorm parameters into the preceding
#   Conv weights (not applicable to sklearn models, but runs harmlessly).
#
# On sklearn ONNX graphs the optimizer mainly helps through constant
# folding and identity elimination, reducing the number of nodes the
# runtime must execute.

# %% [markdown]
# ## Cell 12 — Apply ONNX INT8 dynamic quantization

# %%
int8_paths    = {}   # {model_name: path_to_int8_onnx}
int8_sizes_mb = {}

print("Applying INT8 dynamic quantization...\n")
for name, fp32_path in onnx_paths.items():
    int8_path = artifact(f"{name.lower()}_int8.onnx")

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,  # unsigned INT8 for weights
    )

    fp32_mb = onnx_sizes_mb[name]
    int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
    reduction = (1 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 else 0

    int8_paths[name]    = int8_path
    int8_sizes_mb[name] = int8_mb

    print(f"  {name}")
    print(f"    FP32 size : {fp32_mb:.4f} MB")
    print(f"    INT8 size : {int8_mb:.4f} MB  ({reduction:+.1f}%)")
    print()

# %% [markdown]
# ## Cell 13 — Apply onnxoptimizer graph fusion
#
# We apply a set of safe, lossless graph-rewrite passes. These don't
# change any numerical values — they only restructure the graph so the
# runtime has fewer nodes to execute and can cache computations better.

# %%
opt_paths    = {}
opt_sizes_mb = {}

PASSES = [
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_unused_initializer",
    "fuse_consecutive_transposes",
    "fuse_bn_into_conv",
    "fuse_matmul_add_bias_into_gemm",
]

print(f"onnxoptimizer available: {ONNXOPT_OK}")
print(f"Passes: {PASSES}\n")

for name, fp32_path in onnx_paths.items():
    opt_path = artifact(f"{name.lower()}_optimized.onnx")

    if ONNXOPT_OK:
        m_in     = onnx.load(fp32_path)
        n_before = len(m_in.graph.node)

        m_out    = onnxoptimizer.optimize(m_in, PASSES)
        n_after  = len(m_out.graph.node)
        onnx.save(m_out, opt_path)

        opt_mb = os.path.getsize(opt_path) / (1024 * 1024)
        print(f"  {name}")
        print(f"    Nodes before: {n_before}  →  after: {n_after}  (Δ {n_after - n_before:+d})")
        print(f"    Optimized size: {opt_mb:.4f} MB")
        print()
    else:
        # Fallback: just copy FP32 model so the benchmark cells still work
        import shutil
        shutil.copy(fp32_path, opt_path)
        opt_mb = onnx_sizes_mb[name]
        print(f"  {name}: onnxoptimizer not installed — using FP32 model as fallback")

    opt_paths[name]    = opt_path
    opt_sizes_mb[name] = opt_mb

# %% [markdown]
# ## Cell 14 — Verify compressed models still match sklearn predictions
#
# After quantization and graph optimization, predictions must still match
# the original sklearn model. For these three sklearn model families on
# clean tabular data, 100% label match is the expected outcome.

# %%
print("Prediction match: sklearn vs INT8 and vs Optimized ONNX\n")

for name in pipelines:
    sklearn_preds = pipelines[name].predict(X_test)

    # --- INT8 check ---
    sess_i    = ort.InferenceSession(int8_paths[name], providers=["CPUExecutionProvider"])
    in_name_i = sess_i.get_inputs()[0].name
    out_i     = sess_i.run([sess_i.get_outputs()[0].name], {in_name_i: X_test})
    int8_pred = np.array(out_i[0])
    int8_ok   = (int8_pred == sklearn_preds).all()

    # --- Optimized check ---
    sess_o    = ort.InferenceSession(opt_paths[name], providers=["CPUExecutionProvider"])
    in_name_o = sess_o.get_inputs()[0].name
    out_o     = sess_o.run([sess_o.get_outputs()[0].name], {in_name_o: X_test})
    opt_pred  = np.array(out_o[0])
    opt_ok    = (opt_pred == sklearn_preds).all()

    int8_acc  = accuracy_score(y_test, int8_pred)
    opt_acc   = accuracy_score(y_test, opt_pred)

    print(f"  {name}")
    print(f"    INT8      — match: {'✓' if int8_ok else '✗'}  accuracy: {int8_acc:.4f}")
    print(f"    Optimized — match: {'✓' if opt_ok  else '✗'}  accuracy: {opt_acc:.4f}")
    print()

# %% [markdown]
# ---
# # Section 3 — Optimized Runtime
#
# ## Cell 15 — Theory: ONNX Runtime session options
#
# When you call `ort.InferenceSession(path)` with no extra arguments,
# ONNX Runtime applies a **default** set of internal optimizations.
# `SessionOptions` lets us control exactly what the runtime does:
#
# ### Graph optimization levels
# | Level | Constant | What it does |
# |-------|----------|-------------|
# | 0 | `ORT_DISABLE_ALL` | No optimization — raw graph execution |
# | 1 | `ORT_ENABLE_BASIC` | Simple rewrites: identity elimination, constant folding |
# | 2 | `ORT_ENABLE_EXTENDED` | More aggressive: node fusion, memory layout optimization |
# | 99 | `ORT_ENABLE_ALL` | All of the above plus layout-specific tuning |
#
# ### Thread tuning
# `intra_op_num_threads` controls how many CPU threads a single operator
# can use for parallelism. On a small model like sklearn ONNX graphs,
# using 1 thread often beats the default because the overhead of
# spawning threads exceeds the benefit of parallelism for small inputs.
#
# ### Execution providers
# `CPUExecutionProvider` — runs on CPU using optimized BLAS/SIMD kernels.
# On a machine with a GPU you could add `CUDAExecutionProvider` as the
# first choice and ONNX Runtime will use the GPU for supported operators.
#
# ### Why this matters
# For neural networks the runtime's internal optimizations (fused ops,
# memory-efficient layouts) can produce 10–30% speedups on top of the
# model-level optimizations we already applied. For small sklearn models
# the gains are more modest but thread tuning often helps significantly.

# %% [markdown]
# ## Cell 16 — Configure runtime with full session options
#
# We create four session configurations and will benchmark all of them:
#
# | Config | What it enables |
# |--------|----------------|
# | `default` | ORT's default (= EXTENDED internally) |
# | `ort_all` | `ORT_ENABLE_ALL` |
# | `ort_all_1t` | `ORT_ENABLE_ALL` + single thread |
# | `ort_disable` | All optimizations disabled (useful as a lower-bound) |

# %%
def make_session(onnx_path: str, opt_level, n_threads: int = 0) -> ort.InferenceSession:
    """Create an ONNX Runtime session with explicit options.

    Args:
        onnx_path : path to the .onnx file
        opt_level : GraphOptimizationLevel constant
        n_threads : intra-op thread count; 0 = ORT default
    """
    opts = SessionOptions()
    opts.graph_optimization_level = opt_level
    if n_threads > 0:
        opts.intra_op_num_threads = n_threads
    return ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


SESSION_CONFIGS = {
    "ORT Default":      lambda p: ort.InferenceSession(p, providers=["CPUExecutionProvider"]),
    "ORT_ENABLE_ALL":   lambda p: make_session(p, GraphOptimizationLevel.ORT_ENABLE_ALL),
    "ORT_ALL 1-thread": lambda p: make_session(p, GraphOptimizationLevel.ORT_ENABLE_ALL, n_threads=1),
    "ORT_DISABLE_ALL":  lambda p: make_session(p, GraphOptimizationLevel.ORT_DISABLE_ALL),
}

print("Session configurations defined:")
for cfg_name in SESSION_CONFIGS:
    print(f"  {cfg_name}")

# %% [markdown]
# ## Cell 17 — Latency benchmark: all variants, all session configs
#
# We benchmark four model variants × four session configs × three sklearn
# models. Each benchmark uses:
# - **20 warmup runs** — let the runtime settle
# - **200 timed runs** — enough for stable p50 / p95 / p99
# - **Batch size 1** — single-sample latency (the most demanding SLO)
#
# We also benchmark sklearn's native `predict()` as the baseline to beat.

# %%
N_WARMUP = 20
N_RUNS   = 200

# We benchmark one sample at a time (the hardest case for latency SLOs)
x_single = X_test[[0]]   # shape (1, 4), float32


def bench_ort(sess: ort.InferenceSession, x: np.ndarray) -> list:
    """Return a list of per-call latencies in ms (after warmup)."""
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    for _ in range(N_WARMUP):
        sess.run([out_name], {in_name: x})
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run([out_name], {in_name: x})
        times.append((time.perf_counter() - t0) * 1e6)  # µs
    return times


def bench_sklearn(pipe, x: np.ndarray) -> list:
    """Return per-call latencies in µs for a sklearn pipeline."""
    for _ in range(N_WARMUP):
        pipe.predict(x)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        pipe.predict(x)
        times.append((time.perf_counter() - t0) * 1e6)
    return times


def percentiles(times: list) -> dict:
    arr = np.array(times)
    return {"p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
            "mean": np.mean(arr)}


# Results structure: latency_data[model_name][variant_label] = percentile_dict
latency_data = {n: {} for n in pipelines}

print("Running latency benchmarks (batch=1, 200 runs each)...\n")
print("NOTE: units are microseconds (µs), not milliseconds — sklearn models are very fast.\n")

for model_name in pipelines:
    print(f"  {model_name}")

    # sklearn native
    sk_times = bench_sklearn(pipelines[model_name], x_single)
    sk_p     = percentiles(sk_times)
    latency_data[model_name]["sklearn"] = sk_p
    print(f"    sklearn native   p50={sk_p['p50']:6.1f} µs  p95={sk_p['p95']:6.1f} µs  p99={sk_p['p99']:6.1f} µs")

    # FP32 ONNX variants
    for cfg_name, sess_fn in SESSION_CONFIGS.items():
        sess   = sess_fn(onnx_paths[model_name])
        times  = bench_ort(sess, x_single)
        p      = percentiles(times)
        label  = f"FP32 / {cfg_name}"
        latency_data[model_name][label] = p
        print(f"    FP32/{cfg_name:<18} p50={p['p50']:6.1f} µs  p95={p['p95']:6.1f} µs  p99={p['p99']:6.1f} µs")

    # INT8
    sess_i  = ort.InferenceSession(int8_paths[model_name], providers=["CPUExecutionProvider"])
    times_i = bench_ort(sess_i, x_single)
    pi      = percentiles(times_i)
    latency_data[model_name]["INT8 ONNX"] = pi
    print(f"    INT8 ONNX         p50={pi['p50']:6.1f} µs  p95={pi['p95']:6.1f} µs  p99={pi['p99']:6.1f} µs")

    # Optimized
    sess_o  = ort.InferenceSession(opt_paths[model_name], providers=["CPUExecutionProvider"])
    times_o = bench_ort(sess_o, x_single)
    po      = percentiles(times_o)
    latency_data[model_name]["Optimized ONNX"] = po
    print(f"    Optimized ONNX    p50={po['p50']:6.1f} µs  p95={po['p95']:6.1f} µs  p99={po['p99']:6.1f} µs")
    print()

# %% [markdown]
# ## Cell 18 — Throughput benchmark: batch sizes [10, 50, 100, 500]
#
# Throughput = samples processed per second.
# We test on the FP32 ONNX default session and sklearn native for each
# model, across batch sizes that reflect realistic serving patterns
# (single online requests vs micro-batched API calls vs batch jobs).

# %%
BATCH_SIZES  = [10, 50, 100, 500]
N_WARMUP_TP  = 5
N_RUNS_TP    = 30

# throughput_data[model_name]["sklearn" / "ONNX FP32"][batch_size] = samples/sec
throughput_data = {n: {"sklearn": {}, "ONNX FP32": {}} for n in pipelines}

print("Running throughput sweep...\n")

for model_name in pipelines:
    print(f"  {model_name}")
    pipe    = pipelines[model_name]
    sess_fp = ort.InferenceSession(onnx_paths[model_name], providers=["CPUExecutionProvider"])
    in_name = sess_fp.get_inputs()[0].name
    out_name= sess_fp.get_outputs()[0].name

    for bs in BATCH_SIZES:
        x_batch = X_test[np.random.choice(len(X_test), bs, replace=True)].astype(np.float32)

        # sklearn
        for _ in range(N_WARMUP_TP): pipe.predict(x_batch)
        t0 = time.perf_counter()
        for _ in range(N_RUNS_TP): pipe.predict(x_batch)
        sk_tp = (bs * N_RUNS_TP) / (time.perf_counter() - t0)

        # ONNX FP32
        for _ in range(N_WARMUP_TP):
            sess_fp.run([out_name], {in_name: x_batch})
        t0 = time.perf_counter()
        for _ in range(N_RUNS_TP):
            sess_fp.run([out_name], {in_name: x_batch})
        ort_tp = (bs * N_RUNS_TP) / (time.perf_counter() - t0)

        throughput_data[model_name]["sklearn"][bs]   = sk_tp
        throughput_data[model_name]["ONNX FP32"][bs] = ort_tp
        print(f"    batch={bs:>3d}: sklearn={sk_tp:>10,.0f} samp/s  |  ONNX={ort_tp:>10,.0f} samp/s")
    print()

# %% [markdown]
# ## Cell 19 — Matplotlib benchmarking dashboard

# %%
MODEL_LIST   = list(pipelines.keys())
VARIANT_BARS = ["sklearn", "FP32 / ORT Default", "INT8 ONNX", "Optimized ONNX"]
BAR_COLORS   = ["#5A7184", "#2a6496", "#3aafa9", "#d9534f"]
TP_COLORS    = {"sklearn": "#5A7184", "ONNX FP32": "#2a6496"}
MODEL_COLORS = ["#2a6496", "#3aafa9", "#d9534f"]

fig  = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor("#f7f9fa")
gs   = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

# ── Panels 0-2: latency p50/p95/p99 per model ─────────────────────────────
for mi, model_name in enumerate(MODEL_LIST):
    ax = fig.add_subplot(gs[0, mi])
    ld = latency_data[model_name]

    x      = np.arange(len(VARIANT_BARS))
    width  = 0.25
    pcts   = ["p50", "p95", "p99"]
    pct_c  = ["#2a6496", "#f0ad4e", "#d9534f"]

    for pi, pct in enumerate(pcts):
        vals = [ld.get(v, {}).get(pct, 0) for v in VARIANT_BARS]
        bars = ax.bar(x + pi * width, vals, width, label=pct,
                      color=pct_c[pi], edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(f"Latency — {model_name}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latency (µs)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["sklearn", "FP32\nORT", "INT8", "Opt."],
                       fontsize=8, rotation=0)
    ax.legend(fontsize=7, title="Percentile")
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Panels 3-5: model size on disk (all variants per model) ───────────────
for mi, model_name in enumerate(MODEL_LIST):
    ax = fig.add_subplot(gs[1, mi])
    variant_names = ["FP32 ONNX", "INT8 ONNX", "Optimized\nONNX"]
    sizes = [
        onnx_sizes_mb[model_name],
        int8_sizes_mb[model_name],
        opt_sizes_mb[model_name],
    ]
    bars = ax.bar(variant_names, sizes,
                  color=["#2a6496", "#3aafa9", "#d9534f"],
                  edgecolor="white", linewidth=0.7, width=0.5)
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.00002,
                f"{v:.4f} MB", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Model Size — {model_name}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Size (MB)")
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Panels 6-8: throughput vs batch size (one per model) ──────────────────
for mi, model_name in enumerate(MODEL_LIST):
    ax = fig.add_subplot(gs[2, mi])
    for runtime_name, color in TP_COLORS.items():
        td  = throughput_data[model_name][runtime_name]
        xs  = sorted(td.keys())
        ys  = [td[b] / 1000 for b in xs]   # convert to k-samples/s
        ax.plot(xs, ys, marker="o", label=runtime_name, color=color, linewidth=2, markersize=6)
        for b, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}k", (b, y),
                        textcoords="offset points", xytext=(3, 4), fontsize=7, color=color)
    ax.set_title(f"Throughput — {model_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (k samples / sec)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(fontsize=8)
    ax.set_facecolor("#ffffff")
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Week 7 · sklearn → ONNX Benchmark Dashboard\n"
    "Iris Dataset · 3 Models · 4 Variants · CPU Inference",
    fontsize=13, fontweight="bold", y=1.01, color="#1a1a2e"
)

dashboard_path = artifact("sklearn_onnx_dashboard.png")
plt.savefig(dashboard_path, dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print(f"Dashboard saved: {dashboard_path}")

# %% [markdown]
# ## Cell 20 — Full comparison summary table

# %%
# Pre-compute INT8 and optimized accuracy for each model
int8_accuracy = {}
opt_accuracy  = {}
for model_name in MODEL_LIST:
    sk_preds = pipelines[model_name].predict(X_test)

    sess_i   = ort.InferenceSession(int8_paths[model_name],  providers=["CPUExecutionProvider"])
    out_i    = sess_i.run([sess_i.get_outputs()[0].name],
                          {sess_i.get_inputs()[0].name: X_test})[0]
    int8_accuracy[model_name] = accuracy_score(y_test, np.array(out_i))

    sess_o   = ort.InferenceSession(opt_paths[model_name], providers=["CPUExecutionProvider"])
    out_o    = sess_o.run([sess_o.get_outputs()[0].name],
                          {sess_o.get_inputs()[0].name: X_test})[0]
    opt_accuracy[model_name] = accuracy_score(y_test, np.array(out_o))

print("=" * 90)
print("FULL SUMMARY — sklearn to ONNX · Iris Dataset")
print("=" * 90)
print(f"{'Model':<22} {'Variant':<18} {'Size (MB)':>10} "
      f"{'p50 (µs)':>10} {'p95 (µs)':>10} {'Speedup':>9} {'Acc':>7} {'Match':>6}")
print("-" * 90)

for model_name in MODEL_LIST:
    sk_acc  = sklearn_accuracy[model_name]
    sk_ld   = latency_data[model_name]
    sk_p50  = sk_ld["sklearn"]["p50"]
    sk_p95  = sk_ld["sklearn"]["p95"]

    def row(variant, size_mb, p50, p95, acc):
        speedup = sk_p50 / p50 if p50 > 0 else float("nan")
        size_s  = f"{size_mb:.4f}" if isinstance(size_mb, float) else str(size_mb)
        match   = abs(acc - sk_acc) < 1e-6
        return (f"{variant:<18} {size_s:>10} {p50:>10.1f} {p95:>10.1f} "
                f"{speedup:>9.2f}x {acc:>7.4f} {'✓' if match else '✗':>6}")

    fp32_ld = sk_ld.get("FP32 / ORT Default", {})
    int8_ld = sk_ld.get("INT8 ONNX", {})
    opt_ld  = sk_ld.get("Optimized ONNX", {})

    print(f"  {model_name:<20} " + row("sklearn",       "-",                   sk_p50, sk_p95, sk_acc))
    print(f"  {'':20} " + row("FP32 ONNX",     onnx_sizes_mb[model_name], fp32_ld.get("p50",0), fp32_ld.get("p95",0), sk_acc))
    print(f"  {'':20} " + row("INT8 ONNX",     int8_sizes_mb[model_name], int8_ld.get("p50",0), int8_ld.get("p95",0), int8_accuracy[model_name]))
    print(f"  {'':20} " + row("Optimized ONNX",opt_sizes_mb[model_name],  opt_ld.get("p50",0),  opt_ld.get("p95",0),  opt_accuracy[model_name]))
    print()

# %% [markdown]
# ## Cell 21 — Key takeaways
#
# ### What we covered
# This notebook walked all three Week 7 levers applied to sklearn models:
#
# **Lever 1 — Standard formats (`skl2onnx`)**
# - `convert_sklearn()` exports any sklearn estimator or Pipeline to ONNX.
# - The `initial_type` hint is mandatory because sklearn doesn't store
#   dtype/shape information alongside its trained parameters.
# - Pipelines (scaler + classifier) export as a single end-to-end graph —
#   no separate preprocessing step at inference time.
#
# **Lever 2 — Compression**
# - `quantize_dynamic` reduces file size significantly for linear models
#   and SVMs; tree models are already compact and see smaller reductions.
# - `onnxoptimizer` reduces node count through algebraic rewrites; gains
#   are most visible on linear models with multiple consecutive MatMul ops.
# - Prediction match remains 100% for all three sklearn families on
#   clean tabular data — a good sign that compression is safe here.
#
# **Lever 3 — Optimized runtime**
# - `ORT_ENABLE_ALL` with single-threaded execution often wins for
#   small models where threading overhead exceeds the parallelism benefit.
# - Throughput scales roughly linearly with batch size up to the point
#   where CPU cache pressure kicks in.
#
# ### When to use `skl2onnx`
# ✅ You have sklearn models serving online API traffic (latency matters)
# ✅ You want to decouple the serving environment from Python/sklearn
# ✅ Your org standardizes on ONNX Runtime for all model serving
#
# ### Limitations
# - Custom transformers that use arbitrary Python code cannot be exported.
# - Some newer sklearn estimators may not be supported in older `skl2onnx`
#   versions — check the [operator support matrix](https://onnx.ai/sklearn-onnx/).
# - For the Iris dataset the models are so small and fast that ONNX
#   overhead can sometimes *exceed* sklearn's own overhead for a single
#   sample. The real wins appear at batch sizes > 50 and in production
#   serving contexts where the Python interpreter overhead of sklearn
#   becomes the bottleneck.
#
# ### The full Week 7 pipeline, applied
# ```
# sklearn fit()
#     └─► skl2onnx export     (Lever 1 — Standard format)
#             └─► quantize_dynamic + onnxoptimizer  (Lever 2 — Compression)
#                     └─► ORT_ENABLE_ALL session    (Lever 3 — Optimized runtime)
#                             └─► Production API
# ```
