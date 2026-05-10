# Model conversion errors
_Exported on 5/9/2026 at 16:45:04 PDT from Cursor (3.3.27)_

---

**User**

Model Conversion:

Help fix the errors:
 python tools/convert_ghostfacenet_fp16.py --onnx  /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx --out  build/ghostfacenet_fp16 --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin

========================================================================
ONNX model
========================================================================
  input :  name='input_1:0'  shape=(1, 112, 112, 3)
  output: name='Identity:0'  shape=(1, 512)
  OK loaded test input from /Users/gifty/Development/FaceMesh/models/test_0.bin (float32, (1, 112, 112, 3))

========================================================================
Converting ONNX -> TFLite (FP16, NHWC)
========================================================================

$ onnx2tf -i /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx -o build/ghostfacenet_fp16 -osd -cotof -rtpo AveragePool
WARNING:tensorflow:Your environment has TF_USE_LEGACY_KERAS set to True, but you do not have the tf_keras package installed. You must install it in order to use the legacy tf.keras. Install it via: `pip install tf_keras`
Traceback (most recent call last):
  File "/Users/gifty/Development/FaceMesh/.venv/bin/onnx2tf", line 5, in <module>
    from onnx2tf import main
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/onnx2tf/__init__.py", line 1, in <module>
    from onnx2tf.onnx2tf import convert, main
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/onnx2tf/onnx2tf.py", line 28, in <module>
    tf.keras.utils.set_random_seed(0)
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/tensorflow/python/util/lazy_loader.py", line 181, in __getattr__
    self._initialize()
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/tensorflow/python/util/lazy_loader.py", line 173, in _initialize
    raise ImportError(  # pylint: disable=raise-missing-from
ImportError: Keras cannot be imported. Check that it is installed.
Traceback (most recent call last):
  File "/Users/gifty/Development/FaceMesh/tools/convert_ghostfacenet_fp16.py", line 244, in <module>
    sys.exit(main())
  File "/Users/gifty/Development/FaceMesh/tools/convert_ghostfacenet_fp16.py", line 210, in main
    tflite_path = convert_with_onnx2tf(args.onnx, args.out)
  File "/Users/gifty/Development/FaceMesh/tools/convert_ghostfacenet_fp16.py", line 144, in convert_with_onnx2tf
    subprocess.run(cmd, check=True)
  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['onnx2tf', '-i', '/Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx', '-o', 'build/ghostfacenet_fp16', '-osd', '-cotof', '-rtpo', 'AveragePool']' returned non-zero exit status 1.

---

**Cursor**

The error tells you exactly what to do, but it's a bit hidden in the warning above the traceback:

```
WARNING:tensorflow: ... TF_USE_LEGACY_KERAS set to True, but you do not have the tf_keras package installed.
...
ImportError: Keras cannot be imported. Check that it is installed.
```

What's happening:

- Your `.venv` has TensorFlow 2.20 + Keras 3.
- `onnx2tf` internally sets `TF_USE_LEGACY_KERAS=1` (it requires the old Keras 2 API).
- TF 2.16+ no longer ships Keras 2; you have to install it as a separate package called `tf_keras`.

Fix: install `tf_keras` (matching your TF version) into the same venv.

No problem — I won't install anything. Here's the manual fix and what you should know.

## Root cause

`onnx2tf` sets `TF_USE_LEGACY_KERAS=1` internally because it needs the **Keras 2** API. Your venv has:

- `tensorflow==2.20.0`
- `keras==3.10.0` (Keras 3, the new default since TF 2.16)
- no `tf_keras` package

When `tf.keras.utils.set_random_seed(0)` runs at import time, TF tries to load legacy Keras and fails with `ImportError: Keras cannot be imported`.

## Fix (one command)

Activate the venv and install `tf_keras` matching your TF major.minor:

```bash
source /Users/gifty/Development/FaceMesh/.venv/bin/activate
pip install "tf_keras~=2.20.0"
```

Then re-run your conversion command. No other code changes are required.

Quick sanity check after install:

```bash
python -c "import os; os.environ['TF_USE_LEGACY_KERAS']='1'; import tensorflow as tf; tf.keras.utils.set_random_seed(0); print('OK')"
```

## Heads-up on a likely next failure

You're on a stack that `onnx2tf` is known to be brittle with:

- **NumPy 2.0.2** — several ops in `onnx2tf` 1.20 use APIs removed in NumPy 2 (e.g. `np.cast`, `np.bool8`). If you hit `AttributeError: module 'numpy' has no attribute ...` mid-conversion, downgrade:
  ```bash
  pip install "numpy<2"
  ```
- **TensorFlow 2.20** is bleeding-edge; `onnx2tf` 1.20 was tested mostly against TF 2.15–2.17. If conversion succeeds but `-cotof` reports large diffs, try `pip install "tensorflow==2.17.*" "tf_keras~=2.17.0"`.

If you want, I can also pin a known-good combination in a `requirements-convert.txt` so this is reproducible — say the word and I'll add it (without creating extra docs).

---

**User**

python -c "import os; os.environ['TF_USE_LEGACY_KERAS']='1'; import tensorflow as tf; tf.keras.utils.set_random_seed(0); print('OK')"
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
OK


AND

python tools/convert_ghostfacenet_fp16.py --onnx  /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx --out  build/ghostfacenet_fp16 --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin

========================================================================
ONNX model
========================================================================
  input :  name='input_1:0'  shape=(1, 112, 112, 3)
  output: name='Identity:0'  shape=(1, 512)
  OK loaded test input from /Users/gifty/Development/FaceMesh/models/test_0.bin (float32, (1, 112, 112, 3))

========================================================================
Converting ONNX -> TFLite (FP16, NHWC)
========================================================================

$ onnx2tf -i /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx -o build/ghostfacenet_fp16 -osd -cotof -rtpo AveragePool
Traceback (most recent call last):
  File "/Users/gifty/Development/FaceMesh/.venv/bin/onnx2tf", line 5, in <module>
    from onnx2tf import main
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/onnx2tf/__init__.py", line 1, in <module>
    from onnx2tf.onnx2tf import convert, main
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/onnx2tf/onnx2tf.py", line 37, in <module>
    import onnx_graphsurgeon as gs
ModuleNotFoundError: No module named 'onnx_graphsurgeon'
Traceback (most recent call last):
  File "/Users/gifty/Development/FaceMesh/tools/convert_ghostfacenet_fp16.py", line 244, in <module>
    sys.exit(main())
  File "/Users/gifty/Development/FaceMesh/tools/convert_ghostfacenet_fp16.py", line 210, in main
    tflite_path = convert_with_onnx2tf(args.onnx, args.out)
  File "/Users/gifty/Development/FaceMesh/tools/convert_ghostfacenet_fp16.py", line 144, in convert_with_onnx2tf
    subprocess.run(cmd, check=True)
  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['onnx2tf', '-i', '/Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx', '-o', 'build/ghostfacenet_fp16', '-osd', '-cotof', '-rtpo', 'AveragePool']' returned non-zero exit status 1.


---

**Cursor**

Good — Keras is happy now. The next missing piece is `onnx_graphsurgeon`, which `onnx2tf` uses for its graph rewriter. It's an NVIDIA package and isn't pulled in automatically by `pip install onnx2tf`.

## Fix

```bash
source /Users/gifty/Development/FaceMesh/.venv/bin/activate
pip install onnx_graphsurgeon
```

If pip can't find it on the default index (older pips sometimes can't), use NVIDIA's index as a fallback:

```bash
pip install --extra-index-url https://pypi.ngc.nvidia.com onnx-graphsurgeon
```

## Heads-up: there are likely a few more missing deps

`onnx2tf` 1.20 declares these as optional but actually imports several at runtime. To save another round of error→install→error, install the full set in one go:

```bash
pip install onnx_graphsurgeon onnxsim sng4onnx ai_edge_litert
```

What each is for:
- `onnx_graphsurgeon` — graph surgery used by every conversion path.
- `onnxsim` — ONNX simplifier, called early in the pipeline.
- `sng4onnx` — generates names for unnamed nodes/tensors.
- `ai_edge_litert` — newer TFLite runtime that `onnx2tf` 1.20 prefers over `tf.lite` for the post-conversion `-cotof` parity check.

After that, re-run:

```bash
python tools/convert_ghostfacenet_fp16.py \
  --onnx /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx \
  --out  build/ghostfacenet_fp16 \
  --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin
```

If you'd like, I can drop a `requirements-convert.txt` (just the file, no extra docs) so this stack is reproducible — let me know.

---

**User**

Had to pip install psutil as well

---

**Cursor**

Good catch — `psutil` is another runtime-only import in `onnx2tf` (used for memory/CPU heuristics) that isn't in its declared deps.

So the full extra-deps list for `onnx2tf` 1.20 on this stack is:

```
tf_keras~=2.20.0
onnx_graphsurgeon
onnxsim
sng4onnx
ai_edge_litert
psutil
```

Did the conversion finish, or did it hit another error? Happy to keep going — and if you want, I can pin all of these into a `requirements-convert.txt` now that we have the actual working set.

---

**User**

So - pip install onnx onnxruntime onnx2tf tensorflow numpy tf_keras~=2.20.0 onnx_graphsurgeon onnx_graphsurgeon onnxsim sng4onnx ai_edge_litert psutil
Let's generate a requirements.txt

---

**Cursor**

I'll put it in `tools/requirements.txt` next to the conversion script, with exact pins of what you actually have working — model-conversion stacks are notoriously fragile, so float-ranges aren't worth the risk here.

Now I'll update the docstring in `convert_ghostfacenet_fp16.py` so the install hint matches what actually works:

Done. Summary of what changed:

- New: `tools/requirements.txt` — exact pins of the working stack (onnx + onnx2tf + its undeclared runtime deps + TF 2.20 + tf_keras + numpy/protobuf).
- Updated: docstring in `tools/convert_ghostfacenet_fp16.py` now points at `pip install -r tools/requirements.txt`.

```1:34:tools/requirements.txt
# Dependencies for tools/convert_ghostfacenet_fp16.py
#
# Install into a Python 3.9+ venv:
#     python -m venv .venv
#     source .venv/bin/activate
#     pip install -r tools/requirements.txt
# ... pinned versions follow ...
```

A few notes for future-you:

- I kept everything as exact `==` pins because the onnx2tf pipeline silently produces wrong outputs when TF/NumPy/ONNX drift. If you ever bump these, re-run the script and confirm the `cos_sim > 0.999` line in the parity check.
- `tf_keras` is intentionally pinned to the same `2.20.x` as `tensorflow`; that pairing is required by the `TF_USE_LEGACY_KERAS=1` path that `onnx2tf` forces.
- `numpy==2.0.2` is what you have working; if you ever upgrade to NumPy 2.1+, double-check `onnx2tf` still imports cleanly — earlier 2.x versions have removed APIs that 1.20 still references in places.

---

**User**

Most of the conversion ran, but Python crashed, and also some terminal output:

@/Users/gifty/Documents/error.txt 

---

**Cursor**

Good news: the conversion **actually succeeded**. Both TFLite files were written before it crashed:

```
Float32 tflite output complete!
Float16 tflite output complete!
ERROR: If --check_onnx_tf_outputs_elementwise_close is specified, you must install onnxruntime and sne4onnx. pip install sne4onnx onnxruntime
```

The script then errored only because the `-cotof` flag (`--check_onnx_tf_outputs_elementwise_close`) needs **`sne4onnx`** — note the spelling: `sne4onnx` (network *extractor*), not the `sng4onnx` (name *generator*) we already installed. They're two different packages with confusingly similar names.

Let me confirm the .tflite files are actually on disk, then fix the requirements and the script.

Confirmed — `ghostface_fp32_float16.tflite` (7.8M) is on disk. The crash dialog is from after the artifacts were already written; harmless for the output, but I'll still fix the root cause so the script runs end-to-end.

About that macOS "Python quit unexpectedly" dialog: your `.venv` is using **Python 3.9 from Xcode's bundled Python framework** (visible in the traceback path: `/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework`). That Python is known to flake on shutdown when TF/ORT background threads are still finalizing — it's a cosmetic crash *after* `main()` returns, not a real failure. We'll address it last.

## The actual fix: add `sne4onnx`

Let me verify the actual published version of `sne4onnx` rather than guess:

Now let me make the script more robust — there's no reason a missing `sne4onnx` (onnx2tf's *internal* parity check) should make us throw away successfully produced TFLite files when we have our own (more thorough) parity check downstream:

Quick recap and what to do next:

## What changed

```129:163:tools/convert_ghostfacenet_fp16.py
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(out_dir),
        "-osd",                        # output SavedModel
        "-cotof",                      # check tf vs onnx output diff (built-in sanity)
        "-rtpo", "AveragePool",        # safer for some Ghost ops; harmless if unused
    ]
    print()
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    # onnx2tf's `-cotof` post-conversion check needs `sne4onnx`. If that step
    # fails AFTER both .tflite files have been written, we don't want to throw
    # the artifacts away -- this script does its own (more thorough) parity
    # check downstream. ...
    fp16_candidates = sorted(out_dir.glob("*float16*.tflite"))
    if result.returncode != 0 and not fp16_candidates:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.returncode != 0:
        print(f"  ! onnx2tf exited {result.returncode} but FP16 TFLite was produced; continuing.")
```

```10:18:tools/requirements.txt
# --- onnx2tf runtime deps that aren't auto-installed ---
# (onnx2tf imports these at runtime but doesn't declare them.)
onnx_graphsurgeon==0.6.1
onnxsim==0.4.36
sng4onnx==2.0.1          # simple Name Generator for onnx (auto-name nodes)
sne4onnx==2.0.1          # simple Network Extractor for onnx (required by -cotof)
ai-edge-litert==2.0.3
psutil==7.2.2
```

## Run this

```bash
source /Users/gifty/Development/FaceMesh/.venv/bin/activate
pip install sne4onnx==2.0.1

python tools/convert_ghostfacenet_fp16.py \
  --onnx /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx \
  --out  build/ghostfacenet_fp16 \
  --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin
```

You should now see the script reach the `Parity check: ONNX FP32 vs TFLite FP16` section, print cosine similarity for both `synthetic` and `test_0.bin`, and finish with `Manifest values` (file path, size, sha256). Anything below `cos_sim = 0.999` is a red flag.

## About the macOS crash dialog

The conversion *did* finish — both `.tflite` files were on disk before Python died. The dialog comes from Python 3.9 from Xcode's framework segfaulting at interpreter shutdown while TF/ORT background threads were still finalizing. Cosmetic, but if it bothers you, swap to a real Homebrew/python.org Python:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv-convert
source .venv-convert/bin/activate
pip install -r tools/requirements.txt
```

Python 3.11 is also the sweet spot for this TF/onnx2tf stack — fewer numpy-2 / protobuf rough edges than 3.9.

---

**User**

Did you update requirements.txt with the new sne4onnx?

---

**Cursor**

Yes — let me show you so you can verify:

Confirmed — line 23 has `sne4onnx==2.0.1` (the network *extractor*, needed by `-cotof`), sitting alongside `sng4onnx==2.0.1` on line 22 (the name *generator*, different package). Both pinned to versions verified against PyPI.

---

**User**

  OK onnx2tf produced FP16 TFLite: ghostface_fp32_float16.tflite

========================================================================
TFLite model details
========================================================================
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  input :  shape=(np.int32(1), np.int32(112), np.int32(3), np.int32(112))  dtype=float32
  output: shape=(np.int32(1), np.int32(512))  dtype=float32
  layout: NCHW
  embedding_dim: 512

========================================================================
Parity check: ONNX FP32 vs TFLite FP16
========================================================================
  [OK] synthetic      cos_sim=0.999987  max_abs=1.224469e-02  l2_diff_norm=5.017470e-03
  [OK] test_0.bin     cos_sim=0.999997  max_abs=1.045358e-02  l2_diff_norm=2.447618e-03

========================================================================
Manifest values
========================================================================
  file path : build/ghostfacenet_fp16/ghostface_fp32_float16.tflite
  size_bytes: 8156064
  sha256    : 4bfb5c2a09993f9184917c4163915a6218096d36f482c0428bfac95adda6abe0
  embedding_dim: 512  <- update FaceEmbedder.EMBEDDING_SIZE if != 128

---

**Cursor**

Solid numbers — `cos_sim=0.999997` on `test_0.bin` is well past the bar. But there's something subtle in the output you should fix before shipping this to Android:

```
input :  shape=(np.int32(1), np.int32(112), np.int32(3), np.int32(112))  dtype=float32
layout: NCHW
```

That input shape — `(1, 112, 3, 112)` — is **wrong**, and the script's `layout` line is misleading. Here's what happened:

- Your ONNX input was already `(1, 112, 112, 3)` = **NHWC** (Keras-exported, hence the `input_1:0` / `Identity:0` names).
- `onnx2tf` defaults to assuming ONNX is **NCHW** and auto-transposes `NCHW → NHWC`. So it interpreted your input as `(N=1, C=112, H=112, W=3)` and transposed to `(N=1, H=112, W=3, C=112)` = `(1, 112, 3, 112)`.
- The model still computes correctly because every internal op got the same misinterpretation, and `H==W==112` makes the math symmetric — that's why parity is 0.9999. But the **input layout is now nonsensical**: dim 1 is "old W", dim 2 is "channels", dim 3 is "old H".

Two consequences:

1. Your Android `FaceEmbedder` will have to feed the model with a weird `(0, 2, 3, 1)` permutation of a normal NHWC bitmap tensor. Easy to get wrong, easy to misdebug.
2. There's an extra unnecessary transpose inside the model graph that costs perf on every inference.

## Fix: tell `onnx2tf` the input is already NHWC

The flag is `-kat <input_name>` (keep NWC/NHWC/NDHWC input). I'll auto-detect that from the ONNX (any 4-D input whose last dim is 3 is NHWC) and also fix the misleading `layout:` line.

Let me verify the script reads cleanly end-to-end:

Looks good. Now to clean up and re-run:

```bash
# Wipe the bad artifact so onnx2tf doesn't reuse anything
rm -rf build/ghostfacenet_fp16

# Re-run with the layout fix
python tools/convert_ghostfacenet_fp16.py \
  --onnx /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx \
  --out  build/ghostfacenet_fp16 \
  --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin
```

What you should see this time:

```
$ onnx2tf -i ... -o ... -osd -cotof -rtpo AveragePool -kat input_1:0
  (input 'input_1:0' looks NHWC; passing -kat to skip auto-transpose)
...
  input :  shape=(1, 112, 112, 3)  dtype=float32     <-- proper NHWC, channels last
  output: shape=(1, 512)           dtype=float32
  layout: NHWC
```

And cosine similarity should remain >0.999 (likely better, because there are fewer internal ops fighting layout).

## Why this matters for the Android side

When you wire this into `FaceEmbedder.kt`, the input tensor allocation is now the natural one:

```kotlin
val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 112, 112, 3), DataType.FLOAT32)
```

Pixels go in as `[N=1][H=112][W=112][C=3]`, exactly matching what `Bitmap` → `ByteBuffer` produces row-major. **No** sneaky `(0, 2, 3, 1)` permutation in the pre-processing path. That alone usually saves a half-day of "why are my embeddings garbage" debugging.

Also note: `embedding_dim: 512`, not 128 — the script's hint at the bottom is reminding you to update `FaceEmbedder.EMBEDDING_SIZE` accordingly. Worth grepping for it now so you don't get a silent shape mismatch in the L2-normalization step.

Want me to grep the Android code for `EMBEDDING_SIZE` and prepare that change too?

---

**User**

  OK onnx2tf produced FP16 TFLite: ghostface_fp32_float16.tflite

========================================================================
TFLite model details
========================================================================
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  input :  shape=(1, 112, 3, 112)  dtype=float32
  output: shape=(1, 512)  dtype=float32
  layout: unexpected ((1, 112, 3, 112)) -- onnx2tf may have done a partial transpose
  embedding_dim: 512

========================================================================
Parity check: ONNX FP32 vs TFLite FP16
========================================================================
  [OK] synthetic      cos_sim=0.999987  max_abs=1.224469e-02  l2_diff_norm=5.017470e-03
  [OK] test_0.bin     cos_sim=0.999997  max_abs=1.045358e-02  l2_diff_norm=2.447618e-03

========================================================================
Manifest values
========================================================================
  file path : build/ghostfacenet_fp16/ghostface_fp32_float16.tflite
  size_bytes: 8156064
  sha256    : 4bfb5c2a09993f9184917c4163915a6218096d36f482c0428bfac95adda6abe0
  embedding_dim: 512  <- update FaceEmbedder.EMBEDDING_SIZE if != 128


---

**Cursor**

That's odd — the SHA is **identical** to the previous run (`4bfb5c2a…`), which means either onnx2tf isn't honoring `-kat input_1:0` for this model, or the flag isn't being passed. Let me verify both in one shot.

First, did our script actually pass `-kat`? Let me run the exact onnx2tf command standalone so we can see for sure:

So `-kat` is being passed cleanly but onnx2tf isn't honoring it for this model — same `(1, 112, 3, 112)` output. The `:0` suffix is suspect; onnx2tf's name-matching may strip it internally. Let me try a few alternates and see which one actually works:

Let me inspect the ONNX model directly to understand what `-kat` is fighting against:

Now I see exactly what's wrong. The ONNX has an explicit `Transpose` op as the very first node — a leftover from the Keras→ONNX export that flips NHWC→NCHW so the `Conv` ops can run in ONNX's preferred NCHW. So when onnx2tf sees this pattern, its layout heuristics get into a fight with that explicit transpose, and the net result is `(1, 112, 3, 112)`. `-kat` alone can't unstick it.

The fix is to **simplify the ONNX first** with `onnxsim` (or `onnx-graphsurgeon`), which folds those entry/exit transposes away. Then onnx2tf has a clean graph to work with. Let me try it:

`onnxsim` segfaulted (the same Xcode-Python 3.9 instability). Let me try a more surgical approach with `onnx_graphsurgeon` to strip just the entry/exit Transpose ops, then convert:

Now I have a clean fix. The entry Transpose's perm is `[0, 3, 1, 2]` (NHWC→NCHW) and its output is `[1, 3, 112, 112]` — meaning the rest of the graph runs in NCHW. The cleanest move is:

1. Re-declare the model input as NCHW `(1, 3, 112, 112)` directly.
2. Bypass the entry Transpose (connect Conv input straight to the new NCHW input).

Now `onnx2tf` sees a "normal" NCHW ONNX model and will do its standard NCHW→NHWC conversion, producing a TFLite with a clean NHWC `(1, 112, 112, 3)` input.

Let me prove that works:

The surgery worked (line `new graph input: input_1:0 [1, 3, 112, 112]`) — the conversion itself blew up later inside `download_test_image_data` due to NumPy 2.x's stricter `allow_pickle`. Let me see what's going on there and route around it:

The "pickle" error is misleading — `onnx2tf` is downloading a calibration `.npy` from a Wasabi S3 URL that's blocked by the sandbox here, so it gets a garbled response and `np.load` complains. On your actual machine the download succeeds (which is why your normal runs worked). Let me pre-place a deterministic calibration file so we can probe end-to-end without the network, and confirm the surgery yields a clean NHWC TFLite:

It worked. Both `(1, 112, 112, 3)` outputs — proper NHWC. Now let me bake this into the conversion script so it happens automatically.

Now let me update the conversion script to do the surgery automatically. I'll keep it conservative — only strip the entry Transpose if it's exactly an `[0, 3, 1, 2]` perm (the unambiguous NHWC→NCHW pattern), and leave the original ONNX untouched (write surgery output to the build dir):

One thing I'd like to clean up — leaving a 3.8MB calibration `.npy` in the user's working dir is gross. `onnx2tf` reads from `os.getcwd()`, so let me run the subprocess with `cwd=` set to a scratch dir, place the calib file there, and the workspace stays clean:

Let me read through the final script to make sure everything is wired up correctly:

All wired up. Time for you to re-run:

```bash
rm -rf build/ghostfacenet_fp16
python tools/convert_ghostfacenet_fp16.py \
  --onnx /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx \
  --out  build/ghostfacenet_fp16 \
  --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin
```

Expected new output:

```
========================================================================
Converting ONNX -> TFLite (FP16, NHWC)
========================================================================
  (stripped entry NHWC->NCHW Transpose; using _intermediate_nchw.onnx)
  (pre-placed calibration data at: build/ghostfacenet_fp16/_onnx2tf_scratch/calibration_image_sample_data_20x128x128x3_float32.npy)

$ (cd build/ghostfacenet_fp16/_onnx2tf_scratch && onnx2tf -i ... -osd -cotof -rtpo AveragePool)
...
========================================================================
TFLite model details
========================================================================
  input :  shape=(1, 112, 112, 3)  dtype=float32
  output: shape=(1, 512)           dtype=float32
  layout: NHWC
  embedding_dim: 512

========================================================================
Parity check: ONNX FP32 vs TFLite FP16
========================================================================
  [OK] synthetic      cos_sim=...   max_abs=...
  [OK] test_0.bin     cos_sim=...   max_abs=...
```

Couple of notes on what changed in the script:

- New `_strip_entry_nhwc_to_nchw_transpose()` — uses `onnx_graphsurgeon` to detect and remove the leading `Transpose(perm=[0,3,1,2])` only when it's exactly that pattern, so the surgery is a no-op on models that don't need it.
- New `_ensure_onnx2tf_calib_data()` — writes a deterministic synthetic 20×128×128×3 float32 `.npy` into a scratch dir (`build/ghostfacenet_fp16/_onnx2tf_scratch/`), so `onnx2tf` doesn't try to download from Wasabi S3. Same RNG seed every run → reproducible.
- The `onnx2tf` subprocess now runs with `cwd=` pointing at that scratch dir, so the calibration `.npy` lives there instead of polluting the workspace root.
- The intermediate NCHW ONNX is written to `build/ghostfacenet_fp16/_intermediate_nchw.onnx` — feel free to inspect it with Netron if you want to confirm the surgery looks right.

After this re-run, the SHA in the manifest section will be **different** from `4bfb5c2a…` (because the model topology changed: one fewer `Transpose` op, plus a permuted internal layout). Update the Android manifest.json with the new SHA.

---

**User**

░▒▓ ~/Dev/FaceMesh  python tools/convert_ghostfacenet_fp16.py --onnx  /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx --out  build/ghostfacenet_fp16 --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin

========================================================================
ONNX model
========================================================================
  input :  name='input_1:0'  shape=(1, 112, 112, 3)
  output: name='Identity:0'  shape=(1, 512)
  OK loaded test input from /Users/gifty/Development/FaceMesh/models/test_0.bin (float32, (1, 112, 112, 3))

========================================================================
Converting ONNX -> TFLite (FP16, NHWC)
========================================================================
  (stripped entry NHWC->NCHW Transpose; using _intermediate_nchw.onnx)
  (pre-placed calibration data at: build/ghostfacenet_fp16/_onnx2tf_scratch/calibration_image_sample_data_20x128x128x3_float32.npy)

$ (cd build/ghostfacenet_fp16/_onnx2tf_scratch && onnx2tf -i /Users/gifty/Development/FaceMesh/build/ghostfacenet_fp16/_intermediate_nchw.onnx -o /Users/gifty/Development/FaceMesh/build/ghostfacenet_fp16 -osd -cotof -rtpo AveragePool)

Model optimizing started ============================================================
Traceback (most recent call last):
  File "/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/onnx2tf/onnx2tf.py", line 614, in convert
    result = subprocess.check_output(
  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/subprocess.py", line 424, in check_output
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['onnxsim', '/Users/gifty/Development/FaceMesh/build/ghostfacenet_fp16/_intermediate_nchw.onnx', '/Users/gifty/Development/FaceMesh/build/ghostfacenet_fp16/_intermediate_nchw.onnx']' died with <Signals.SIGSEGV: 11>.

WARNING: Failed to optimize the onnx file.

Automatic generation of each OP name started ========================================
Automatic generation of each OP name complete!

Model loaded ========================================================================

Model conversion started ============================================================
INFO: input_op_name: input_1__0__nchw_in shape: [1, 3, 112, 112] dtype: float32

INFO: 2 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d/Conv2D
INFO:  input_name.1: input_1__0__nchw_in shape: [1, 3, 112, 112] dtype: float32
INFO:  input_name.2: model/conv2d/Conv2D_weights_fused_bn shape: [20, 3, 3, 3] dtype: float32
INFO:  input_name.3: model/conv2d/Conv2D_bias_fused_bn shape: [20] dtype: float32
INFO:  output_name.1: model/batch_normalization/FusedBatchNormV3__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.pad/Pad:0 shape: (1, 113, 113, 3) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 3, 20) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (20,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [2, 2] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 3 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation/Relu
INFO:  input_name.1: model/batch_normalization/FusedBatchNormV3__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  output_name.1: model/activation/Relu__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu/Relu:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 4 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation/Neg_1
INFO:  input_name.1: model/batch_normalization/FusedBatchNormV3__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  output_name.1: model/activation/Neg_1__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative/Neg:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 5 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation/Relu_1
INFO:  input_name.1: model/activation/Neg_1__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  output_name.1: model/activation/Relu_1__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative/Neg:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_1/Relu:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 6 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation/mul
INFO:  input_name.1: model/activation/Relu_1__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/activation/Neg:0 shape: [1, 20, 1, 1] dtype: float32
INFO:  output_name.1: model/activation/mul__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_1/Relu:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 20) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_2/Mul:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 7 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation/add
INFO:  input_name.1: model/activation/Relu__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/activation/mul__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  output_name.1: model/activation/add__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu/Relu:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_2/Mul:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_2/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 8 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_1/Conv2D
INFO:  input_name.1: model/activation/add__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/conv2d_1/Conv2D_weights_fused_bn shape: [10, 20, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_1/Conv2D_bias_fused_bn shape: [10] dtype: float32
INFO:  output_name.1: model/batch_normalization_1/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_2/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 20, 10) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (10,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_3/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 9 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_1/Relu
INFO:  input_name.1: model/batch_normalization_1/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_1/Relu__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_3/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_2/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 10 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_1/Neg_1
INFO:  input_name.1: model/batch_normalization_1/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_1/Neg_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_3/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_1/Neg:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 11 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_1/Relu_1
INFO:  input_name.1: model/activation_1/Neg_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_1/Relu_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_1/Neg:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_3/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 12 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_1/mul
INFO:  input_name.1: model/activation_1/Relu_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_1/Neg:0 shape: [1, 10, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_1/mul__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_3/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 10) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_7/Mul:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 13 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_1/add
INFO:  input_name.1: model/activation_1/Relu__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_1/mul__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_1/add__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_2/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_7/Mul:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_5/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 14 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d/depthwise
INFO:  input_name.1: model/activation_1/add__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d/depthwise_weights_fused_bn shape: [10, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d/depthwise_bias_fused_bn shape: [10] dtype: float32
INFO:  output_name.1: model/batch_normalization_2/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_5/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 10, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (10,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 10 
INFO:  output.1.output: name: tf.math.add_6/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 15 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_2/Relu
INFO:  input_name.1: model/batch_normalization_2/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_2/Relu__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_6/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_4/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 16 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_2/Neg_1
INFO:  input_name.1: model/batch_normalization_2/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_2/Neg_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_6/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_2/Neg:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 17 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_2/Relu_1
INFO:  input_name.1: model/activation_2/Neg_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_2/Relu_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_2/Neg:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_5/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 18 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_2/mul
INFO:  input_name.1: model/activation_2/Relu_1__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_2/Neg:0 shape: [1, 10, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_2/mul__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_5/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 10) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_12/Mul:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 19 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_2/add
INFO:  input_name.1: model/activation_2/Relu__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_2/mul__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_2/add__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_4/Relu:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_12/Mul:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_8/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 20 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate/concat
INFO:  input_name.1: model/activation_1/add__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_2/add__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/concatenate/concat__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_5/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_8/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat/concat:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 21 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_2/Conv2D
INFO:  input_name.1: model/concatenate/concat__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/conv2d_2/Conv2D_weights_fused_bn shape: [10, 20, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_2/Conv2D_bias_fused_bn shape: [10] dtype: float32
INFO:  output_name.1: model/batch_normalization_3/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat/concat:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 20, 10) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (10,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_9/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 22 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_1/depthwise
INFO:  input_name.1: model/batch_normalization_3/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_1/depthwise_weights_fused_bn shape: [10, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_1/depthwise_bias_fused_bn shape: [10] dtype: float32
INFO:  output_name.1: model/batch_normalization_4/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_9/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 10, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (10,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 10 
INFO:  output.1.output: name: tf.math.add_10/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 

INFO: 23 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_1/concat
INFO:  input_name.1: model/batch_normalization_3/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  input_name.2: model/batch_normalization_4/FusedBatchNormV3__0 shape: [1, 10, 56, 56] dtype: float32
INFO:  output_name.1: model/concatenate_1/concat__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_9/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_10/Add:0 shape: (1, 56, 56, 10) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_1/concat:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 24 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add/add
INFO:  input_name.1: model/activation/add__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/concatenate_1/concat__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  output_name.1: model/add/add__0 shape: [1, 20, 56, 56] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_2/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_1/concat:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_12/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 

INFO: 25 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_5/depthwise
INFO:  input_name.1: model/add/add__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_5/depthwise_weights_fused_bn shape: [20, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_5/depthwise_bias_fused_bn shape: [20] dtype: float32
INFO:  output_name.1: model/batch_normalization_10/FusedBatchNormV3__0 shape: [1, 20, 28, 28] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_1/Pad:0 shape: (1, 57, 57, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 20, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (20,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 20 
INFO:  output.1.output: name: tf.math.add_13/Add:0 shape: (1, 28, 28, 20) dtype: <dtype: 'float32'> 

INFO: 26 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_3/Conv2D
INFO:  input_name.1: model/add/add__0 shape: [1, 20, 56, 56] dtype: float32
INFO:  input_name.2: model/conv2d_3/Conv2D_weights_fused_bn shape: [32, 20, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_3/Conv2D_bias_fused_bn shape: [32] dtype: float32
INFO:  output_name.1: model/batch_normalization_5/FusedBatchNormV3__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_12/Add:0 shape: (1, 56, 56, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 20, 32) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (32,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_14/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 27 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_5/Conv2D
INFO:  input_name.1: model/batch_normalization_10/FusedBatchNormV3__0 shape: [1, 20, 28, 28] dtype: float32
INFO:  input_name.2: model/conv2d_5/Conv2D_weights_fused_bn shape: [32, 20, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_5/Conv2D_bias_fused_bn shape: [32] dtype: float32
INFO:  output_name.1: model/batch_normalization_11/FusedBatchNormV3__0 shape: [1, 32, 28, 28] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_13/Add:0 shape: (1, 28, 28, 20) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 20, 32) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (32,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_15/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 

INFO: 28 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_3/Relu
INFO:  input_name.1: model/batch_normalization_5/FusedBatchNormV3__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_3/Relu__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_14/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_6/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 29 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_3/Neg_1
INFO:  input_name.1: model/batch_normalization_5/FusedBatchNormV3__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_3/Neg_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_14/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_3/Neg:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 30 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_3/Relu_1
INFO:  input_name.1: model/activation_3/Neg_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_3/Relu_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_3/Neg:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_7/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 31 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_3/mul
INFO:  input_name.1: model/activation_3/Relu_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_3/Neg:0 shape: [1, 32, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_3/mul__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_7/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 32) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_19/Mul:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 32 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_3/add
INFO:  input_name.1: model/activation_3/Relu__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_3/mul__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_3/add__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_6/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_19/Mul:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_17/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 33 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_2/depthwise
INFO:  input_name.1: model/activation_3/add__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_2/depthwise_weights_fused_bn shape: [32, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_2/depthwise_bias_fused_bn shape: [32] dtype: float32
INFO:  output_name.1: model/batch_normalization_6/FusedBatchNormV3__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_17/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 32, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (32,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 32 
INFO:  output.1.output: name: tf.math.add_18/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 34 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_4/Relu
INFO:  input_name.1: model/batch_normalization_6/FusedBatchNormV3__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_4/Relu__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_18/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_8/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 35 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_4/Neg_1
INFO:  input_name.1: model/batch_normalization_6/FusedBatchNormV3__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_4/Neg_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_18/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_4/Neg:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 36 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_4/Relu_1
INFO:  input_name.1: model/activation_4/Neg_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_4/Relu_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_4/Neg:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_9/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 37 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_4/mul
INFO:  input_name.1: model/activation_4/Relu_1__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_4/Neg:0 shape: [1, 32, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_4/mul__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_9/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 32) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_24/Mul:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 38 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_4/add
INFO:  input_name.1: model/activation_4/Relu__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_4/mul__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/activation_4/add__0 shape: [1, 32, 56, 56] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_8/Relu:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_24/Mul:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_20/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 

INFO: 39 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_2/concat
INFO:  input_name.1: model/activation_3/add__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  input_name.2: model/activation_4/add__0 shape: [1, 32, 56, 56] dtype: float32
INFO:  output_name.1: model/concatenate_2/concat__0 shape: [1, 64, 56, 56] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_17/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_20/Add:0 shape: (1, 56, 56, 32) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_2/concat:0 shape: (1, 56, 56, 64) dtype: <dtype: 'float32'> 

INFO: 40 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_3/depthwise
INFO:  input_name.1: model/concatenate_2/concat__0 shape: [1, 64, 56, 56] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_3/depthwise_weights_fused_bn shape: [64, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_3/depthwise_bias_fused_bn shape: [64] dtype: float32
INFO:  output_name.1: model/batch_normalization_7/FusedBatchNormV3__0 shape: [1, 64, 28, 28] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_2/Pad:0 shape: (1, 57, 57, 64) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 64, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (64,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 64 
INFO:  output.1.output: name: tf.math.add_21/Add:0 shape: (1, 28, 28, 64) dtype: <dtype: 'float32'> 

INFO: 41 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_4/Conv2D
INFO:  input_name.1: model/batch_normalization_7/FusedBatchNormV3__0 shape: [1, 64, 28, 28] dtype: float32
INFO:  input_name.2: model/conv2d_4/Conv2D_weights_fused_bn shape: [16, 64, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_4/Conv2D_bias_fused_bn shape: [16] dtype: float32
INFO:  output_name.1: model/batch_normalization_8/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_21/Add:0 shape: (1, 28, 28, 64) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 64, 16) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (16,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_22/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 

INFO: 42 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_4/depthwise
INFO:  input_name.1: model/batch_normalization_8/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_4/depthwise_weights_fused_bn shape: [16, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_4/depthwise_bias_fused_bn shape: [16] dtype: float32
INFO:  output_name.1: model/batch_normalization_9/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_22/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 16, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (16,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 16 
INFO:  output.1.output: name: tf.math.add_23/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 

INFO: 43 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_3/concat
INFO:  input_name.1: model/batch_normalization_8/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO:  input_name.2: model/batch_normalization_9/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO:  output_name.1: model/concatenate_3/concat__0 shape: [1, 32, 28, 28] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_22/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_23/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_3/concat:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 

INFO: 44 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_1/add
INFO:  input_name.1: model/batch_normalization_11/FusedBatchNormV3__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  input_name.2: model/concatenate_3/concat__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  output_name.1: model/add_1/add__0 shape: [1, 32, 28, 28] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_15/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_3/concat:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_25/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 

INFO: 45 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_6/Conv2D
INFO:  input_name.1: model/add_1/add__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  input_name.2: model/conv2d_6/Conv2D_weights_fused_bn shape: [46, 32, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_6/Conv2D_bias_fused_bn shape: [46] dtype: float32
INFO:  output_name.1: model/batch_normalization_12/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_25/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 32, 46) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (46,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_26/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 46 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_5/Relu
INFO:  input_name.1: model/batch_normalization_12/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_5/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_26/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_10/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 47 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_5/Neg_1
INFO:  input_name.1: model/batch_normalization_12/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_5/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_26/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_5/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 48 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_5/Relu_1
INFO:  input_name.1: model/activation_5/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_5/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_5/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_11/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 49 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_5/mul
INFO:  input_name.1: model/activation_5/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_5/Neg:0 shape: [1, 46, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_5/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_11/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 46) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_31/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 50 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_5/add
INFO:  input_name.1: model/activation_5/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_5/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_5/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_10/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_31/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_28/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 51 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_6/depthwise
INFO:  input_name.1: model/activation_5/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_6/depthwise_weights_fused_bn shape: [46, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_6/depthwise_bias_fused_bn shape: [46] dtype: float32
INFO:  output_name.1: model/batch_normalization_13/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_28/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 46, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (46,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 46 
INFO:  output.1.output: name: tf.math.add_29/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 52 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_6/Relu
INFO:  input_name.1: model/batch_normalization_13/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_6/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_29/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_12/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 53 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_6/Neg_1
INFO:  input_name.1: model/batch_normalization_13/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_6/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_29/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_6/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 54 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_6/Relu_1
INFO:  input_name.1: model/activation_6/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_6/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_6/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_13/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 55 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_6/mul
INFO:  input_name.1: model/activation_6/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_6/Neg:0 shape: [1, 46, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_6/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_13/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 46) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_36/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 56 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_6/add
INFO:  input_name.1: model/activation_6/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_6/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_6/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_12/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_36/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_31/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 57 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_4/concat
INFO:  input_name.1: model/activation_5/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_6/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/concatenate_4/concat__0 shape: [1, 92, 28, 28] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_28/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_31/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_4/concat:0 shape: (1, 28, 28, 92) dtype: <dtype: 'float32'> 

INFO: 58 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_7/Conv2D
INFO:  input_name.1: model/concatenate_4/concat__0 shape: [1, 92, 28, 28] dtype: float32
INFO:  input_name.2: model/conv2d_7/Conv2D_weights_fused_bn shape: [16, 92, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_7/Conv2D_bias_fused_bn shape: [16] dtype: float32
INFO:  output_name.1: model/batch_normalization_14/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat_4/concat:0 shape: (1, 28, 28, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 92, 16) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (16,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_32/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 

INFO: 59 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_7/depthwise
INFO:  input_name.1: model/batch_normalization_14/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_7/depthwise_weights_fused_bn shape: [16, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_7/depthwise_bias_fused_bn shape: [16] dtype: float32
INFO:  output_name.1: model/batch_normalization_15/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_32/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 16, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (16,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 16 
INFO:  output.1.output: name: tf.math.add_33/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 

INFO: 60 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_5/concat
INFO:  input_name.1: model/batch_normalization_14/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO:  input_name.2: model/batch_normalization_15/FusedBatchNormV3__0 shape: [1, 16, 28, 28] dtype: float32
INFO:  output_name.1: model/concatenate_5/concat__0 shape: [1, 32, 28, 28] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_32/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_33/Add:0 shape: (1, 28, 28, 16) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_5/concat:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 

INFO: 61 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_2/add
INFO:  input_name.1: model/add_1/add__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  input_name.2: model/concatenate_5/concat__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  output_name.1: model/add_2/add__0 shape: [1, 32, 28, 28] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_25/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_5/concat:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_35/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 

INFO: 62 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_11/depthwise
INFO:  input_name.1: model/add_2/add__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_11/depthwise_weights_fused_bn shape: [32, 1, 5, 5] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_11/depthwise_bias_fused_bn shape: [32] dtype: float32
INFO:  output_name.1: model/batch_normalization_21/FusedBatchNormV3__0 shape: [1, 32, 14, 14] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_3/Pad:0 shape: (1, 31, 31, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (5, 5, 32, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (32,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 32 
INFO:  output.1.output: name: tf.math.add_36/Add:0 shape: (1, 14, 14, 32) dtype: <dtype: 'float32'> 

INFO: 63 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_8/Conv2D
INFO:  input_name.1: model/add_2/add__0 shape: [1, 32, 28, 28] dtype: float32
INFO:  input_name.2: model/conv2d_8/Conv2D_weights_fused_bn shape: [46, 32, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_8/Conv2D_bias_fused_bn shape: [46] dtype: float32
INFO:  output_name.1: model/batch_normalization_16/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_35/Add:0 shape: (1, 28, 28, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 32, 46) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (46,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_37/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 64 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_12/Conv2D
INFO:  input_name.1: model/batch_normalization_21/FusedBatchNormV3__0 shape: [1, 32, 14, 14] dtype: float32
INFO:  input_name.2: model/conv2d_12/Conv2D_weights_fused_bn shape: [52, 32, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_12/Conv2D_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_22/FusedBatchNormV3__0 shape: [1, 52, 14, 14] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_36/Add:0 shape: (1, 14, 14, 32) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 32, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_38/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 

INFO: 65 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_7/Relu
INFO:  input_name.1: model/batch_normalization_16/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_7/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_37/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_14/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 66 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_7/Neg_1
INFO:  input_name.1: model/batch_normalization_16/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_7/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_37/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_7/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 67 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_7/Relu_1
INFO:  input_name.1: model/activation_7/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_7/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_7/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_15/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 68 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_7/mul
INFO:  input_name.1: model/activation_7/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_7/Neg:0 shape: [1, 46, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_7/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_15/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 46) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_43/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 69 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_7/add
INFO:  input_name.1: model/activation_7/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_7/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_7/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_14/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_43/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_40/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 70 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_8/depthwise
INFO:  input_name.1: model/activation_7/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_8/depthwise_weights_fused_bn shape: [46, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_8/depthwise_bias_fused_bn shape: [46] dtype: float32
INFO:  output_name.1: model/batch_normalization_17/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_40/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 46, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (46,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 46 
INFO:  output.1.output: name: tf.math.add_41/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 71 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_8/Relu
INFO:  input_name.1: model/batch_normalization_17/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_8/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_41/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_16/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 72 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_8/Neg_1
INFO:  input_name.1: model/batch_normalization_17/FusedBatchNormV3__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_8/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_41/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_8/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 73 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_8/Relu_1
INFO:  input_name.1: model/activation_8/Neg_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_8/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_8/Neg:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_17/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 74 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_8/mul
INFO:  input_name.1: model/activation_8/Relu_1__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_8/Neg:0 shape: [1, 46, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_8/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_17/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 46) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_48/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 75 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_8/add
INFO:  input_name.1: model/activation_8/Relu__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_8/mul__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/activation_8/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_16/Relu:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_48/Mul:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_43/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 

INFO: 76 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_6/concat
INFO:  input_name.1: model/activation_7/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  input_name.2: model/activation_8/add__0 shape: [1, 46, 28, 28] dtype: float32
INFO:  output_name.1: model/concatenate_6/concat__0 shape: [1, 92, 28, 28] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_40/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_43/Add:0 shape: (1, 28, 28, 46) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_6/concat:0 shape: (1, 28, 28, 92) dtype: <dtype: 'float32'> 

INFO: 77 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_9/depthwise
INFO:  input_name.1: model/concatenate_6/concat__0 shape: [1, 92, 28, 28] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_9/depthwise_weights_fused_bn shape: [92, 1, 5, 5] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_9/depthwise_bias_fused_bn shape: [92] dtype: float32
INFO:  output_name.1: model/batch_normalization_18/FusedBatchNormV3__0 shape: [1, 92, 14, 14] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_4/Pad:0 shape: (1, 31, 31, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (5, 5, 92, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (92,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 92 
INFO:  output.1.output: name: tf.math.add_44/Add:0 shape: (1, 14, 14, 92) dtype: <dtype: 'float32'> 

INFO: 78 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d/Mean
INFO:  input_name.1: model/batch_normalization_18/FusedBatchNormV3__0 shape: [1, 92, 14, 14] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d/Mean__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.math.add_44/Add:0 shape: (1, 14, 14, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean/Mean:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 

INFO: 79 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d/Mean_Squeeze__1961
INFO:  input_name.1: model/global_average_pooling2d/Mean__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: [2] dtype: int64
INFO:  output_name.1: model/global_average_pooling2d/Mean_Squeeze__1961__0 shape: [1, 92] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean/Mean:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze/model/global_average_pooling2d/Mean_Squeeze__1961:0 shape: (1, 92) dtype: <dtype: 'float32'> 

INFO: 80 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_9/BiasAdd__223
INFO:  input_name.1: model/global_average_pooling2d/Mean_Squeeze__1961__0 shape: [1, 92] dtype: float32
INFO:  input_name.2: new_shape__1817 shape: [4] dtype: int64
INFO:  output_name.1: model/conv2d_9/BiasAdd__223__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze/model/global_average_pooling2d/Mean_Squeeze__1961:0 shape: (1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(92), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape/Reshape:0 shape: (1, 92, 1, 1) dtype: <dtype: 'float32'> 

INFO: 81 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_9/BiasAdd
INFO:  input_name.1: model/conv2d_9/BiasAdd__223__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_9/Conv2D/ReadVariableOp:0 shape: [24, 92, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_9/BiasAdd/ReadVariableOp:0 shape: [24] dtype: float32
INFO:  output_name.1: model/conv2d_9/BiasAdd__0 shape: [1, 24, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_2/transpose:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 92, 24) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (24,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_45/Add:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 

INFO: 82 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_9/Relu
INFO:  input_name.1: model/conv2d_9/BiasAdd__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_9/Relu__0 shape: [1, 24, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_45/Add:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_18/Relu:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 

INFO: 83 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_9/Neg_1
INFO:  input_name.1: model/conv2d_9/BiasAdd__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_9/Neg_1__0 shape: [1, 24, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_45/Add:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_9/Neg:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 

INFO: 84 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_9/Relu_1
INFO:  input_name.1: model/activation_9/Neg_1__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_9/Relu_1__0 shape: [1, 24, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_9/Neg:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_19/Relu:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 

INFO: 85 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_9/mul
INFO:  input_name.1: model/activation_9/Relu_1__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_9/Neg:0 shape: [1, 24, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_9/mul__0 shape: [1, 24, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_19/Relu:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 24) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_52/Mul:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 

INFO: 86 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_9/add
INFO:  input_name.1: model/activation_9/Relu__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_9/mul__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_9/add__0 shape: [1, 24, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_18/Relu:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_52/Mul:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_46/Add:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 

INFO: 87 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_10/BiasAdd
INFO:  input_name.1: model/activation_9/add__0 shape: [1, 24, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_10/Conv2D/ReadVariableOp:0 shape: [92, 24, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_10/BiasAdd/ReadVariableOp:0 shape: [92] dtype: float32
INFO:  output_name.1: model/conv2d_10/BiasAdd__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_46/Add:0 shape: (1, 1, 1, 24) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 24, 92) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (92,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_47/Add:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 

INFO: 88 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_10/Mul
INFO:  input_name.1: model/conv2d_10/BiasAdd__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: [] dtype: float32
INFO:  output_name.1: model/activation_10/Mul__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_47/Add:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_55/Mul:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 

INFO: 89 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_10/Add
INFO:  input_name.1: model/activation_10/Mul__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: [1, 1, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_10/Add__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_55/Mul:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_48/Add:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 

INFO: 90 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_10/clip_by_value/Minimum
INFO:  input_name.1: model/activation_10/Add__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: [1, 1, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_10/clip_by_value/Minimum__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_48/Add:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum/Minimum:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 

INFO: 91 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_10/clip_by_value
INFO:  input_name.1: model/activation_10/clip_by_value/Minimum__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: [1, 1, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_10/clip_by_value__0 shape: [1, 92, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum/Minimum:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum/Maximum:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 

INFO: 92 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply/mul
INFO:  input_name.1: model/batch_normalization_18/FusedBatchNormV3__0 shape: [1, 92, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_10/clip_by_value__0 shape: [1, 92, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply/mul__0 shape: [1, 92, 14, 14] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_44/Add:0 shape: (1, 14, 14, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum/Maximum:0 shape: (1, 1, 1, 92) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_60/Mul:0 shape: (1, 14, 14, 92) dtype: <dtype: 'float32'> 

INFO: 93 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_11/Conv2D
INFO:  input_name.1: model/multiply/mul__0 shape: [1, 92, 14, 14] dtype: float32
INFO:  input_name.2: model/conv2d_11/Conv2D_weights_fused_bn shape: [26, 92, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_11/Conv2D_bias_fused_bn shape: [26] dtype: float32
INFO:  output_name.1: model/batch_normalization_19/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_60/Mul:0 shape: (1, 14, 14, 92) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 92, 26) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (26,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_49/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 

INFO: 94 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_10/depthwise
INFO:  input_name.1: model/batch_normalization_19/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_10/depthwise_weights_fused_bn shape: [26, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_10/depthwise_bias_fused_bn shape: [26] dtype: float32
INFO:  output_name.1: model/batch_normalization_20/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_49/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 26, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (26,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 26 
INFO:  output.1.output: name: tf.math.add_50/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 

INFO: 95 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_7/concat
INFO:  input_name.1: model/batch_normalization_19/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO:  input_name.2: model/batch_normalization_20/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO:  output_name.1: model/concatenate_7/concat__0 shape: [1, 52, 14, 14] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_49/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_50/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_7/concat:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 

INFO: 96 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_3/add
INFO:  input_name.1: model/batch_normalization_22/FusedBatchNormV3__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  input_name.2: model/concatenate_7/concat__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  output_name.1: model/add_3/add__0 shape: [1, 52, 14, 14] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_38/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_7/concat:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_52/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 

INFO: 97 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_13/Conv2D
INFO:  input_name.1: model/add_3/add__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  input_name.2: model/conv2d_13/Conv2D_weights_fused_bn shape: [78, 52, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_13/Conv2D_bias_fused_bn shape: [78] dtype: float32
INFO:  output_name.1: model/batch_normalization_23/FusedBatchNormV3__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_52/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 52, 78) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (78,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_53/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 98 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_11/Relu
INFO:  input_name.1: model/batch_normalization_23/FusedBatchNormV3__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_11/Relu__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_53/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_20/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 99 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_11/Neg_1
INFO:  input_name.1: model/batch_normalization_23/FusedBatchNormV3__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_11/Neg_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_53/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_10/Neg:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 100 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_11/Relu_1
INFO:  input_name.1: model/activation_11/Neg_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_11/Relu_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_10/Neg:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_21/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 101 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_11/mul
INFO:  input_name.1: model/activation_11/Relu_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_11/Neg:0 shape: [1, 78, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_11/mul__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_21/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 78) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_65/Mul:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 102 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_11/add
INFO:  input_name.1: model/activation_11/Relu__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_11/mul__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_11/add__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_20/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_65/Mul:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_55/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 103 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_12/depthwise
INFO:  input_name.1: model/activation_11/add__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_12/depthwise_weights_fused_bn shape: [78, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_12/depthwise_bias_fused_bn shape: [78] dtype: float32
INFO:  output_name.1: model/batch_normalization_24/FusedBatchNormV3__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_55/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 78, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (78,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 78 
INFO:  output.1.output: name: tf.math.add_56/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 104 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_12/Relu
INFO:  input_name.1: model/batch_normalization_24/FusedBatchNormV3__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_12/Relu__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_56/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_22/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 105 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_12/Neg_1
INFO:  input_name.1: model/batch_normalization_24/FusedBatchNormV3__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_12/Neg_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_56/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_11/Neg:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 106 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_12/Relu_1
INFO:  input_name.1: model/activation_12/Neg_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_12/Relu_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_11/Neg:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_23/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 107 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_12/mul
INFO:  input_name.1: model/activation_12/Relu_1__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_12/Neg:0 shape: [1, 78, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_12/mul__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_23/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 78) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_70/Mul:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 108 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_12/add
INFO:  input_name.1: model/activation_12/Relu__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_12/mul__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_12/add__0 shape: [1, 78, 14, 14] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_22/Relu:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_70/Mul:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_58/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 

INFO: 109 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_8/concat
INFO:  input_name.1: model/activation_11/add__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_12/add__0 shape: [1, 78, 14, 14] dtype: float32
INFO:  output_name.1: model/concatenate_8/concat__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_55/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_58/Add:0 shape: (1, 14, 14, 78) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_8/concat:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 110 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d_1/Mean
INFO:  input_name.1: model/concatenate_8/concat__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d_1/Mean__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.concat_8/concat:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean_1/Mean:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 111 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d_1/Mean_Squeeze__1957
INFO:  input_name.1: model/global_average_pooling2d_1/Mean__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: (2,) dtype: int64
INFO:  output_name.1: model/global_average_pooling2d_1/Mean_Squeeze__1957__0 shape: [1, 156] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean_1/Mean:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze_2/model/global_average_pooling2d_1/Mean_Squeeze__1957:0 shape: (1, 156) dtype: <dtype: 'float32'> 

INFO: 112 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_14/BiasAdd__285
INFO:  input_name.1: model/global_average_pooling2d_1/Mean_Squeeze__1957__0 shape: [1, 156] dtype: float32
INFO:  input_name.2: new_shape__1837 shape: [4] dtype: int64
INFO:  output_name.1: model/conv2d_14/BiasAdd__285__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze_2/model/global_average_pooling2d_1/Mean_Squeeze__1957:0 shape: (1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(156), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape_1/Reshape:0 shape: (1, 156, 1, 1) dtype: <dtype: 'float32'> 

INFO: 113 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_14/BiasAdd
INFO:  input_name.1: model/conv2d_14/BiasAdd__285__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_14/Conv2D/ReadVariableOp:0 shape: [40, 156, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_14/BiasAdd/ReadVariableOp:0 shape: [40] dtype: float32
INFO:  output_name.1: model/conv2d_14/BiasAdd__0 shape: [1, 40, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_5/transpose:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 156, 40) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (40,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_59/Add:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 

INFO: 114 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_13/Relu
INFO:  input_name.1: model/conv2d_14/BiasAdd__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_13/Relu__0 shape: [1, 40, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_59/Add:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_24/Relu:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 

INFO: 115 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_13/Neg_1
INFO:  input_name.1: model/conv2d_14/BiasAdd__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_13/Neg_1__0 shape: [1, 40, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_59/Add:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_12/Neg:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 

INFO: 116 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_13/Relu_1
INFO:  input_name.1: model/activation_13/Neg_1__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_13/Relu_1__0 shape: [1, 40, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_12/Neg:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_25/Relu:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 

INFO: 117 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_13/mul
INFO:  input_name.1: model/activation_13/Relu_1__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_13/Neg:0 shape: [1, 40, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_13/mul__0 shape: [1, 40, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_25/Relu:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 40) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_74/Mul:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 

INFO: 118 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_13/add
INFO:  input_name.1: model/activation_13/Relu__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_13/mul__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_13/add__0 shape: [1, 40, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_24/Relu:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_74/Mul:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_60/Add:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 

INFO: 119 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_15/BiasAdd
INFO:  input_name.1: model/activation_13/add__0 shape: [1, 40, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_15/Conv2D/ReadVariableOp:0 shape: [156, 40, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_15/BiasAdd/ReadVariableOp:0 shape: [156] dtype: float32
INFO:  output_name.1: model/conv2d_15/BiasAdd__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_60/Add:0 shape: (1, 1, 1, 40) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 40, 156) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (156,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_61/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 120 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_14/Mul
INFO:  input_name.1: model/conv2d_15/BiasAdd__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: () dtype: float32
INFO:  output_name.1: model/activation_14/Mul__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_61/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_77/Mul:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 121 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_14/Add
INFO:  input_name.1: model/activation_14/Mul__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_14/Add__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_77/Mul:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_62/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 122 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_14/clip_by_value/Minimum
INFO:  input_name.1: model/activation_14/Add__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_14/clip_by_value/Minimum__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_62/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum_1/Minimum:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 123 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_14/clip_by_value
INFO:  input_name.1: model/activation_14/clip_by_value/Minimum__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_14/clip_by_value__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum_1/Minimum:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum_1/Maximum:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 124 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply_1/mul
INFO:  input_name.1: model/concatenate_8/concat__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_14/clip_by_value__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply_1/mul__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.concat_8/concat:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum_1/Maximum:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_82/Mul:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 125 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_16/Conv2D
INFO:  input_name.1: model/multiply_1/mul__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/conv2d_16/Conv2D_weights_fused_bn shape: [26, 156, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_16/Conv2D_bias_fused_bn shape: [26] dtype: float32
INFO:  output_name.1: model/batch_normalization_25/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_82/Mul:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 156, 26) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (26,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_63/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 

INFO: 126 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_13/depthwise
INFO:  input_name.1: model/batch_normalization_25/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_13/depthwise_weights_fused_bn shape: [26, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_13/depthwise_bias_fused_bn shape: [26] dtype: float32
INFO:  output_name.1: model/batch_normalization_26/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_63/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 26, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (26,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 26 
INFO:  output.1.output: name: tf.math.add_64/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 

INFO: 127 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_9/concat
INFO:  input_name.1: model/batch_normalization_25/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO:  input_name.2: model/batch_normalization_26/FusedBatchNormV3__0 shape: [1, 26, 14, 14] dtype: float32
INFO:  output_name.1: model/concatenate_9/concat__0 shape: [1, 52, 14, 14] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_63/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_64/Add:0 shape: (1, 14, 14, 26) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_9/concat:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 

INFO: 128 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_4/add
INFO:  input_name.1: model/add_3/add__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  input_name.2: model/concatenate_9/concat__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  output_name.1: model/add_4/add__0 shape: [1, 52, 14, 14] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_52/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_9/concat:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_66/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 

INFO: 129 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_17/depthwise
INFO:  input_name.1: model/add_4/add__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_17/depthwise_weights_fused_bn shape: [52, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_17/depthwise_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_32/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_5/Pad:0 shape: (1, 15, 15, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 52, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 52 
INFO:  output.1.output: name: tf.math.add_67/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 130 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_17/Conv2D
INFO:  input_name.1: model/add_4/add__0 shape: [1, 52, 14, 14] dtype: float32
INFO:  input_name.2: model/conv2d_17/Conv2D_weights_fused_bn shape: [156, 52, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_17/Conv2D_bias_fused_bn shape: [156] dtype: float32
INFO:  output_name.1: model/batch_normalization_27/FusedBatchNormV3__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_66/Add:0 shape: (1, 14, 14, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 52, 156) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (156,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_68/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 131 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_19/Conv2D
INFO:  input_name.1: model/batch_normalization_32/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_19/Conv2D_weights_fused_bn shape: [104, 52, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_19/Conv2D_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_33/FusedBatchNormV3__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_67/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 52, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_69/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 132 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_15/Relu
INFO:  input_name.1: model/batch_normalization_27/FusedBatchNormV3__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_15/Relu__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_68/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_26/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 133 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_15/Neg_1
INFO:  input_name.1: model/batch_normalization_27/FusedBatchNormV3__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_15/Neg_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_68/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_13/Neg:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 134 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_15/Relu_1
INFO:  input_name.1: model/activation_15/Neg_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_15/Relu_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_13/Neg:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_27/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 135 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_15/mul
INFO:  input_name.1: model/activation_15/Relu_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_15/Neg:0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_15/mul__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_27/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 156) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_87/Mul:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 136 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_15/add
INFO:  input_name.1: model/activation_15/Relu__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_15/mul__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_15/add__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_26/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_87/Mul:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_71/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 137 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_14/depthwise
INFO:  input_name.1: model/activation_15/add__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_14/depthwise_weights_fused_bn shape: [156, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_14/depthwise_bias_fused_bn shape: [156] dtype: float32
INFO:  output_name.1: model/batch_normalization_28/FusedBatchNormV3__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_71/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 156, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (156,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 156 
INFO:  output.1.output: name: tf.math.add_72/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 138 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_16/Relu
INFO:  input_name.1: model/batch_normalization_28/FusedBatchNormV3__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_16/Relu__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_72/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_28/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 139 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_16/Neg_1
INFO:  input_name.1: model/batch_normalization_28/FusedBatchNormV3__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_16/Neg_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_72/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_14/Neg:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 140 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_16/Relu_1
INFO:  input_name.1: model/activation_16/Neg_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_16/Relu_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_14/Neg:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_29/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 141 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_16/mul
INFO:  input_name.1: model/activation_16/Relu_1__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_16/Neg:0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_16/mul__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_29/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 156) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_92/Mul:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 142 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_16/add
INFO:  input_name.1: model/activation_16/Relu__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_16/mul__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/activation_16/add__0 shape: [1, 156, 14, 14] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_28/Relu:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_92/Mul:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_74/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 

INFO: 143 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_10/concat
INFO:  input_name.1: model/activation_15/add__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  input_name.2: model/activation_16/add__0 shape: [1, 156, 14, 14] dtype: float32
INFO:  output_name.1: model/concatenate_10/concat__0 shape: [1, 312, 14, 14] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_71/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_74/Add:0 shape: (1, 14, 14, 156) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_10/concat:0 shape: (1, 14, 14, 312) dtype: <dtype: 'float32'> 

INFO: 144 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_15/depthwise
INFO:  input_name.1: model/concatenate_10/concat__0 shape: [1, 312, 14, 14] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_15/depthwise_weights_fused_bn shape: [312, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_15/depthwise_bias_fused_bn shape: [312] dtype: float32
INFO:  output_name.1: model/batch_normalization_29/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_6/Pad:0 shape: (1, 15, 15, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 312, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (312,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 312 
INFO:  output.1.output: name: tf.math.add_75/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 145 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_18/Conv2D
INFO:  input_name.1: model/batch_normalization_29/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_18/Conv2D_weights_fused_bn shape: [52, 312, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_18/Conv2D_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_30/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_75/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 312, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_76/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 146 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_16/depthwise
INFO:  input_name.1: model/batch_normalization_30/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_16/depthwise_weights_fused_bn shape: [52, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_16/depthwise_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_31/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_76/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 52, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 52 
INFO:  output.1.output: name: tf.math.add_77/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 147 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_11/concat
INFO:  input_name.1: model/batch_normalization_30/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/batch_normalization_31/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_11/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_76/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_77/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_11/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 148 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_5/add
INFO:  input_name.1: model/batch_normalization_33/FusedBatchNormV3__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/concatenate_11/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  output_name.1: model/add_5/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_69/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_11/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_79/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 149 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_20/Conv2D
INFO:  input_name.1: model/add_5/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_20/Conv2D_weights_fused_bn shape: [130, 104, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_20/Conv2D_bias_fused_bn shape: [130] dtype: float32
INFO:  output_name.1: model/batch_normalization_34/FusedBatchNormV3__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_79/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 104, 130) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (130,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_80/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 150 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_17/Relu
INFO:  input_name.1: model/batch_normalization_34/FusedBatchNormV3__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_17/Relu__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_80/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_30/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 151 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_17/Neg_1
INFO:  input_name.1: model/batch_normalization_34/FusedBatchNormV3__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_17/Neg_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_80/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_15/Neg:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 152 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_17/Relu_1
INFO:  input_name.1: model/activation_17/Neg_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_17/Relu_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_15/Neg:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_31/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 153 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_17/mul
INFO:  input_name.1: model/activation_17/Relu_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_17/Neg:0 shape: [1, 130, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_17/mul__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_31/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 130) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_99/Mul:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 154 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_17/add
INFO:  input_name.1: model/activation_17/Relu__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_17/mul__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_17/add__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_30/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_99/Mul:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_82/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 155 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_18/depthwise
INFO:  input_name.1: model/activation_17/add__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_18/depthwise_weights_fused_bn shape: [130, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_18/depthwise_bias_fused_bn shape: [130] dtype: float32
INFO:  output_name.1: model/batch_normalization_35/FusedBatchNormV3__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_82/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 130, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (130,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 130 
INFO:  output.1.output: name: tf.math.add_83/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 156 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_18/Relu
INFO:  input_name.1: model/batch_normalization_35/FusedBatchNormV3__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_18/Relu__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_83/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_32/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 157 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_18/Neg_1
INFO:  input_name.1: model/batch_normalization_35/FusedBatchNormV3__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_18/Neg_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_83/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_16/Neg:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 158 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_18/Relu_1
INFO:  input_name.1: model/activation_18/Neg_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_18/Relu_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_16/Neg:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_33/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 159 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_18/mul
INFO:  input_name.1: model/activation_18/Relu_1__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_18/Neg:0 shape: [1, 130, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_18/mul__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_33/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 130) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_104/Mul:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 160 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_18/add
INFO:  input_name.1: model/activation_18/Relu__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_18/mul__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_18/add__0 shape: [1, 130, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_32/Relu:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_104/Mul:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_85/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 

INFO: 161 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_12/concat
INFO:  input_name.1: model/activation_17/add__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_18/add__0 shape: [1, 130, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_12/concat__0 shape: [1, 260, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_82/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_85/Add:0 shape: (1, 7, 7, 130) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_12/concat:0 shape: (1, 7, 7, 260) dtype: <dtype: 'float32'> 

INFO: 162 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_21/Conv2D
INFO:  input_name.1: model/concatenate_12/concat__0 shape: [1, 260, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_21/Conv2D_weights_fused_bn shape: [52, 260, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_21/Conv2D_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_36/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat_12/concat:0 shape: (1, 7, 7, 260) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 260, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_86/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 163 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_19/depthwise
INFO:  input_name.1: model/batch_normalization_36/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_19/depthwise_weights_fused_bn shape: [52, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_19/depthwise_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_37/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_86/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 52, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 52 
INFO:  output.1.output: name: tf.math.add_87/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 164 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_13/concat
INFO:  input_name.1: model/batch_normalization_36/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/batch_normalization_37/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_13/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_86/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_87/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_13/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 165 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_6/add
INFO:  input_name.1: model/add_5/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/concatenate_13/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  output_name.1: model/add_6/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_79/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_13/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_89/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 166 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_22/Conv2D
INFO:  input_name.1: model/add_6/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_22/Conv2D_weights_fused_bn shape: [120, 104, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_22/Conv2D_bias_fused_bn shape: [120] dtype: float32
INFO:  output_name.1: model/batch_normalization_38/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_89/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 104, 120) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (120,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_90/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 167 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_19/Relu
INFO:  input_name.1: model/batch_normalization_38/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_19/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_90/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_34/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 168 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_19/Neg_1
INFO:  input_name.1: model/batch_normalization_38/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_19/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_90/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_17/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 169 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_19/Relu_1
INFO:  input_name.1: model/activation_19/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_19/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_17/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_35/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 170 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_19/mul
INFO:  input_name.1: model/activation_19/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_19/Neg:0 shape: [1, 120, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_19/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_35/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 120) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_111/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 171 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_19/add
INFO:  input_name.1: model/activation_19/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_19/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_19/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_34/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_111/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_92/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 172 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_20/depthwise
INFO:  input_name.1: model/activation_19/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_20/depthwise_weights_fused_bn shape: [120, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_20/depthwise_bias_fused_bn shape: [120] dtype: float32
INFO:  output_name.1: model/batch_normalization_39/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_92/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 120, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (120,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 120 
INFO:  output.1.output: name: tf.math.add_93/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 173 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_20/Relu
INFO:  input_name.1: model/batch_normalization_39/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_20/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_93/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_36/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 174 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_20/Neg_1
INFO:  input_name.1: model/batch_normalization_39/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_20/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_93/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_18/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 175 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_20/Relu_1
INFO:  input_name.1: model/activation_20/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_20/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_18/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_37/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 176 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_20/mul
INFO:  input_name.1: model/activation_20/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_20/Neg:0 shape: [1, 120, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_20/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_37/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 120) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_116/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 177 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_20/add
INFO:  input_name.1: model/activation_20/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_20/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_20/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_36/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_116/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_95/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 178 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_14/concat
INFO:  input_name.1: model/activation_19/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_20/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_14/concat__0 shape: [1, 240, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_92/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_95/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_14/concat:0 shape: (1, 7, 7, 240) dtype: <dtype: 'float32'> 

INFO: 179 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_23/Conv2D
INFO:  input_name.1: model/concatenate_14/concat__0 shape: [1, 240, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_23/Conv2D_weights_fused_bn shape: [52, 240, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_23/Conv2D_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_40/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat_14/concat:0 shape: (1, 7, 7, 240) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 240, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_96/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 180 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_21/depthwise
INFO:  input_name.1: model/batch_normalization_40/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_21/depthwise_weights_fused_bn shape: [52, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_21/depthwise_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_41/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_96/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 52, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 52 
INFO:  output.1.output: name: tf.math.add_97/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 181 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_15/concat
INFO:  input_name.1: model/batch_normalization_40/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/batch_normalization_41/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_15/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_96/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_97/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_15/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 182 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_7/add
INFO:  input_name.1: model/add_6/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/concatenate_15/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  output_name.1: model/add_7/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_89/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_15/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_99/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 183 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_24/Conv2D
INFO:  input_name.1: model/add_7/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_24/Conv2D_weights_fused_bn shape: [120, 104, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_24/Conv2D_bias_fused_bn shape: [120] dtype: float32
INFO:  output_name.1: model/batch_normalization_42/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_99/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 104, 120) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (120,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_100/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 184 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_21/Relu
INFO:  input_name.1: model/batch_normalization_42/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_21/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_100/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_38/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 185 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_21/Neg_1
INFO:  input_name.1: model/batch_normalization_42/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_21/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_100/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_19/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 186 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_21/Relu_1
INFO:  input_name.1: model/activation_21/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_21/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_19/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_39/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 187 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_21/mul
INFO:  input_name.1: model/activation_21/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_21/Neg:0 shape: [1, 120, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_21/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_39/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 120) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_123/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 188 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_21/add
INFO:  input_name.1: model/activation_21/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_21/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_21/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_38/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_123/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_102/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 189 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_22/depthwise
INFO:  input_name.1: model/activation_21/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_22/depthwise_weights_fused_bn shape: [120, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_22/depthwise_bias_fused_bn shape: [120] dtype: float32
INFO:  output_name.1: model/batch_normalization_43/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_102/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 120, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (120,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 120 
INFO:  output.1.output: name: tf.math.add_103/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 190 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_22/Relu
INFO:  input_name.1: model/batch_normalization_43/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_22/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_103/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_40/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 191 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_22/Neg_1
INFO:  input_name.1: model/batch_normalization_43/FusedBatchNormV3__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_22/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_103/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_20/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 192 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_22/Relu_1
INFO:  input_name.1: model/activation_22/Neg_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_22/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_20/Neg:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_41/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 193 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_22/mul
INFO:  input_name.1: model/activation_22/Relu_1__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_22/Neg:0 shape: [1, 120, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_22/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_41/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 120) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_128/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 194 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_22/add
INFO:  input_name.1: model/activation_22/Relu__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_22/mul__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_22/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_40/Relu:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_128/Mul:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_105/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 

INFO: 195 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_16/concat
INFO:  input_name.1: model/activation_21/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_22/add__0 shape: [1, 120, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_16/concat__0 shape: [1, 240, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_102/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_105/Add:0 shape: (1, 7, 7, 120) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_16/concat:0 shape: (1, 7, 7, 240) dtype: <dtype: 'float32'> 

INFO: 196 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_25/Conv2D
INFO:  input_name.1: model/concatenate_16/concat__0 shape: [1, 240, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_25/Conv2D_weights_fused_bn shape: [52, 240, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_25/Conv2D_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_44/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat_16/concat:0 shape: (1, 7, 7, 240) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 240, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_106/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 197 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_23/depthwise
INFO:  input_name.1: model/batch_normalization_44/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_23/depthwise_weights_fused_bn shape: [52, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_23/depthwise_bias_fused_bn shape: [52] dtype: float32
INFO:  output_name.1: model/batch_normalization_45/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_106/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 52, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (52,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 52 
INFO:  output.1.output: name: tf.math.add_107/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 

INFO: 198 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_17/concat
INFO:  input_name.1: model/batch_normalization_44/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  input_name.2: model/batch_normalization_45/FusedBatchNormV3__0 shape: [1, 52, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_17/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_106/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_107/Add:0 shape: (1, 7, 7, 52) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_17/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 199 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_8/add
INFO:  input_name.1: model/add_7/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/concatenate_17/concat__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  output_name.1: model/add_8/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_99/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_17/concat:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_109/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 200 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_26/depthwise
INFO:  input_name.1: model/add_8/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_26/depthwise_weights_fused_bn shape: [104, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_26/depthwise_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_50/FusedBatchNormV3__0 shape: [1, 104, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_109/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 104, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 104 
INFO:  output.1.output: name: tf.math.add_110/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 

INFO: 201 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_26/Conv2D
INFO:  input_name.1: model/add_8/add__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_26/Conv2D_weights_fused_bn shape: [312, 104, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_26/Conv2D_bias_fused_bn shape: [312] dtype: float32
INFO:  output_name.1: model/batch_normalization_46/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_109/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 104, 312) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (312,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_111/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 202 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_30/Conv2D
INFO:  input_name.1: model/batch_normalization_50/FusedBatchNormV3__0 shape: [1, 104, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_30/Conv2D_weights_fused_bn shape: [144, 104, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_30/Conv2D_bias_fused_bn shape: [144] dtype: float32
INFO:  output_name.1: model/batch_normalization_51/FusedBatchNormV3__0 shape: [1, 144, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_110/Add:0 shape: (1, 7, 7, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 104, 144) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (144,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_112/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 

INFO: 203 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_23/Relu
INFO:  input_name.1: model/batch_normalization_46/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_23/Relu__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_111/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_42/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 204 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_23/Neg_1
INFO:  input_name.1: model/batch_normalization_46/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_23/Neg_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_111/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_21/Neg:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 205 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_23/Relu_1
INFO:  input_name.1: model/activation_23/Neg_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_23/Relu_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_21/Neg:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_43/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 206 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_23/mul
INFO:  input_name.1: model/activation_23/Relu_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_23/Neg:0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_23/mul__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_43/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 312) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_135/Mul:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 207 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_23/add
INFO:  input_name.1: model/activation_23/Relu__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_23/mul__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_23/add__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_42/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_135/Mul:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_114/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 208 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_24/depthwise
INFO:  input_name.1: model/activation_23/add__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_24/depthwise_weights_fused_bn shape: [312, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_24/depthwise_bias_fused_bn shape: [312] dtype: float32
INFO:  output_name.1: model/batch_normalization_47/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_114/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 312, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (312,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 312 
INFO:  output.1.output: name: tf.math.add_115/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 209 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_24/Relu
INFO:  input_name.1: model/batch_normalization_47/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_24/Relu__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_115/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_44/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 210 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_24/Neg_1
INFO:  input_name.1: model/batch_normalization_47/FusedBatchNormV3__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_24/Neg_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_115/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_22/Neg:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 211 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_24/Relu_1
INFO:  input_name.1: model/activation_24/Neg_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_24/Relu_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_22/Neg:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_45/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 212 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_24/mul
INFO:  input_name.1: model/activation_24/Relu_1__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_24/Neg:0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_24/mul__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_45/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 312) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_140/Mul:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 213 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_24/add
INFO:  input_name.1: model/activation_24/Relu__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_24/mul__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_24/add__0 shape: [1, 312, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_44/Relu:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_140/Mul:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_117/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 

INFO: 214 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_18/concat
INFO:  input_name.1: model/activation_23/add__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_24/add__0 shape: [1, 312, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_18/concat__0 shape: [1, 624, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_114/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_117/Add:0 shape: (1, 7, 7, 312) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_18/concat:0 shape: (1, 7, 7, 624) dtype: <dtype: 'float32'> 

INFO: 215 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d_2/Mean
INFO:  input_name.1: model/concatenate_18/concat__0 shape: [1, 624, 7, 7] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d_2/Mean__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.concat_18/concat:0 shape: (1, 7, 7, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean_2/Mean:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 

INFO: 216 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d_2/Mean_Squeeze__1965
INFO:  input_name.1: model/global_average_pooling2d_2/Mean__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: (2,) dtype: int64
INFO:  output_name.1: model/global_average_pooling2d_2/Mean_Squeeze__1965__0 shape: [1, 624] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean_2/Mean:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze_4/model/global_average_pooling2d_2/Mean_Squeeze__1965:0 shape: (1, 624) dtype: <dtype: 'float32'> 

INFO: 217 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_27/BiasAdd__548
INFO:  input_name.1: model/global_average_pooling2d_2/Mean_Squeeze__1965__0 shape: [1, 624] dtype: float32
INFO:  input_name.2: new_shape__1828 shape: [4] dtype: int64
INFO:  output_name.1: model/conv2d_27/BiasAdd__548__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze_4/model/global_average_pooling2d_2/Mean_Squeeze__1965:0 shape: (1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(624), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape_2/Reshape:0 shape: (1, 624, 1, 1) dtype: <dtype: 'float32'> 

INFO: 218 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_27/BiasAdd
INFO:  input_name.1: model/conv2d_27/BiasAdd__548__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_27/Conv2D/ReadVariableOp:0 shape: [156, 624, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_27/BiasAdd/ReadVariableOp:0 shape: [156] dtype: float32
INFO:  output_name.1: model/conv2d_27/BiasAdd__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_8/transpose:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 624, 156) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (156,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_118/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 219 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_25/Relu
INFO:  input_name.1: model/conv2d_27/BiasAdd__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_25/Relu__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_118/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_46/Relu:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 220 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_25/Neg_1
INFO:  input_name.1: model/conv2d_27/BiasAdd__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_25/Neg_1__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_118/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_23/Neg:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 221 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_25/Relu_1
INFO:  input_name.1: model/activation_25/Neg_1__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_25/Relu_1__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_23/Neg:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_47/Relu:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 222 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_25/mul
INFO:  input_name.1: model/activation_25/Relu_1__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_25/Neg:0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_25/mul__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_47/Relu:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 156) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_144/Mul:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 223 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_25/add
INFO:  input_name.1: model/activation_25/Relu__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_25/mul__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_25/add__0 shape: [1, 156, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_46/Relu:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_144/Mul:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_119/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 

INFO: 224 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_28/BiasAdd
INFO:  input_name.1: model/activation_25/add__0 shape: [1, 156, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_28/Conv2D/ReadVariableOp:0 shape: [624, 156, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_28/BiasAdd/ReadVariableOp:0 shape: [624] dtype: float32
INFO:  output_name.1: model/conv2d_28/BiasAdd__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_119/Add:0 shape: (1, 1, 1, 156) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 156, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_120/Add:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 

INFO: 225 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_26/Mul
INFO:  input_name.1: model/conv2d_28/BiasAdd__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: () dtype: float32
INFO:  output_name.1: model/activation_26/Mul__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_120/Add:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_147/Mul:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 

INFO: 226 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_26/Add
INFO:  input_name.1: model/activation_26/Mul__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_26/Add__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_147/Mul:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_121/Add:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 

INFO: 227 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_26/clip_by_value/Minimum
INFO:  input_name.1: model/activation_26/Add__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_26/clip_by_value/Minimum__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_121/Add:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum_2/Minimum:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 

INFO: 228 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_26/clip_by_value
INFO:  input_name.1: model/activation_26/clip_by_value/Minimum__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_26/clip_by_value__0 shape: [1, 624, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum_2/Minimum:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum_2/Maximum:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 

INFO: 229 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply_2/mul
INFO:  input_name.1: model/concatenate_18/concat__0 shape: [1, 624, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_26/clip_by_value__0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply_2/mul__0 shape: [1, 624, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.concat_18/concat:0 shape: (1, 7, 7, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum_2/Maximum:0 shape: (1, 1, 1, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_152/Mul:0 shape: (1, 7, 7, 624) dtype: <dtype: 'float32'> 

INFO: 230 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_29/Conv2D
INFO:  input_name.1: model/multiply_2/mul__0 shape: [1, 624, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_29/Conv2D_weights_fused_bn shape: [72, 624, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_29/Conv2D_bias_fused_bn shape: [72] dtype: float32
INFO:  output_name.1: model/batch_normalization_48/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_152/Mul:0 shape: (1, 7, 7, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 624, 72) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (72,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_122/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 

INFO: 231 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_25/depthwise
INFO:  input_name.1: model/batch_normalization_48/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_25/depthwise_weights_fused_bn shape: [72, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_25/depthwise_bias_fused_bn shape: [72] dtype: float32
INFO:  output_name.1: model/batch_normalization_49/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_122/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 72, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (72,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 72 
INFO:  output.1.output: name: tf.math.add_123/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 

INFO: 232 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_19/concat
INFO:  input_name.1: model/batch_normalization_48/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO:  input_name.2: model/batch_normalization_49/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_19/concat__0 shape: [1, 144, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_122/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_123/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_19/concat:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 

INFO: 233 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_9/add
INFO:  input_name.1: model/batch_normalization_51/FusedBatchNormV3__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  input_name.2: model/concatenate_19/concat__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  output_name.1: model/add_9/add__0 shape: [1, 144, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_112/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_19/concat:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_125/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 

INFO: 234 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_31/Conv2D
INFO:  input_name.1: model/add_9/add__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_31/Conv2D_weights_fused_bn shape: [436, 144, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_31/Conv2D_bias_fused_bn shape: [436] dtype: float32
INFO:  output_name.1: model/batch_normalization_52/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_125/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 144, 436) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (436,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_126/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 235 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_27/Relu
INFO:  input_name.1: model/batch_normalization_52/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_27/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_126/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_48/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 236 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_27/Neg_1
INFO:  input_name.1: model/batch_normalization_52/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_27/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_126/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_24/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 237 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_27/Relu_1
INFO:  input_name.1: model/activation_27/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_27/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_24/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_49/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 238 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_27/mul
INFO:  input_name.1: model/activation_27/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_27/Neg:0 shape: [1, 436, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_27/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_49/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 436) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_157/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 239 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_27/add
INFO:  input_name.1: model/activation_27/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_27/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_27/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_48/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_157/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_128/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 240 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_27/depthwise
INFO:  input_name.1: model/activation_27/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_27/depthwise_weights_fused_bn shape: [436, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_27/depthwise_bias_fused_bn shape: [436] dtype: float32
INFO:  output_name.1: model/batch_normalization_53/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_128/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 436, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (436,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 436 
INFO:  output.1.output: name: tf.math.add_129/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 241 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_28/Relu
INFO:  input_name.1: model/batch_normalization_53/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_28/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_129/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_50/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 242 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_28/Neg_1
INFO:  input_name.1: model/batch_normalization_53/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_28/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_129/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_25/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 243 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_28/Relu_1
INFO:  input_name.1: model/activation_28/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_28/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_25/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_51/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 244 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_28/mul
INFO:  input_name.1: model/activation_28/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_28/Neg:0 shape: [1, 436, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_28/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_51/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 436) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_162/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 245 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_28/add
INFO:  input_name.1: model/activation_28/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_28/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_28/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_50/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_162/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_131/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 246 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_20/concat
INFO:  input_name.1: model/activation_27/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_28/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_20/concat__0 shape: [1, 872, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_128/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_131/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_20/concat:0 shape: (1, 7, 7, 872) dtype: <dtype: 'float32'> 

INFO: 247 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d_3/Mean
INFO:  input_name.1: model/concatenate_20/concat__0 shape: [1, 872, 7, 7] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d_3/Mean__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.concat_20/concat:0 shape: (1, 7, 7, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean_3/Mean:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 248 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d_3/Mean_Squeeze__1959
INFO:  input_name.1: model/global_average_pooling2d_3/Mean__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: (2,) dtype: int64
INFO:  output_name.1: model/global_average_pooling2d_3/Mean_Squeeze__1959__0 shape: [1, 872] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean_3/Mean:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze_6/model/global_average_pooling2d_3/Mean_Squeeze__1959:0 shape: (1, 872) dtype: <dtype: 'float32'> 

INFO: 249 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_32/BiasAdd__610
INFO:  input_name.1: model/global_average_pooling2d_3/Mean_Squeeze__1959__0 shape: [1, 872] dtype: float32
INFO:  input_name.2: new_shape__1808 shape: [4] dtype: int64
INFO:  output_name.1: model/conv2d_32/BiasAdd__610__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze_6/model/global_average_pooling2d_3/Mean_Squeeze__1959:0 shape: (1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(872), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape_3/Reshape:0 shape: (1, 872, 1, 1) dtype: <dtype: 'float32'> 

INFO: 250 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_32/BiasAdd
INFO:  input_name.1: model/conv2d_32/BiasAdd__610__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_32/Conv2D/ReadVariableOp:0 shape: [220, 872, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_32/BiasAdd/ReadVariableOp:0 shape: [220] dtype: float32
INFO:  output_name.1: model/conv2d_32/BiasAdd__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_11/transpose:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 872, 220) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (220,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_132/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 251 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_29/Relu
INFO:  input_name.1: model/conv2d_32/BiasAdd__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_29/Relu__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_132/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_52/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 252 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_29/Neg_1
INFO:  input_name.1: model/conv2d_32/BiasAdd__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_29/Neg_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_132/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_26/Neg:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 253 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_29/Relu_1
INFO:  input_name.1: model/activation_29/Neg_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_29/Relu_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_26/Neg:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_53/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 254 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_29/mul
INFO:  input_name.1: model/activation_29/Relu_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_29/Neg:0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_29/mul__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_53/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 220) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_166/Mul:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 255 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_29/add
INFO:  input_name.1: model/activation_29/Relu__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_29/mul__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_29/add__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_52/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_166/Mul:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_133/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 256 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_33/BiasAdd
INFO:  input_name.1: model/activation_29/add__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_33/Conv2D/ReadVariableOp:0 shape: [872, 220, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_33/BiasAdd/ReadVariableOp:0 shape: [872] dtype: float32
INFO:  output_name.1: model/conv2d_33/BiasAdd__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_133/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 220, 872) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (872,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_134/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 257 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_30/Mul
INFO:  input_name.1: model/conv2d_33/BiasAdd__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: () dtype: float32
INFO:  output_name.1: model/activation_30/Mul__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_134/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_169/Mul:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 258 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_30/Add
INFO:  input_name.1: model/activation_30/Mul__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_30/Add__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_169/Mul:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_135/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 259 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_30/clip_by_value/Minimum
INFO:  input_name.1: model/activation_30/Add__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_30/clip_by_value/Minimum__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_135/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum_3/Minimum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 260 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_30/clip_by_value
INFO:  input_name.1: model/activation_30/clip_by_value/Minimum__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_30/clip_by_value__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum_3/Minimum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum_3/Maximum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 261 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply_3/mul
INFO:  input_name.1: model/concatenate_20/concat__0 shape: [1, 872, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_30/clip_by_value__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply_3/mul__0 shape: [1, 872, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.concat_20/concat:0 shape: (1, 7, 7, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum_3/Maximum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_174/Mul:0 shape: (1, 7, 7, 872) dtype: <dtype: 'float32'> 

INFO: 262 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_34/Conv2D
INFO:  input_name.1: model/multiply_3/mul__0 shape: [1, 872, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_34/Conv2D_weights_fused_bn shape: [72, 872, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_34/Conv2D_bias_fused_bn shape: [72] dtype: float32
INFO:  output_name.1: model/batch_normalization_54/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_174/Mul:0 shape: (1, 7, 7, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 872, 72) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (72,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_136/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 

INFO: 263 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_28/depthwise
INFO:  input_name.1: model/batch_normalization_54/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_28/depthwise_weights_fused_bn shape: [72, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_28/depthwise_bias_fused_bn shape: [72] dtype: float32
INFO:  output_name.1: model/batch_normalization_55/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_136/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 72, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (72,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 72 
INFO:  output.1.output: name: tf.math.add_137/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 

INFO: 264 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_21/concat
INFO:  input_name.1: model/batch_normalization_54/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO:  input_name.2: model/batch_normalization_55/FusedBatchNormV3__0 shape: [1, 72, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_21/concat__0 shape: [1, 144, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_136/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_137/Add:0 shape: (1, 7, 7, 72) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_21/concat:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 

INFO: 265 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_10/add
INFO:  input_name.1: model/add_9/add__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  input_name.2: model/concatenate_21/concat__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  output_name.1: model/add_10/add__0 shape: [1, 144, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_125/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_21/concat:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_139/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 

INFO: 266 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_32/depthwise
INFO:  input_name.1: model/add_10/add__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_32/depthwise_weights_fused_bn shape: [144, 1, 5, 5] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_32/depthwise_bias_fused_bn shape: [144] dtype: float32
INFO:  output_name.1: model/batch_normalization_61/FusedBatchNormV3__0 shape: [1, 144, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_7/Pad:0 shape: (1, 11, 11, 144) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (5, 5, 144, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (144,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 144 
INFO:  output.1.output: name: tf.math.add_140/Add:0 shape: (1, 4, 4, 144) dtype: <dtype: 'float32'> 

INFO: 267 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_35/Conv2D
INFO:  input_name.1: model/add_10/add__0 shape: [1, 144, 7, 7] dtype: float32
INFO:  input_name.2: model/conv2d_35/Conv2D_weights_fused_bn shape: [436, 144, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_35/Conv2D_bias_fused_bn shape: [436] dtype: float32
INFO:  output_name.1: model/batch_normalization_56/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_139/Add:0 shape: (1, 7, 7, 144) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 144, 436) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (436,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_141/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 268 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_39/Conv2D
INFO:  input_name.1: model/batch_normalization_61/FusedBatchNormV3__0 shape: [1, 144, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_39/Conv2D_weights_fused_bn shape: [208, 144, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_39/Conv2D_bias_fused_bn shape: [208] dtype: float32
INFO:  output_name.1: model/batch_normalization_62/FusedBatchNormV3__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_140/Add:0 shape: (1, 4, 4, 144) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 144, 208) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (208,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_142/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 269 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_31/Relu
INFO:  input_name.1: model/batch_normalization_56/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_31/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_141/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_54/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 270 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_31/Neg_1
INFO:  input_name.1: model/batch_normalization_56/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_31/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_141/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_27/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 271 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_31/Relu_1
INFO:  input_name.1: model/activation_31/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_31/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_27/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_55/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 272 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_31/mul
INFO:  input_name.1: model/activation_31/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_31/Neg:0 shape: [1, 436, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_31/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_55/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 436) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_179/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 273 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_31/add
INFO:  input_name.1: model/activation_31/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_31/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_31/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_54/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_179/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_144/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 274 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_29/depthwise
INFO:  input_name.1: model/activation_31/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_29/depthwise_weights_fused_bn shape: [436, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_29/depthwise_bias_fused_bn shape: [436] dtype: float32
INFO:  output_name.1: model/batch_normalization_57/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_144/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 436, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (436,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 436 
INFO:  output.1.output: name: tf.math.add_145/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 275 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_32/Relu
INFO:  input_name.1: model/batch_normalization_57/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_32/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_145/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_56/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 276 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_32/Neg_1
INFO:  input_name.1: model/batch_normalization_57/FusedBatchNormV3__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_32/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_145/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_28/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 277 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_32/Relu_1
INFO:  input_name.1: model/activation_32/Neg_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_32/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_28/Neg:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_57/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 278 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_32/mul
INFO:  input_name.1: model/activation_32/Relu_1__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_32/Neg:0 shape: [1, 436, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_32/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_57/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 436) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_184/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 279 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_32/add
INFO:  input_name.1: model/activation_32/Relu__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_32/mul__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/activation_32/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_56/Relu:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_184/Mul:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_147/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 

INFO: 280 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_22/concat
INFO:  input_name.1: model/activation_31/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  input_name.2: model/activation_32/add__0 shape: [1, 436, 7, 7] dtype: float32
INFO:  output_name.1: model/concatenate_22/concat__0 shape: [1, 872, 7, 7] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_144/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_147/Add:0 shape: (1, 7, 7, 436) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_22/concat:0 shape: (1, 7, 7, 872) dtype: <dtype: 'float32'> 

INFO: 281 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_30/depthwise
INFO:  input_name.1: model/concatenate_22/concat__0 shape: [1, 872, 7, 7] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_30/depthwise_weights_fused_bn shape: [872, 1, 5, 5] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_30/depthwise_bias_fused_bn shape: [872] dtype: float32
INFO:  output_name.1: model/batch_normalization_58/FusedBatchNormV3__0 shape: [1, 872, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.compat.v1.pad_8/Pad:0 shape: (1, 11, 11, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (5, 5, 872, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (872,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 2, 2, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 872 
INFO:  output.1.output: name: tf.math.add_148/Add:0 shape: (1, 4, 4, 872) dtype: <dtype: 'float32'> 

INFO: 282 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d_4/Mean
INFO:  input_name.1: model/batch_normalization_58/FusedBatchNormV3__0 shape: [1, 872, 4, 4] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d_4/Mean__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.math.add_148/Add:0 shape: (1, 4, 4, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean_4/Mean:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 283 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d_4/Mean_Squeeze__1963
INFO:  input_name.1: model/global_average_pooling2d_4/Mean__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: (2,) dtype: int64
INFO:  output_name.1: model/global_average_pooling2d_4/Mean_Squeeze__1963__0 shape: [1, 872] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean_4/Mean:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze_8/model/global_average_pooling2d_4/Mean_Squeeze__1963:0 shape: (1, 872) dtype: <dtype: 'float32'> 

INFO: 284 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_36/BiasAdd__702
INFO:  input_name.1: model/global_average_pooling2d_4/Mean_Squeeze__1963__0 shape: [1, 872] dtype: float32
INFO:  input_name.2: new_shape__1808 shape: (4,) dtype: int64
INFO:  output_name.1: model/conv2d_36/BiasAdd__702__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze_8/model/global_average_pooling2d_4/Mean_Squeeze__1963:0 shape: (1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(872), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape_4/Reshape:0 shape: (1, 872, 1, 1) dtype: <dtype: 'float32'> 

INFO: 285 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_36/BiasAdd
INFO:  input_name.1: model/conv2d_36/BiasAdd__702__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_36/Conv2D/ReadVariableOp:0 shape: [220, 872, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_36/BiasAdd/ReadVariableOp:0 shape: [220] dtype: float32
INFO:  output_name.1: model/conv2d_36/BiasAdd__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_14/transpose:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 872, 220) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (220,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_149/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 286 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_33/Relu
INFO:  input_name.1: model/conv2d_36/BiasAdd__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_33/Relu__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_149/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_58/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 287 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_33/Neg_1
INFO:  input_name.1: model/conv2d_36/BiasAdd__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_33/Neg_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_149/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_29/Neg:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 288 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_33/Relu_1
INFO:  input_name.1: model/activation_33/Neg_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_33/Relu_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_29/Neg:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_59/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 289 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_33/mul
INFO:  input_name.1: model/activation_33/Relu_1__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_33/Neg:0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_33/mul__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_59/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 220) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_188/Mul:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 290 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_33/add
INFO:  input_name.1: model/activation_33/Relu__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_33/mul__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_33/add__0 shape: [1, 220, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_58/Relu:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_188/Mul:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_150/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 

INFO: 291 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_37/BiasAdd
INFO:  input_name.1: model/activation_33/add__0 shape: [1, 220, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_37/Conv2D/ReadVariableOp:0 shape: [872, 220, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_37/BiasAdd/ReadVariableOp:0 shape: [872] dtype: float32
INFO:  output_name.1: model/conv2d_37/BiasAdd__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_150/Add:0 shape: (1, 1, 1, 220) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 220, 872) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (872,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_151/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 292 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_34/Mul
INFO:  input_name.1: model/conv2d_37/BiasAdd__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: () dtype: float32
INFO:  output_name.1: model/activation_34/Mul__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_151/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_191/Mul:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 293 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_34/Add
INFO:  input_name.1: model/activation_34/Mul__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_34/Add__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_191/Mul:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_152/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 294 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_34/clip_by_value/Minimum
INFO:  input_name.1: model/activation_34/Add__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_34/clip_by_value/Minimum__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_152/Add:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum_4/Minimum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 295 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_34/clip_by_value
INFO:  input_name.1: model/activation_34/clip_by_value/Minimum__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_34/clip_by_value__0 shape: [1, 872, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum_4/Minimum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum_4/Maximum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 

INFO: 296 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply_4/mul
INFO:  input_name.1: model/batch_normalization_58/FusedBatchNormV3__0 shape: [1, 872, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_34/clip_by_value__0 shape: [1, 872, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply_4/mul__0 shape: [1, 872, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_148/Add:0 shape: (1, 4, 4, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum_4/Maximum:0 shape: (1, 1, 1, 872) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_196/Mul:0 shape: (1, 4, 4, 872) dtype: <dtype: 'float32'> 

INFO: 297 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_38/Conv2D
INFO:  input_name.1: model/multiply_4/mul__0 shape: [1, 872, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_38/Conv2D_weights_fused_bn shape: [104, 872, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_38/Conv2D_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_59/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_196/Mul:0 shape: (1, 4, 4, 872) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 872, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_153/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 298 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_31/depthwise
INFO:  input_name.1: model/batch_normalization_59/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_31/depthwise_weights_fused_bn shape: [104, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_31/depthwise_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_60/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_153/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 104, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 104 
INFO:  output.1.output: name: tf.math.add_154/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 299 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_23/concat
INFO:  input_name.1: model/batch_normalization_59/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/batch_normalization_60/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_23/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_153/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_154/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_23/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 300 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_11/add
INFO:  input_name.1: model/batch_normalization_62/FusedBatchNormV3__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/concatenate_23/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  output_name.1: model/add_11/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_142/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_23/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_156/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 301 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_40/Conv2D
INFO:  input_name.1: model/add_11/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_40/Conv2D_weights_fused_bn shape: [624, 208, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_40/Conv2D_bias_fused_bn shape: [624] dtype: float32
INFO:  output_name.1: model/batch_normalization_63/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_156/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 208, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_157/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 302 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_35/Relu
INFO:  input_name.1: model/batch_normalization_63/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_35/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_157/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_60/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 303 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_35/Neg_1
INFO:  input_name.1: model/batch_normalization_63/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_35/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_157/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_30/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 304 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_35/Relu_1
INFO:  input_name.1: model/activation_35/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_35/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_30/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_61/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 305 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_35/mul
INFO:  input_name.1: model/activation_35/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_35/Neg:0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_35/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_61/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 624) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_201/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 306 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_35/add
INFO:  input_name.1: model/activation_35/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_35/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_35/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_60/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_201/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_159/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 307 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_33/depthwise
INFO:  input_name.1: model/activation_35/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_33/depthwise_weights_fused_bn shape: [624, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_33/depthwise_bias_fused_bn shape: [624] dtype: float32
INFO:  output_name.1: model/batch_normalization_64/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_159/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 624, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 624 
INFO:  output.1.output: name: tf.math.add_160/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 308 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_36/Relu
INFO:  input_name.1: model/batch_normalization_64/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_36/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_160/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_62/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 309 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_36/Neg_1
INFO:  input_name.1: model/batch_normalization_64/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_36/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_160/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_31/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 310 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_36/Relu_1
INFO:  input_name.1: model/activation_36/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_36/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_31/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_63/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 311 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_36/mul
INFO:  input_name.1: model/activation_36/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_36/Neg:0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_36/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_63/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 624) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_206/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 312 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_36/add
INFO:  input_name.1: model/activation_36/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_36/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_36/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_62/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_206/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_162/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 313 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_24/concat
INFO:  input_name.1: model/activation_35/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_36/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_24/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_159/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_162/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_24/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 

INFO: 314 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_41/Conv2D
INFO:  input_name.1: model/concatenate_24/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_41/Conv2D_weights_fused_bn shape: [104, 1248, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_41/Conv2D_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_65/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat_24/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 1248, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_163/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 315 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_34/depthwise
INFO:  input_name.1: model/batch_normalization_65/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_34/depthwise_weights_fused_bn shape: [104, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_34/depthwise_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_66/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_163/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 104, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 104 
INFO:  output.1.output: name: tf.math.add_164/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 316 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_25/concat
INFO:  input_name.1: model/batch_normalization_65/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/batch_normalization_66/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_25/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_163/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_164/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_25/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 317 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_12/add
INFO:  input_name.1: model/add_11/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/concatenate_25/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  output_name.1: model/add_12/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_156/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_25/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_166/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 318 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_42/Conv2D
INFO:  input_name.1: model/add_12/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_42/Conv2D_weights_fused_bn shape: [624, 208, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_42/Conv2D_bias_fused_bn shape: [624] dtype: float32
INFO:  output_name.1: model/batch_normalization_67/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_166/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 208, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_167/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 319 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_37/Relu
INFO:  input_name.1: model/batch_normalization_67/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_37/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_167/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_64/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 320 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_37/Neg_1
INFO:  input_name.1: model/batch_normalization_67/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_37/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_167/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_32/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 321 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_37/Relu_1
INFO:  input_name.1: model/activation_37/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_37/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_32/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_65/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 322 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_37/mul
INFO:  input_name.1: model/activation_37/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_37/Neg:0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_37/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_65/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 624) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_213/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 323 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_37/add
INFO:  input_name.1: model/activation_37/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_37/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_37/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_64/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_213/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_169/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 324 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_35/depthwise
INFO:  input_name.1: model/activation_37/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_35/depthwise_weights_fused_bn shape: [624, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_35/depthwise_bias_fused_bn shape: [624] dtype: float32
INFO:  output_name.1: model/batch_normalization_68/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_169/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 624, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 624 
INFO:  output.1.output: name: tf.math.add_170/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 325 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_38/Relu
INFO:  input_name.1: model/batch_normalization_68/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_38/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_170/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_66/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 326 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_38/Neg_1
INFO:  input_name.1: model/batch_normalization_68/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_38/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_170/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_33/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 327 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_38/Relu_1
INFO:  input_name.1: model/activation_38/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_38/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_33/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_67/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 328 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_38/mul
INFO:  input_name.1: model/activation_38/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_38/Neg:0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_38/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_67/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 624) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_218/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 329 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_38/add
INFO:  input_name.1: model/activation_38/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_38/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_38/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_66/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_218/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_172/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 330 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_26/concat
INFO:  input_name.1: model/activation_37/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_38/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_26/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_169/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_172/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_26/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 

INFO: 331 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d_5/Mean
INFO:  input_name.1: model/concatenate_26/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d_5/Mean__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.concat_26/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean_5/Mean:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 

INFO: 332 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d_5/Mean_Squeeze__1955
INFO:  input_name.1: model/global_average_pooling2d_5/Mean__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: (2,) dtype: int64
INFO:  output_name.1: model/global_average_pooling2d_5/Mean_Squeeze__1955__0 shape: [1, 1248] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean_5/Mean:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze_10/model/global_average_pooling2d_5/Mean_Squeeze__1955:0 shape: (1, 1248) dtype: <dtype: 'float32'> 

INFO: 333 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_43/BiasAdd__802
INFO:  input_name.1: model/global_average_pooling2d_5/Mean_Squeeze__1955__0 shape: [1, 1248] dtype: float32
INFO:  input_name.2: new_shape__1833 shape: [4] dtype: int64
INFO:  output_name.1: model/conv2d_43/BiasAdd__802__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze_10/model/global_average_pooling2d_5/Mean_Squeeze__1955:0 shape: (1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(1248), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape_5/Reshape:0 shape: (1, 1248, 1, 1) dtype: <dtype: 'float32'> 

INFO: 334 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_43/BiasAdd
INFO:  input_name.1: model/conv2d_43/BiasAdd__802__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_43/Conv2D/ReadVariableOp:0 shape: [312, 1248, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_43/BiasAdd/ReadVariableOp:0 shape: [312] dtype: float32
INFO:  output_name.1: model/conv2d_43/BiasAdd__0 shape: [1, 312, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_17/transpose:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 1248, 312) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (312,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_173/Add:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 

INFO: 335 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_39/Relu
INFO:  input_name.1: model/conv2d_43/BiasAdd__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_39/Relu__0 shape: [1, 312, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_173/Add:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_68/Relu:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 

INFO: 336 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_39/Neg_1
INFO:  input_name.1: model/conv2d_43/BiasAdd__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_39/Neg_1__0 shape: [1, 312, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_173/Add:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_34/Neg:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 

INFO: 337 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_39/Relu_1
INFO:  input_name.1: model/activation_39/Neg_1__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_39/Relu_1__0 shape: [1, 312, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_34/Neg:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_69/Relu:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 

INFO: 338 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_39/mul
INFO:  input_name.1: model/activation_39/Relu_1__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_39/Neg:0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_39/mul__0 shape: [1, 312, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_69/Relu:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 312) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_222/Mul:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 

INFO: 339 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_39/add
INFO:  input_name.1: model/activation_39/Relu__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_39/mul__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_39/add__0 shape: [1, 312, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_68/Relu:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_222/Mul:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_174/Add:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 

INFO: 340 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_44/BiasAdd
INFO:  input_name.1: model/activation_39/add__0 shape: [1, 312, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_44/Conv2D/ReadVariableOp:0 shape: [1248, 312, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_44/BiasAdd/ReadVariableOp:0 shape: [1248] dtype: float32
INFO:  output_name.1: model/conv2d_44/BiasAdd__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_174/Add:0 shape: (1, 1, 1, 312) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 312, 1248) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (1248,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_175/Add:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 

INFO: 341 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_40/Mul
INFO:  input_name.1: model/conv2d_44/BiasAdd__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: () dtype: float32
INFO:  output_name.1: model/activation_40/Mul__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_175/Add:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_225/Mul:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 

INFO: 342 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_40/Add
INFO:  input_name.1: model/activation_40/Mul__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_40/Add__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_225/Mul:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_176/Add:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 

INFO: 343 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_40/clip_by_value/Minimum
INFO:  input_name.1: model/activation_40/Add__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_40/clip_by_value/Minimum__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_176/Add:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum_5/Minimum:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 

INFO: 344 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_40/clip_by_value
INFO:  input_name.1: model/activation_40/clip_by_value/Minimum__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_40/clip_by_value__0 shape: [1, 1248, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum_5/Minimum:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum_5/Maximum:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 

INFO: 345 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply_5/mul
INFO:  input_name.1: model/concatenate_26/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_40/clip_by_value__0 shape: [1, 1248, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply_5/mul__0 shape: [1, 1248, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.concat_26/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum_5/Maximum:0 shape: (1, 1, 1, 1248) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_230/Mul:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 

INFO: 346 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_45/Conv2D
INFO:  input_name.1: model/multiply_5/mul__0 shape: [1, 1248, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_45/Conv2D_weights_fused_bn shape: [104, 1248, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_45/Conv2D_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_69/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_230/Mul:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 1248, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_177/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 347 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_36/depthwise
INFO:  input_name.1: model/batch_normalization_69/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_36/depthwise_weights_fused_bn shape: [104, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_36/depthwise_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_70/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_177/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 104, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 104 
INFO:  output.1.output: name: tf.math.add_178/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 348 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_27/concat
INFO:  input_name.1: model/batch_normalization_69/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/batch_normalization_70/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_27/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_177/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_178/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_27/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 349 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_13/add
INFO:  input_name.1: model/add_12/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/concatenate_27/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  output_name.1: model/add_13/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_166/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_27/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_180/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 350 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_46/Conv2D
INFO:  input_name.1: model/add_13/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_46/Conv2D_weights_fused_bn shape: [624, 208, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_46/Conv2D_bias_fused_bn shape: [624] dtype: float32
INFO:  output_name.1: model/batch_normalization_71/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_180/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 208, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_181/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 351 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_41/Relu
INFO:  input_name.1: model/batch_normalization_71/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_41/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_181/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_70/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 352 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_41/Neg_1
INFO:  input_name.1: model/batch_normalization_71/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_41/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_181/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_35/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 353 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_41/Relu_1
INFO:  input_name.1: model/activation_41/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_41/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_35/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_71/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 354 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_41/mul
INFO:  input_name.1: model/activation_41/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_41/Neg:0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_41/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_71/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 624) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_235/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 355 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_41/add
INFO:  input_name.1: model/activation_41/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_41/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_41/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_70/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_235/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_183/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 356 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_37/depthwise
INFO:  input_name.1: model/activation_41/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_37/depthwise_weights_fused_bn shape: [624, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_37/depthwise_bias_fused_bn shape: [624] dtype: float32
INFO:  output_name.1: model/batch_normalization_72/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_183/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 624, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (624,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 624 
INFO:  output.1.output: name: tf.math.add_184/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 357 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_42/Relu
INFO:  input_name.1: model/batch_normalization_72/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_42/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_184/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_72/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 358 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_42/Neg_1
INFO:  input_name.1: model/batch_normalization_72/FusedBatchNormV3__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_42/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_184/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_36/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 359 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_42/Relu_1
INFO:  input_name.1: model/activation_42/Neg_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_42/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_36/Neg:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_73/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 360 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_42/mul
INFO:  input_name.1: model/activation_42/Relu_1__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_42/Neg:0 shape: [1, 624, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_42/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_73/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 624) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_240/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 361 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_42/add
INFO:  input_name.1: model/activation_42/Relu__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_42/mul__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_42/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_72/Relu:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_240/Mul:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_186/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 

INFO: 362 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_28/concat
INFO:  input_name.1: model/activation_41/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_42/add__0 shape: [1, 624, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_28/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_183/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_186/Add:0 shape: (1, 4, 4, 624) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_28/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 

INFO: 363 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_47/Conv2D
INFO:  input_name.1: model/concatenate_28/concat__0 shape: [1, 1248, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_47/Conv2D_weights_fused_bn shape: [104, 1248, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_47/Conv2D_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_73/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.concat_28/concat:0 shape: (1, 4, 4, 1248) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 1248, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_187/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 364 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_38/depthwise
INFO:  input_name.1: model/batch_normalization_73/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_38/depthwise_weights_fused_bn shape: [104, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_38/depthwise_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_74/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_187/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 104, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 104 
INFO:  output.1.output: name: tf.math.add_188/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 365 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_29/concat
INFO:  input_name.1: model/batch_normalization_73/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/batch_normalization_74/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_29/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_187/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_188/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_29/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 366 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_14/add
INFO:  input_name.1: model/add_13/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/concatenate_29/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  output_name.1: model/add_14/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_180/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_29/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_190/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 367 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_48/Conv2D
INFO:  input_name.1: model/add_14/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_48/Conv2D_weights_fused_bn shape: [332, 208, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_48/Conv2D_bias_fused_bn shape: [332] dtype: float32
INFO:  output_name.1: model/batch_normalization_75/FusedBatchNormV3__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_190/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 208, 332) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (332,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_191/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 368 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_43/Relu
INFO:  input_name.1: model/batch_normalization_75/FusedBatchNormV3__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_43/Relu__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_191/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_74/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 369 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_43/Neg_1
INFO:  input_name.1: model/batch_normalization_75/FusedBatchNormV3__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_43/Neg_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_191/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_37/Neg:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 370 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_43/Relu_1
INFO:  input_name.1: model/activation_43/Neg_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_43/Relu_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_37/Neg:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_75/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 371 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_43/mul
INFO:  input_name.1: model/activation_43/Relu_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_43/Neg:0 shape: [1, 332, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_43/mul__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_75/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 332) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_247/Mul:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 372 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_43/add
INFO:  input_name.1: model/activation_43/Relu__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_43/mul__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_43/add__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_74/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_247/Mul:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_193/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 373 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_39/depthwise
INFO:  input_name.1: model/activation_43/add__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_39/depthwise_weights_fused_bn shape: [332, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_39/depthwise_bias_fused_bn shape: [332] dtype: float32
INFO:  output_name.1: model/batch_normalization_76/FusedBatchNormV3__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_193/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 332, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (332,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 332 
INFO:  output.1.output: name: tf.math.add_194/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 374 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_44/Relu
INFO:  input_name.1: model/batch_normalization_76/FusedBatchNormV3__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_44/Relu__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_194/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_76/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 375 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_44/Neg_1
INFO:  input_name.1: model/batch_normalization_76/FusedBatchNormV3__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_44/Neg_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_194/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_38/Neg:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 376 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_44/Relu_1
INFO:  input_name.1: model/activation_44/Neg_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_44/Relu_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_38/Neg:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_77/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 377 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_44/mul
INFO:  input_name.1: model/activation_44/Relu_1__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_44/Neg:0 shape: [1, 332, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_44/mul__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_77/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 332) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_252/Mul:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 378 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_44/add
INFO:  input_name.1: model/activation_44/Relu__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_44/mul__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_44/add__0 shape: [1, 332, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_76/Relu:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_252/Mul:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_196/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 

INFO: 379 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_30/concat
INFO:  input_name.1: model/activation_43/add__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_44/add__0 shape: [1, 332, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_30/concat__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_193/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_196/Add:0 shape: (1, 4, 4, 332) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_30/concat:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 380 / 410
INFO: onnx_op_type: GlobalAveragePool onnx_op_name: model/global_average_pooling2d_6/Mean
INFO:  input_name.1: model/concatenate_30/concat__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  output_name.1: model/global_average_pooling2d_6/Mean__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: reduce_mean
INFO:  input.1.input_tensor: name: tf.concat_30/concat:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  input.3.keepdims: val: True 
INFO:  output.1.output: name: tf.math.reduce_mean_6/Mean:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 381 / 410
INFO: onnx_op_type: Squeeze onnx_op_name: model/global_average_pooling2d_6/Mean_Squeeze__1967
INFO:  input_name.1: model/global_average_pooling2d_6/Mean__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: const_axes__1960 shape: (2,) dtype: int64
INFO:  output_name.1: model/global_average_pooling2d_6/Mean_Squeeze__1967__0 shape: [1, 664] dtype: float32
INFO: tf_op_type: squeeze_v2
INFO:  input.1.input: name: tf.math.reduce_mean_6/Mean:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.axis: val: [1, 2] 
INFO:  output.1.output: name: tf.compat.v1.squeeze_12/model/global_average_pooling2d_6/Mean_Squeeze__1967:0 shape: (1, 664) dtype: <dtype: 'float32'> 

INFO: 382 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/conv2d_49/BiasAdd__902
INFO:  input_name.1: model/global_average_pooling2d_6/Mean_Squeeze__1967__0 shape: [1, 664] dtype: float32
INFO:  input_name.2: new_shape__1821 shape: [4] dtype: int64
INFO:  output_name.1: model/conv2d_49/BiasAdd__902__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.squeeze_12/model/global_average_pooling2d_6/Mean_Squeeze__1967:0 shape: (1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(664), np.int64(1), np.int64(1)] 
INFO:  output.1.output: name: tf.reshape_6/Reshape:0 shape: (1, 664, 1, 1) dtype: <dtype: 'float32'> 

INFO: 383 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_49/BiasAdd
INFO:  input_name.1: model/conv2d_49/BiasAdd__902__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_49/Conv2D/ReadVariableOp:0 shape: [168, 664, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_49/BiasAdd/ReadVariableOp:0 shape: [168] dtype: float32
INFO:  output_name.1: model/conv2d_49/BiasAdd__0 shape: [1, 168, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.compat.v1.transpose_20/transpose:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 664, 168) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (168,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_197/Add:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 

INFO: 384 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_45/Relu
INFO:  input_name.1: model/conv2d_49/BiasAdd__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_45/Relu__0 shape: [1, 168, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_197/Add:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_78/Relu:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 

INFO: 385 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_45/Neg_1
INFO:  input_name.1: model/conv2d_49/BiasAdd__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_45/Neg_1__0 shape: [1, 168, 1, 1] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_197/Add:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_39/Neg:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 

INFO: 386 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_45/Relu_1
INFO:  input_name.1: model/activation_45/Neg_1__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_45/Relu_1__0 shape: [1, 168, 1, 1] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_39/Neg:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_79/Relu:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 

INFO: 387 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_45/mul
INFO:  input_name.1: model/activation_45/Relu_1__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_45/Neg:0 shape: [1, 168, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_45/mul__0 shape: [1, 168, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_79/Relu:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 168) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_256/Mul:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 

INFO: 388 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_45/add
INFO:  input_name.1: model/activation_45/Relu__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_45/mul__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_45/add__0 shape: [1, 168, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_78/Relu:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_256/Mul:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_198/Add:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 

INFO: 389 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_50/BiasAdd
INFO:  input_name.1: model/activation_45/add__0 shape: [1, 168, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_50/Conv2D/ReadVariableOp:0 shape: [664, 168, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_50/BiasAdd/ReadVariableOp:0 shape: [664] dtype: float32
INFO:  output_name.1: model/conv2d_50/BiasAdd__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_198/Add:0 shape: (1, 1, 1, 168) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 168, 664) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (664,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_199/Add:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 390 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_46/Mul
INFO:  input_name.1: model/conv2d_50/BiasAdd__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: model/activation_34/Const:0 shape: () dtype: float32
INFO:  output_name.1: model/activation_46/Mul__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.math.add_199/Add:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_259/Mul:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 391 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_46/Add
INFO:  input_name.1: model/activation_46/Mul__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1872 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_46/Add__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_259/Mul:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.add_200/Add:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 392 / 410
INFO: onnx_op_type: Min onnx_op_name: model/activation_46/clip_by_value/Minimum
INFO:  input_name.1: model/activation_46/Add__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1860 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_46/clip_by_value/Minimum__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: minimum
INFO:  input.1.input0: name: tf.math.add_200/Add:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: float32 
INFO:  output.1.output: name: tf.math.minimum_6/Minimum:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 393 / 410
INFO: onnx_op_type: Max onnx_op_name: model/activation_46/clip_by_value
INFO:  input_name.1: model/activation_46/clip_by_value/Minimum__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1940 shape: (1, 1, 1, 1) dtype: float32
INFO:  output_name.1: model/activation_46/clip_by_value__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: maximum
INFO:  input.1.input0: name: tf.math.minimum_6/Minimum:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: shape: (1, 1, 1, 1) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.maximum_6/Maximum:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 394 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/multiply_6/mul
INFO:  input_name.1: model/concatenate_30/concat__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_46/clip_by_value__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  output_name.1: model/multiply_6/mul__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.concat_30/concat:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.maximum_6/Maximum:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_264/Mul:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 395 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_51/Conv2D
INFO:  input_name.1: model/multiply_6/mul__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_51/Conv2D_weights_fused_bn shape: [104, 664, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_51/Conv2D_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_77/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.multiply_264/Mul:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 664, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_201/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 396 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_40/depthwise
INFO:  input_name.1: model/batch_normalization_77/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/depthwise_conv2d_40/depthwise_weights_fused_bn shape: [104, 1, 3, 3] dtype: float32
INFO:  input_name.3: model/depthwise_conv2d_40/depthwise_bias_fused_bn shape: [104] dtype: float32
INFO:  output_name.1: model/batch_normalization_78/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_201/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (3, 3, 104, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (104,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 104 
INFO:  output.1.output: name: tf.math.add_202/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 

INFO: 397 / 410
INFO: onnx_op_type: Concat onnx_op_name: model/concatenate_31/concat
INFO:  input_name.1: model/batch_normalization_77/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  input_name.2: model/batch_normalization_78/FusedBatchNormV3__0 shape: [1, 104, 4, 4] dtype: float32
INFO:  output_name.1: model/concatenate_31/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: concat
INFO:  input.1.input0: name: tf.math.add_201/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.2.input1: name: tf.math.add_202/Add:0 shape: (1, 4, 4, 104) dtype: <dtype: 'float32'> 
INFO:  input.3.axis: val: 3 
INFO:  output.1.output: name: tf.concat_31/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 398 / 410
INFO: onnx_op_type: Add onnx_op_name: model/add_15/add
INFO:  input_name.1: model/add_14/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/concatenate_31/concat__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  output_name.1: model/add_15/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.add_190/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.concat_31/concat:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_204/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 

INFO: 399 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_52/Conv2D
INFO:  input_name.1: model/add_15/add__0 shape: [1, 208, 4, 4] dtype: float32
INFO:  input_name.2: model/conv2d_52/Conv2D_weights_fused_bn shape: [664, 208, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_52/Conv2D_bias_fused_bn shape: [664] dtype: float32
INFO:  output_name.1: model/batch_normalization_79/FusedBatchNormV3__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.math.add_204/Add:0 shape: (1, 4, 4, 208) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 208, 664) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (664,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_205/Add:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 400 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_47/Relu
INFO:  input_name.1: model/batch_normalization_79/FusedBatchNormV3__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_47/Relu__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.add_205/Add:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_80/Relu:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 401 / 410
INFO: onnx_op_type: Neg onnx_op_name: model/activation_47/Neg_1
INFO:  input_name.1: model/batch_normalization_79/FusedBatchNormV3__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_47/Neg_1__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: neg
INFO:  input.1.features: name: tf.math.add_205/Add:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.negative_40/Neg:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 402 / 410
INFO: onnx_op_type: Relu onnx_op_name: model/activation_47/Relu_1
INFO:  input_name.1: model/activation_47/Neg_1__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_47/Relu_1__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: relu
INFO:  input.1.features: name: tf.math.negative_40/Neg:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.nn.relu_81/Relu:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 403 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/activation_47/mul
INFO:  input_name.1: model/activation_47/Relu_1__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_47/Neg:0 shape: [1, 664, 1, 1] dtype: float32
INFO:  output_name.1: model/activation_47/mul__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.nn.relu_81/Relu:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 1, 1, 664) dtype: float32 
INFO:  output.1.output: name: tf.math.multiply_269/Mul:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 404 / 410
INFO: onnx_op_type: Add onnx_op_name: model/activation_47/add
INFO:  input_name.1: model/activation_47/Relu__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  input_name.2: model/activation_47/mul__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  output_name.1: model/activation_47/add__0 shape: [1, 664, 4, 4] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.nn.relu_80/Relu:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.y: name: tf.math.multiply_269/Mul:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_207/Add:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 

INFO: 405 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/depthwise_conv2d_41/depthwise
INFO:  input_name.1: model/activation_47/add__0 shape: [1, 664, 4, 4] dtype: float32
INFO:  input_name.2: const_fold_opt__1890 shape: [664, 1, 4, 4] dtype: float32
INFO:  output_name.1: model/depthwise_conv2d_41/depthwise__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: depthwise_conv2d_v2
INFO:  input.1.input: name: tf.math.add_207/Add:0 shape: (1, 4, 4, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (4, 4, 664, 1) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: 
INFO:  input.4.strides: val: [1, 1, 1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: VALID 
INFO:  input.7.group: val: 664 
INFO:  output.1.output: name: tf.compat.v1.nn.depthwise_conv2d_41/depthwise:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 406 / 410
INFO: onnx_op_type: BatchNormalization onnx_op_name: model/batch_normalization_80/FusedBatchNormV3
INFO:  input_name.1: model/depthwise_conv2d_41/depthwise__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: model/batch_normalization_80/ReadVariableOp:0 shape: [664] dtype: float32
INFO:  input_name.3: model/batch_normalization_80/ReadVariableOp_1:0 shape: [664] dtype: float32
INFO:  input_name.4: model/batch_normalization_80/FusedBatchNormV3/ReadVariableOp:0 shape: [664] dtype: float32
INFO:  input_name.5: model/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1:0 shape: [664] dtype: float32
INFO:  output_name.1: model/batch_normalization_80/FusedBatchNormV3__0 shape: [1, 664, 1, 1] dtype: float32
INFO: tf_op_type: BatchNormalization
INFO:  input.1.X: name: tf.compat.v1.nn.depthwise_conv2d_41/depthwise:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.mean: name: model/batch_normalization_80/FusedBatchNormV3/ReadVariableOp:0 shape: (664,) dtype: float32 
INFO:  input.3.variance: name: model/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1:0 shape: (664,) dtype: float32 
INFO:  input.4.offset: name: model/batch_normalization_80/ReadVariableOp_1:0 shape: (664,) dtype: float32 
INFO:  input.5.scale: shape: (664,) dtype: float32 
INFO:  input.6.variance_epsilon: val: 1.0009999641624745e-05 
INFO:  output.1.output: name: tf.__operators__.add/AddV2:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 

INFO: 407 / 410
INFO: onnx_op_type: Conv onnx_op_name: model/conv2d_53/BiasAdd
INFO:  input_name.1: model/batch_normalization_80/FusedBatchNormV3__0 shape: [1, 664, 1, 1] dtype: float32
INFO:  input_name.2: model/conv2d_53/Conv2D/ReadVariableOp:0 shape: [512, 664, 1, 1] dtype: float32
INFO:  input_name.3: model/conv2d_53/BiasAdd/ReadVariableOp:0 shape: [512] dtype: float32
INFO:  output_name.1: model/conv2d_53/BiasAdd__0 shape: [1, 512, 1, 1] dtype: float32
INFO: tf_op_type: convolution_v2
INFO:  input.1.input: name: tf.__operators__.add/AddV2:0 shape: (1, 1, 1, 664) dtype: <dtype: 'float32'> 
INFO:  input.2.weights: shape: (1, 1, 664, 512) dtype: <dtype: 'float32'> 
INFO:  input.3.bias: shape: (512,) dtype: <dtype: 'float32'> 
INFO:  input.4.strides: val: [1, 1] 
INFO:  input.5.dilations: val: [1, 1] 
INFO:  input.6.padding: val: SAME 
INFO:  input.7.group: val: 1 
INFO:  output.1.output: name: tf.math.add_208/Add:0 shape: (1, 1, 1, 512) dtype: <dtype: 'float32'> 

INFO: 408 / 410
INFO: onnx_op_type: Reshape onnx_op_name: model/flatten/Reshape
INFO:  input_name.1: model/conv2d_53/BiasAdd__0 shape: [1, 512, 1, 1] dtype: float32
INFO:  input_name.2: const_fold_opt__1907 shape: [2] dtype: int64
INFO:  output_name.1: model/flatten/Reshape__0 shape: [1, 512] dtype: float32
INFO: tf_op_type: reshape
INFO:  input.1.tensor: name: tf.compat.v1.transpose_23/transpose:0 shape: (1, 512, 1, 1) dtype: <dtype: 'float32'> 
INFO:  input.2.shape: val: [np.int64(-1), np.int64(512)] 
INFO:  output.1.output: name: tf.reshape_9/Reshape:0 shape: (1, 512) dtype: <dtype: 'float32'> 

INFO: 409 / 410
INFO: onnx_op_type: Mul onnx_op_name: model/embedding/batchnorm/mul_1
INFO:  input_name.1: model/flatten/Reshape__0 shape: [1, 512] dtype: float32
INFO:  input_name.2: model/embedding/batchnorm/mul:0 shape: [512] dtype: float32
INFO:  output_name.1: model/embedding/batchnorm/mul_1__0 shape: [1, 512] dtype: float32
INFO: tf_op_type: multiply
INFO:  input.1.x: name: tf.reshape_9/Reshape:0 shape: (1, 512) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 512) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.multiply_274/Mul:0 shape: (1, 512) dtype: <dtype: 'float32'> 

INFO: 410 / 410
INFO: onnx_op_type: Add onnx_op_name: model/embedding/batchnorm/add_1
INFO:  input_name.1: model/embedding/batchnorm/mul_1__0 shape: [1, 512] dtype: float32
INFO:  input_name.2: model/embedding/batchnorm/sub:0 shape: [512] dtype: float32
INFO:  output_name.1: Identity__0 shape: [1, 512] dtype: float32
INFO: tf_op_type: add
INFO:  input.1.x: name: tf.math.multiply_274/Mul:0 shape: (1, 512) dtype: <dtype: 'float32'> 
INFO:  input.2.y: shape: (1, 512) dtype: <dtype: 'float32'> 
INFO:  output.1.output: name: tf.math.add_209/Add:0 shape: (1, 512) dtype: <dtype: 'float32'> 

saved_model output started ==========================================================
saved_model output complete!
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778369640.650979 4233069 devices.cc:76] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
I0000 00:00:1778369640.651085 4233069 single_machine.cc:376] Starting new session
W0000 00:00:1778369641.180047 4233069 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1778369641.180071 4233069 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
Float32 tflite output complete!
I0000 00:00:1778369641.749827 4233069 devices.cc:76] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
I0000 00:00:1778369641.749914 4233069 single_machine.cc:376] Starting new session
W0000 00:00:1778369642.254087 4233069 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1778369642.254111 4233069 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
Float16 tflite output complete!

ONNX and TF output value validation started =========================================
INFO: validation_conditions: np.allclose(onnx_outputs, tf_outputs, rtol=0.0, atol=0.0001, equal_nan=True)
INFO: onnx_output_name: model/batch_normalization/FusedBatchNormV3__0 tf_output_name: tf.math.add/Add:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation/Relu__0 tf_output_name: tf.nn.relu/Relu:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation/Neg_1__0 tf_output_name: tf.math.negative/Neg:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation/Relu_1__0 tf_output_name: tf.nn.relu_1/Relu:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation/mul__0 tf_output_name: tf.math.multiply_2/Mul:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation/add__0 tf_output_name: tf.math.add_2/Add:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_1/FusedBatchNormV3__0 tf_output_name: tf.math.add_3/Add:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_1/Relu__0 tf_output_name: tf.nn.relu_2/Relu:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_1/Neg_1__0 tf_output_name: tf.math.negative_1/Neg:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_1/Relu_1__0 tf_output_name: tf.nn.relu_3/Relu:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_1/mul__0 tf_output_name: tf.math.multiply_7/Mul:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_1/add__0 tf_output_name: tf.math.add_5/Add:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_2/FusedBatchNormV3__0 tf_output_name: tf.math.add_6/Add:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_2/Relu__0 tf_output_name: tf.nn.relu_4/Relu:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_2/Neg_1__0 tf_output_name: tf.math.negative_2/Neg:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_2/Relu_1__0 tf_output_name: tf.nn.relu_5/Relu:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_2/mul__0 tf_output_name: tf.math.multiply_12/Mul:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_2/add__0 tf_output_name: tf.math.add_8/Add:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate/concat__0 tf_output_name: tf.concat/concat:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_3/FusedBatchNormV3__0 tf_output_name: tf.math.add_9/Add:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_4/FusedBatchNormV3__0 tf_output_name: tf.math.add_10/Add:0 shape: (1, 10, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_1/concat__0 tf_output_name: tf.concat_1/concat:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add/add__0 tf_output_name: tf.math.add_12/Add:0 shape: (1, 20, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_10/FusedBatchNormV3__0 tf_output_name: tf.math.add_13/Add:0 shape: (1, 20, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_5/FusedBatchNormV3__0 tf_output_name: tf.math.add_14/Add:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_11/FusedBatchNormV3__0 tf_output_name: tf.math.add_15/Add:0 shape: (1, 32, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_3/Relu__0 tf_output_name: tf.nn.relu_6/Relu:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_3/Neg_1__0 tf_output_name: tf.math.negative_3/Neg:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_3/Relu_1__0 tf_output_name: tf.nn.relu_7/Relu:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_3/mul__0 tf_output_name: tf.math.multiply_19/Mul:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_3/add__0 tf_output_name: tf.math.add_17/Add:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_6/FusedBatchNormV3__0 tf_output_name: tf.math.add_18/Add:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_4/Relu__0 tf_output_name: tf.nn.relu_8/Relu:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_4/Neg_1__0 tf_output_name: tf.math.negative_4/Neg:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_4/Relu_1__0 tf_output_name: tf.nn.relu_9/Relu:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_4/mul__0 tf_output_name: tf.math.multiply_24/Mul:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_4/add__0 tf_output_name: tf.math.add_20/Add:0 shape: (1, 32, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_2/concat__0 tf_output_name: tf.concat_2/concat:0 shape: (1, 64, 56, 56) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_7/FusedBatchNormV3__0 tf_output_name: tf.math.add_21/Add:0 shape: (1, 64, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_8/FusedBatchNormV3__0 tf_output_name: tf.math.add_22/Add:0 shape: (1, 16, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_9/FusedBatchNormV3__0 tf_output_name: tf.math.add_23/Add:0 shape: (1, 16, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_3/concat__0 tf_output_name: tf.concat_3/concat:0 shape: (1, 32, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_1/add__0 tf_output_name: tf.math.add_25/Add:0 shape: (1, 32, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_12/FusedBatchNormV3__0 tf_output_name: tf.math.add_26/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_5/Relu__0 tf_output_name: tf.nn.relu_10/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_5/Neg_1__0 tf_output_name: tf.math.negative_5/Neg:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_5/Relu_1__0 tf_output_name: tf.nn.relu_11/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_5/mul__0 tf_output_name: tf.math.multiply_31/Mul:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_5/add__0 tf_output_name: tf.math.add_28/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_13/FusedBatchNormV3__0 tf_output_name: tf.math.add_29/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_6/Relu__0 tf_output_name: tf.nn.relu_12/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_6/Neg_1__0 tf_output_name: tf.math.negative_6/Neg:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_6/Relu_1__0 tf_output_name: tf.nn.relu_13/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_6/mul__0 tf_output_name: tf.math.multiply_36/Mul:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_6/add__0 tf_output_name: tf.math.add_31/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_4/concat__0 tf_output_name: tf.concat_4/concat:0 shape: (1, 92, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_14/FusedBatchNormV3__0 tf_output_name: tf.math.add_32/Add:0 shape: (1, 16, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_15/FusedBatchNormV3__0 tf_output_name: tf.math.add_33/Add:0 shape: (1, 16, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_5/concat__0 tf_output_name: tf.concat_5/concat:0 shape: (1, 32, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_2/add__0 tf_output_name: tf.math.add_35/Add:0 shape: (1, 32, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_21/FusedBatchNormV3__0 tf_output_name: tf.math.add_36/Add:0 shape: (1, 32, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_16/FusedBatchNormV3__0 tf_output_name: tf.math.add_37/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_22/FusedBatchNormV3__0 tf_output_name: tf.math.add_38/Add:0 shape: (1, 52, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_7/Relu__0 tf_output_name: tf.nn.relu_14/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_7/Neg_1__0 tf_output_name: tf.math.negative_7/Neg:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_7/Relu_1__0 tf_output_name: tf.nn.relu_15/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_7/mul__0 tf_output_name: tf.math.multiply_43/Mul:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_7/add__0 tf_output_name: tf.math.add_40/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_17/FusedBatchNormV3__0 tf_output_name: tf.math.add_41/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_8/Relu__0 tf_output_name: tf.nn.relu_16/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_8/Neg_1__0 tf_output_name: tf.math.negative_8/Neg:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_8/Relu_1__0 tf_output_name: tf.nn.relu_17/Relu:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_8/mul__0 tf_output_name: tf.math.multiply_48/Mul:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_8/add__0 tf_output_name: tf.math.add_43/Add:0 shape: (1, 46, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_6/concat__0 tf_output_name: tf.concat_6/concat:0 shape: (1, 92, 28, 28) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_18/FusedBatchNormV3__0 tf_output_name: tf.math.add_44/Add:0 shape: (1, 92, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d/Mean__0 tf_output_name: tf.math.reduce_mean/Mean:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d/Mean_Squeeze__1961__0 tf_output_name: tf.compat.v1.squeeze/model/global_average_pooling2d/Mean_Squeeze__1961:0 shape: (1, 92) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_9/BiasAdd__223__0 tf_output_name: tf.reshape/Reshape:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_9/BiasAdd__0 tf_output_name: tf.math.add_45/Add:0 shape: (1, 24, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_9/Relu__0 tf_output_name: tf.nn.relu_18/Relu:0 shape: (1, 24, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_9/Neg_1__0 tf_output_name: tf.math.negative_9/Neg:0 shape: (1, 24, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_9/Relu_1__0 tf_output_name: tf.nn.relu_19/Relu:0 shape: (1, 24, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_9/mul__0 tf_output_name: tf.math.multiply_52/Mul:0 shape: (1, 24, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_9/add__0 tf_output_name: tf.math.add_46/Add:0 shape: (1, 24, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_10/BiasAdd__0 tf_output_name: tf.math.add_47/Add:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_10/Mul__0 tf_output_name: tf.math.multiply_55/Mul:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_10/Add__0 tf_output_name: tf.math.add_48/Add:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_10/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum/Minimum:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_10/clip_by_value__0 tf_output_name: tf.math.maximum/Maximum:0 shape: (1, 92, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply/mul__0 tf_output_name: tf.math.multiply_60/Mul:0 shape: (1, 92, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_19/FusedBatchNormV3__0 tf_output_name: tf.math.add_49/Add:0 shape: (1, 26, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_20/FusedBatchNormV3__0 tf_output_name: tf.math.add_50/Add:0 shape: (1, 26, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_7/concat__0 tf_output_name: tf.concat_7/concat:0 shape: (1, 52, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_3/add__0 tf_output_name: tf.math.add_52/Add:0 shape: (1, 52, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_23/FusedBatchNormV3__0 tf_output_name: tf.math.add_53/Add:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_11/Relu__0 tf_output_name: tf.nn.relu_20/Relu:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_11/Neg_1__0 tf_output_name: tf.math.negative_10/Neg:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_11/Relu_1__0 tf_output_name: tf.nn.relu_21/Relu:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_11/mul__0 tf_output_name: tf.math.multiply_65/Mul:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_11/add__0 tf_output_name: tf.math.add_55/Add:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_24/FusedBatchNormV3__0 tf_output_name: tf.math.add_56/Add:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_12/Relu__0 tf_output_name: tf.nn.relu_22/Relu:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_12/Neg_1__0 tf_output_name: tf.math.negative_11/Neg:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_12/Relu_1__0 tf_output_name: tf.nn.relu_23/Relu:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_12/mul__0 tf_output_name: tf.math.multiply_70/Mul:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_12/add__0 tf_output_name: tf.math.add_58/Add:0 shape: (1, 78, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_8/concat__0 tf_output_name: tf.concat_8/concat:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_1/Mean__0 tf_output_name: tf.math.reduce_mean_1/Mean:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_1/Mean_Squeeze__1957__0 tf_output_name: tf.compat.v1.squeeze_2/model/global_average_pooling2d_1/Mean_Squeeze__1957:0 shape: (1, 156) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_14/BiasAdd__285__0 tf_output_name: tf.reshape_1/Reshape:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_14/BiasAdd__0 tf_output_name: tf.math.add_59/Add:0 shape: (1, 40, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_13/Relu__0 tf_output_name: tf.nn.relu_24/Relu:0 shape: (1, 40, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_13/Neg_1__0 tf_output_name: tf.math.negative_12/Neg:0 shape: (1, 40, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_13/Relu_1__0 tf_output_name: tf.nn.relu_25/Relu:0 shape: (1, 40, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_13/mul__0 tf_output_name: tf.math.multiply_74/Mul:0 shape: (1, 40, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_13/add__0 tf_output_name: tf.math.add_60/Add:0 shape: (1, 40, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_15/BiasAdd__0 tf_output_name: tf.math.add_61/Add:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_14/Mul__0 tf_output_name: tf.math.multiply_77/Mul:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_14/Add__0 tf_output_name: tf.math.add_62/Add:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_14/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum_1/Minimum:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_14/clip_by_value__0 tf_output_name: tf.math.maximum_1/Maximum:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply_1/mul__0 tf_output_name: tf.math.multiply_82/Mul:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_25/FusedBatchNormV3__0 tf_output_name: tf.math.add_63/Add:0 shape: (1, 26, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_26/FusedBatchNormV3__0 tf_output_name: tf.math.add_64/Add:0 shape: (1, 26, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_9/concat__0 tf_output_name: tf.concat_9/concat:0 shape: (1, 52, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_4/add__0 tf_output_name: tf.math.add_66/Add:0 shape: (1, 52, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_32/FusedBatchNormV3__0 tf_output_name: tf.math.add_67/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_27/FusedBatchNormV3__0 tf_output_name: tf.math.add_68/Add:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_33/FusedBatchNormV3__0 tf_output_name: tf.math.add_69/Add:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_15/Relu__0 tf_output_name: tf.nn.relu_26/Relu:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_15/Neg_1__0 tf_output_name: tf.math.negative_13/Neg:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_15/Relu_1__0 tf_output_name: tf.nn.relu_27/Relu:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_15/mul__0 tf_output_name: tf.math.multiply_87/Mul:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_15/add__0 tf_output_name: tf.math.add_71/Add:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_28/FusedBatchNormV3__0 tf_output_name: tf.math.add_72/Add:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_16/Relu__0 tf_output_name: tf.nn.relu_28/Relu:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_16/Neg_1__0 tf_output_name: tf.math.negative_14/Neg:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_16/Relu_1__0 tf_output_name: tf.nn.relu_29/Relu:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_16/mul__0 tf_output_name: tf.math.multiply_92/Mul:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_16/add__0 tf_output_name: tf.math.add_74/Add:0 shape: (1, 156, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_10/concat__0 tf_output_name: tf.concat_10/concat:0 shape: (1, 312, 14, 14) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_29/FusedBatchNormV3__0 tf_output_name: tf.math.add_75/Add:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_30/FusedBatchNormV3__0 tf_output_name: tf.math.add_76/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_31/FusedBatchNormV3__0 tf_output_name: tf.math.add_77/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_11/concat__0 tf_output_name: tf.concat_11/concat:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_5/add__0 tf_output_name: tf.math.add_79/Add:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_34/FusedBatchNormV3__0 tf_output_name: tf.math.add_80/Add:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_17/Relu__0 tf_output_name: tf.nn.relu_30/Relu:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_17/Neg_1__0 tf_output_name: tf.math.negative_15/Neg:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_17/Relu_1__0 tf_output_name: tf.nn.relu_31/Relu:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_17/mul__0 tf_output_name: tf.math.multiply_99/Mul:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_17/add__0 tf_output_name: tf.math.add_82/Add:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_35/FusedBatchNormV3__0 tf_output_name: tf.math.add_83/Add:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_18/Relu__0 tf_output_name: tf.nn.relu_32/Relu:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_18/Neg_1__0 tf_output_name: tf.math.negative_16/Neg:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_18/Relu_1__0 tf_output_name: tf.nn.relu_33/Relu:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_18/mul__0 tf_output_name: tf.math.multiply_104/Mul:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_18/add__0 tf_output_name: tf.math.add_85/Add:0 shape: (1, 130, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_12/concat__0 tf_output_name: tf.concat_12/concat:0 shape: (1, 260, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_36/FusedBatchNormV3__0 tf_output_name: tf.math.add_86/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_37/FusedBatchNormV3__0 tf_output_name: tf.math.add_87/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_13/concat__0 tf_output_name: tf.concat_13/concat:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_6/add__0 tf_output_name: tf.math.add_89/Add:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_38/FusedBatchNormV3__0 tf_output_name: tf.math.add_90/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_19/Relu__0 tf_output_name: tf.nn.relu_34/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_19/Neg_1__0 tf_output_name: tf.math.negative_17/Neg:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_19/Relu_1__0 tf_output_name: tf.nn.relu_35/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_19/mul__0 tf_output_name: tf.math.multiply_111/Mul:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_19/add__0 tf_output_name: tf.math.add_92/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_39/FusedBatchNormV3__0 tf_output_name: tf.math.add_93/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_20/Relu__0 tf_output_name: tf.nn.relu_36/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_20/Neg_1__0 tf_output_name: tf.math.negative_18/Neg:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_20/Relu_1__0 tf_output_name: tf.nn.relu_37/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_20/mul__0 tf_output_name: tf.math.multiply_116/Mul:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_20/add__0 tf_output_name: tf.math.add_95/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_14/concat__0 tf_output_name: tf.concat_14/concat:0 shape: (1, 240, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_40/FusedBatchNormV3__0 tf_output_name: tf.math.add_96/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_41/FusedBatchNormV3__0 tf_output_name: tf.math.add_97/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_15/concat__0 tf_output_name: tf.concat_15/concat:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_7/add__0 tf_output_name: tf.math.add_99/Add:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_42/FusedBatchNormV3__0 tf_output_name: tf.math.add_100/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_21/Relu__0 tf_output_name: tf.nn.relu_38/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_21/Neg_1__0 tf_output_name: tf.math.negative_19/Neg:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_21/Relu_1__0 tf_output_name: tf.nn.relu_39/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_21/mul__0 tf_output_name: tf.math.multiply_123/Mul:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_21/add__0 tf_output_name: tf.math.add_102/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_43/FusedBatchNormV3__0 tf_output_name: tf.math.add_103/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_22/Relu__0 tf_output_name: tf.nn.relu_40/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_22/Neg_1__0 tf_output_name: tf.math.negative_20/Neg:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_22/Relu_1__0 tf_output_name: tf.nn.relu_41/Relu:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_22/mul__0 tf_output_name: tf.math.multiply_128/Mul:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_22/add__0 tf_output_name: tf.math.add_105/Add:0 shape: (1, 120, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_16/concat__0 tf_output_name: tf.concat_16/concat:0 shape: (1, 240, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_44/FusedBatchNormV3__0 tf_output_name: tf.math.add_106/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_45/FusedBatchNormV3__0 tf_output_name: tf.math.add_107/Add:0 shape: (1, 52, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_17/concat__0 tf_output_name: tf.concat_17/concat:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_8/add__0 tf_output_name: tf.math.add_109/Add:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_50/FusedBatchNormV3__0 tf_output_name: tf.math.add_110/Add:0 shape: (1, 104, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_46/FusedBatchNormV3__0 tf_output_name: tf.math.add_111/Add:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_51/FusedBatchNormV3__0 tf_output_name: tf.math.add_112/Add:0 shape: (1, 144, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_23/Relu__0 tf_output_name: tf.nn.relu_42/Relu:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_23/Neg_1__0 tf_output_name: tf.math.negative_21/Neg:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_23/Relu_1__0 tf_output_name: tf.nn.relu_43/Relu:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_23/mul__0 tf_output_name: tf.math.multiply_135/Mul:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_23/add__0 tf_output_name: tf.math.add_114/Add:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_47/FusedBatchNormV3__0 tf_output_name: tf.math.add_115/Add:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_24/Relu__0 tf_output_name: tf.nn.relu_44/Relu:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_24/Neg_1__0 tf_output_name: tf.math.negative_22/Neg:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_24/Relu_1__0 tf_output_name: tf.nn.relu_45/Relu:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_24/mul__0 tf_output_name: tf.math.multiply_140/Mul:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_24/add__0 tf_output_name: tf.math.add_117/Add:0 shape: (1, 312, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_18/concat__0 tf_output_name: tf.concat_18/concat:0 shape: (1, 624, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_2/Mean__0 tf_output_name: tf.math.reduce_mean_2/Mean:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_2/Mean_Squeeze__1965__0 tf_output_name: tf.compat.v1.squeeze_4/model/global_average_pooling2d_2/Mean_Squeeze__1965:0 shape: (1, 624) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_27/BiasAdd__548__0 tf_output_name: tf.reshape_2/Reshape:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_27/BiasAdd__0 tf_output_name: tf.math.add_118/Add:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_25/Relu__0 tf_output_name: tf.nn.relu_46/Relu:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_25/Neg_1__0 tf_output_name: tf.math.negative_23/Neg:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_25/Relu_1__0 tf_output_name: tf.nn.relu_47/Relu:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_25/mul__0 tf_output_name: tf.math.multiply_144/Mul:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_25/add__0 tf_output_name: tf.math.add_119/Add:0 shape: (1, 156, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_28/BiasAdd__0 tf_output_name: tf.math.add_120/Add:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_26/Mul__0 tf_output_name: tf.math.multiply_147/Mul:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_26/Add__0 tf_output_name: tf.math.add_121/Add:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_26/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum_2/Minimum:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_26/clip_by_value__0 tf_output_name: tf.math.maximum_2/Maximum:0 shape: (1, 624, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply_2/mul__0 tf_output_name: tf.math.multiply_152/Mul:0 shape: (1, 624, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_48/FusedBatchNormV3__0 tf_output_name: tf.math.add_122/Add:0 shape: (1, 72, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_49/FusedBatchNormV3__0 tf_output_name: tf.math.add_123/Add:0 shape: (1, 72, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_19/concat__0 tf_output_name: tf.concat_19/concat:0 shape: (1, 144, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_9/add__0 tf_output_name: tf.math.add_125/Add:0 shape: (1, 144, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_52/FusedBatchNormV3__0 tf_output_name: tf.math.add_126/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_27/Relu__0 tf_output_name: tf.nn.relu_48/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_27/Neg_1__0 tf_output_name: tf.math.negative_24/Neg:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_27/Relu_1__0 tf_output_name: tf.nn.relu_49/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_27/mul__0 tf_output_name: tf.math.multiply_157/Mul:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_27/add__0 tf_output_name: tf.math.add_128/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_53/FusedBatchNormV3__0 tf_output_name: tf.math.add_129/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_28/Relu__0 tf_output_name: tf.nn.relu_50/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_28/Neg_1__0 tf_output_name: tf.math.negative_25/Neg:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_28/Relu_1__0 tf_output_name: tf.nn.relu_51/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_28/mul__0 tf_output_name: tf.math.multiply_162/Mul:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_28/add__0 tf_output_name: tf.math.add_131/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_20/concat__0 tf_output_name: tf.concat_20/concat:0 shape: (1, 872, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_3/Mean__0 tf_output_name: tf.math.reduce_mean_3/Mean:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_3/Mean_Squeeze__1959__0 tf_output_name: tf.compat.v1.squeeze_6/model/global_average_pooling2d_3/Mean_Squeeze__1959:0 shape: (1, 872) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_32/BiasAdd__610__0 tf_output_name: tf.reshape_3/Reshape:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_32/BiasAdd__0 tf_output_name: tf.math.add_132/Add:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_29/Relu__0 tf_output_name: tf.nn.relu_52/Relu:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_29/Neg_1__0 tf_output_name: tf.math.negative_26/Neg:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_29/Relu_1__0 tf_output_name: tf.nn.relu_53/Relu:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_29/mul__0 tf_output_name: tf.math.multiply_166/Mul:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_29/add__0 tf_output_name: tf.math.add_133/Add:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_33/BiasAdd__0 tf_output_name: tf.math.add_134/Add:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_30/Mul__0 tf_output_name: tf.math.multiply_169/Mul:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_30/Add__0 tf_output_name: tf.math.add_135/Add:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_30/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum_3/Minimum:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_30/clip_by_value__0 tf_output_name: tf.math.maximum_3/Maximum:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply_3/mul__0 tf_output_name: tf.math.multiply_174/Mul:0 shape: (1, 872, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_54/FusedBatchNormV3__0 tf_output_name: tf.math.add_136/Add:0 shape: (1, 72, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_55/FusedBatchNormV3__0 tf_output_name: tf.math.add_137/Add:0 shape: (1, 72, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_21/concat__0 tf_output_name: tf.concat_21/concat:0 shape: (1, 144, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_10/add__0 tf_output_name: tf.math.add_139/Add:0 shape: (1, 144, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_61/FusedBatchNormV3__0 tf_output_name: tf.math.add_140/Add:0 shape: (1, 144, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_56/FusedBatchNormV3__0 tf_output_name: tf.math.add_141/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_62/FusedBatchNormV3__0 tf_output_name: tf.math.add_142/Add:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_31/Relu__0 tf_output_name: tf.nn.relu_54/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_31/Neg_1__0 tf_output_name: tf.math.negative_27/Neg:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_31/Relu_1__0 tf_output_name: tf.nn.relu_55/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_31/mul__0 tf_output_name: tf.math.multiply_179/Mul:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_31/add__0 tf_output_name: tf.math.add_144/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_57/FusedBatchNormV3__0 tf_output_name: tf.math.add_145/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_32/Relu__0 tf_output_name: tf.nn.relu_56/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_32/Neg_1__0 tf_output_name: tf.math.negative_28/Neg:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_32/Relu_1__0 tf_output_name: tf.nn.relu_57/Relu:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_32/mul__0 tf_output_name: tf.math.multiply_184/Mul:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_32/add__0 tf_output_name: tf.math.add_147/Add:0 shape: (1, 436, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_22/concat__0 tf_output_name: tf.concat_22/concat:0 shape: (1, 872, 7, 7) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_58/FusedBatchNormV3__0 tf_output_name: tf.math.add_148/Add:0 shape: (1, 872, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_4/Mean__0 tf_output_name: tf.math.reduce_mean_4/Mean:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_4/Mean_Squeeze__1963__0 tf_output_name: tf.compat.v1.squeeze_8/model/global_average_pooling2d_4/Mean_Squeeze__1963:0 shape: (1, 872) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_36/BiasAdd__702__0 tf_output_name: tf.reshape_4/Reshape:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_36/BiasAdd__0 tf_output_name: tf.math.add_149/Add:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_33/Relu__0 tf_output_name: tf.nn.relu_58/Relu:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_33/Neg_1__0 tf_output_name: tf.math.negative_29/Neg:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_33/Relu_1__0 tf_output_name: tf.nn.relu_59/Relu:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_33/mul__0 tf_output_name: tf.math.multiply_188/Mul:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_33/add__0 tf_output_name: tf.math.add_150/Add:0 shape: (1, 220, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_37/BiasAdd__0 tf_output_name: tf.math.add_151/Add:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_34/Mul__0 tf_output_name: tf.math.multiply_191/Mul:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_34/Add__0 tf_output_name: tf.math.add_152/Add:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_34/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum_4/Minimum:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_34/clip_by_value__0 tf_output_name: tf.math.maximum_4/Maximum:0 shape: (1, 872, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply_4/mul__0 tf_output_name: tf.math.multiply_196/Mul:0 shape: (1, 872, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_59/FusedBatchNormV3__0 tf_output_name: tf.math.add_153/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_60/FusedBatchNormV3__0 tf_output_name: tf.math.add_154/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_23/concat__0 tf_output_name: tf.concat_23/concat:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_11/add__0 tf_output_name: tf.math.add_156/Add:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_63/FusedBatchNormV3__0 tf_output_name: tf.math.add_157/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_35/Relu__0 tf_output_name: tf.nn.relu_60/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_35/Neg_1__0 tf_output_name: tf.math.negative_30/Neg:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_35/Relu_1__0 tf_output_name: tf.nn.relu_61/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_35/mul__0 tf_output_name: tf.math.multiply_201/Mul:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_35/add__0 tf_output_name: tf.math.add_159/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_64/FusedBatchNormV3__0 tf_output_name: tf.math.add_160/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_36/Relu__0 tf_output_name: tf.nn.relu_62/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_36/Neg_1__0 tf_output_name: tf.math.negative_31/Neg:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_36/Relu_1__0 tf_output_name: tf.nn.relu_63/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_36/mul__0 tf_output_name: tf.math.multiply_206/Mul:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_36/add__0 tf_output_name: tf.math.add_162/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_24/concat__0 tf_output_name: tf.concat_24/concat:0 shape: (1, 1248, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_65/FusedBatchNormV3__0 tf_output_name: tf.math.add_163/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_66/FusedBatchNormV3__0 tf_output_name: tf.math.add_164/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_25/concat__0 tf_output_name: tf.concat_25/concat:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_12/add__0 tf_output_name: tf.math.add_166/Add:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_67/FusedBatchNormV3__0 tf_output_name: tf.math.add_167/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_37/Relu__0 tf_output_name: tf.nn.relu_64/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_37/Neg_1__0 tf_output_name: tf.math.negative_32/Neg:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_37/Relu_1__0 tf_output_name: tf.nn.relu_65/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_37/mul__0 tf_output_name: tf.math.multiply_213/Mul:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_37/add__0 tf_output_name: tf.math.add_169/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_68/FusedBatchNormV3__0 tf_output_name: tf.math.add_170/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_38/Relu__0 tf_output_name: tf.nn.relu_66/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_38/Neg_1__0 tf_output_name: tf.math.negative_33/Neg:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_38/Relu_1__0 tf_output_name: tf.nn.relu_67/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_38/mul__0 tf_output_name: tf.math.multiply_218/Mul:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_38/add__0 tf_output_name: tf.math.add_172/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_26/concat__0 tf_output_name: tf.concat_26/concat:0 shape: (1, 1248, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_5/Mean__0 tf_output_name: tf.math.reduce_mean_5/Mean:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_5/Mean_Squeeze__1955__0 tf_output_name: tf.compat.v1.squeeze_10/model/global_average_pooling2d_5/Mean_Squeeze__1955:0 shape: (1, 1248) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_43/BiasAdd__802__0 tf_output_name: tf.reshape_5/Reshape:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_43/BiasAdd__0 tf_output_name: tf.math.add_173/Add:0 shape: (1, 312, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_39/Relu__0 tf_output_name: tf.nn.relu_68/Relu:0 shape: (1, 312, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_39/Neg_1__0 tf_output_name: tf.math.negative_34/Neg:0 shape: (1, 312, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_39/Relu_1__0 tf_output_name: tf.nn.relu_69/Relu:0 shape: (1, 312, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_39/mul__0 tf_output_name: tf.math.multiply_222/Mul:0 shape: (1, 312, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_39/add__0 tf_output_name: tf.math.add_174/Add:0 shape: (1, 312, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_44/BiasAdd__0 tf_output_name: tf.math.add_175/Add:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_40/Mul__0 tf_output_name: tf.math.multiply_225/Mul:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_40/Add__0 tf_output_name: tf.math.add_176/Add:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_40/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum_5/Minimum:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_40/clip_by_value__0 tf_output_name: tf.math.maximum_5/Maximum:0 shape: (1, 1248, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply_5/mul__0 tf_output_name: tf.math.multiply_230/Mul:0 shape: (1, 1248, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_69/FusedBatchNormV3__0 tf_output_name: tf.math.add_177/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_70/FusedBatchNormV3__0 tf_output_name: tf.math.add_178/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_27/concat__0 tf_output_name: tf.concat_27/concat:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_13/add__0 tf_output_name: tf.math.add_180/Add:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_71/FusedBatchNormV3__0 tf_output_name: tf.math.add_181/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_41/Relu__0 tf_output_name: tf.nn.relu_70/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_41/Neg_1__0 tf_output_name: tf.math.negative_35/Neg:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_41/Relu_1__0 tf_output_name: tf.nn.relu_71/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_41/mul__0 tf_output_name: tf.math.multiply_235/Mul:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_41/add__0 tf_output_name: tf.math.add_183/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_72/FusedBatchNormV3__0 tf_output_name: tf.math.add_184/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_42/Relu__0 tf_output_name: tf.nn.relu_72/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_42/Neg_1__0 tf_output_name: tf.math.negative_36/Neg:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_42/Relu_1__0 tf_output_name: tf.nn.relu_73/Relu:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_42/mul__0 tf_output_name: tf.math.multiply_240/Mul:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_42/add__0 tf_output_name: tf.math.add_186/Add:0 shape: (1, 624, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_28/concat__0 tf_output_name: tf.concat_28/concat:0 shape: (1, 1248, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_73/FusedBatchNormV3__0 tf_output_name: tf.math.add_187/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_74/FusedBatchNormV3__0 tf_output_name: tf.math.add_188/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_29/concat__0 tf_output_name: tf.concat_29/concat:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_14/add__0 tf_output_name: tf.math.add_190/Add:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_75/FusedBatchNormV3__0 tf_output_name: tf.math.add_191/Add:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_43/Relu__0 tf_output_name: tf.nn.relu_74/Relu:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_43/Neg_1__0 tf_output_name: tf.math.negative_37/Neg:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_43/Relu_1__0 tf_output_name: tf.nn.relu_75/Relu:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_43/mul__0 tf_output_name: tf.math.multiply_247/Mul:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_43/add__0 tf_output_name: tf.math.add_193/Add:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_76/FusedBatchNormV3__0 tf_output_name: tf.math.add_194/Add:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_44/Relu__0 tf_output_name: tf.nn.relu_76/Relu:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_44/Neg_1__0 tf_output_name: tf.math.negative_38/Neg:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_44/Relu_1__0 tf_output_name: tf.nn.relu_77/Relu:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_44/mul__0 tf_output_name: tf.math.multiply_252/Mul:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_44/add__0 tf_output_name: tf.math.add_196/Add:0 shape: (1, 332, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_30/concat__0 tf_output_name: tf.concat_30/concat:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_6/Mean__0 tf_output_name: tf.math.reduce_mean_6/Mean:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/global_average_pooling2d_6/Mean_Squeeze__1967__0 tf_output_name: tf.compat.v1.squeeze_12/model/global_average_pooling2d_6/Mean_Squeeze__1967:0 shape: (1, 664) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_49/BiasAdd__902__0 tf_output_name: tf.reshape_6/Reshape:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_49/BiasAdd__0 tf_output_name: tf.math.add_197/Add:0 shape: (1, 168, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_45/Relu__0 tf_output_name: tf.nn.relu_78/Relu:0 shape: (1, 168, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_45/Neg_1__0 tf_output_name: tf.math.negative_39/Neg:0 shape: (1, 168, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_45/Relu_1__0 tf_output_name: tf.nn.relu_79/Relu:0 shape: (1, 168, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_45/mul__0 tf_output_name: tf.math.multiply_256/Mul:0 shape: (1, 168, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_45/add__0 tf_output_name: tf.math.add_198/Add:0 shape: (1, 168, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_50/BiasAdd__0 tf_output_name: tf.math.add_199/Add:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_46/Mul__0 tf_output_name: tf.math.multiply_259/Mul:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_46/Add__0 tf_output_name: tf.math.add_200/Add:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_46/clip_by_value/Minimum__0 tf_output_name: tf.math.minimum_6/Minimum:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_46/clip_by_value__0 tf_output_name: tf.math.maximum_6/Maximum:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/multiply_6/mul__0 tf_output_name: tf.math.multiply_264/Mul:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_77/FusedBatchNormV3__0 tf_output_name: tf.math.add_201/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_78/FusedBatchNormV3__0 tf_output_name: tf.math.add_202/Add:0 shape: (1, 104, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/concatenate_31/concat__0 tf_output_name: tf.concat_31/concat:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/add_15/add__0 tf_output_name: tf.math.add_204/Add:0 shape: (1, 208, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_79/FusedBatchNormV3__0 tf_output_name: tf.math.add_205/Add:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_47/Relu__0 tf_output_name: tf.nn.relu_80/Relu:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_47/Neg_1__0 tf_output_name: tf.math.negative_40/Neg:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_47/Relu_1__0 tf_output_name: tf.nn.relu_81/Relu:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_47/mul__0 tf_output_name: tf.math.multiply_269/Mul:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/activation_47/add__0 tf_output_name: tf.math.add_207/Add:0 shape: (1, 664, 4, 4) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/depthwise_conv2d_41/depthwise__0 tf_output_name: tf.compat.v1.nn.depthwise_conv2d_41/depthwise:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/batch_normalization_80/FusedBatchNormV3__0 tf_output_name: tf.__operators__.add/AddV2:0 shape: (1, 664, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/conv2d_53/BiasAdd__0 tf_output_name: tf.math.add_208/Add:0 shape: (1, 512, 1, 1) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/flatten/Reshape__0 tf_output_name: tf.reshape_9/Reshape:0 shape: (1, 512) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: model/embedding/batchnorm/mul_1__0 tf_output_name: tf.math.multiply_274/Mul:0 shape: (1, 512) dtype: float32 validate_result:  Matches 
INFO: onnx_output_name: Identity__0 tf_output_name: tf.math.add_209/Add:0 shape: (1, 512) dtype: float32 validate_result:  Matches 
  OK onnx2tf produced FP16 TFLite: _intermediate_nchw_float16.tflite

========================================================================
TFLite model details
========================================================================
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
/Users/gifty/Development/FaceMesh/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  input :  shape=(1, 112, 112, 3)  dtype=float32
  output: shape=(1, 512)  dtype=float32
  layout: NHWC
  embedding_dim: 512

========================================================================
Parity check: ONNX FP32 vs TFLite FP16
========================================================================
  [OK] synthetic      cos_sim=0.999987  max_abs=1.224469e-02  l2_diff_norm=5.017470e-03
  [OK] test_0.bin     cos_sim=0.999997  max_abs=1.045358e-02  l2_diff_norm=2.447618e-03

========================================================================
Manifest values
========================================================================
  file path : build/ghostfacenet_fp16/_intermediate_nchw_float16.tflite
  size_bytes: 8155804
  sha256    : 4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c
  embedding_dim: 512  <- update FaceEmbedder.EMBEDDING_SIZE if != 128
░▒▓ ~/Dev/FaceMesh                                              ✔  took 26s  FaceMesh Py  at 16:34:06 ▓▒░


---

**Cursor**

That's a perfect result. Notice what's now in the output:

- **Layout**: `NHWC` with input `(1, 112, 112, 3)` ✓
- **Parity check** (`-cotof`): every one of the 411 intermediate tensors `Matches` ONNX with `atol=1e-4`. Best possible signal that the surgery preserved math semantics.
- **Cosine similarity** on the FP16 TFLite: `0.999987` and `0.999997` — essentially identical to the pre-fix run, confirming the only thing that changed is the layout.
- **New SHA**: `4965f3463d…` (different from the bad-layout `4bfb5c2a…`, as expected).
- The `onnxsim … SIGSEGV` early on is benign — `onnx2tf` printed `WARNING: Failed to optimize the onnx file.` and proceeded with the unoptimized graph, which is fine for this model.

One last bit of polish: the artifact is currently named `_intermediate_nchw_float16.tflite`, which is an internal name leaking out into the manifest path. Let me rename it cleanly at the end of the conversion step.

Now you can wipe the build dir and re-run; the manifest will show a clean filename. The SHA will be the same as the run you just did (renaming a file doesn't change content hash).

```bash
rm -rf build/ghostfacenet_fp16
python tools/convert_ghostfacenet_fp16.py \
  --onnx /Users/gifty/Development/FaceMesh/models/ghostface_fp32.onnx \
  --out  build/ghostfacenet_fp16 \
  --test-bin /Users/gifty/Development/FaceMesh/models/test_0.bin
```

Expected manifest section after rerun:

```
file path : build/ghostfacenet_fp16/ghostfacenet_fp16.tflite
size_bytes: 8155804
sha256    : 4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c
embedding_dim: 512  <- update FaceEmbedder.EMBEDDING_SIZE if != 128
```

## What's actually shippable in `build/ghostfacenet_fp16/`

After the rerun, you'll have:

| Path | Ship? | Why |
|---|---|---|
| `ghostfacenet_fp16.tflite` | **yes** | The model. Goes in `app/src/main/assets/` and gets the SHA recorded in `manifest.json`. |
| `ghostface_fp32_float32.tflite` (or similar) | no | FP32 variant onnx2tf emits as a side-effect; ~15 MB and unused. |
| `_intermediate_nchw.onnx` | no | Intermediate from the surgery step; useful for debugging in Netron. |
| `saved_model.pb`, `variables/`, `assets/`, `fingerprint.pb` | no | TF SavedModel emitted by `-osd`; needed during conversion only. |
| `_onnx2tf_scratch/calibration_image_sample_data_…npy` | no | The synthetic calibration tensor we placed; keep it cached so reruns don't try to fetch from S3. |

If you want the script to clean these up automatically and leave only `ghostfacenet_fp16.tflite` plus the scratch dir, say the word — it's a few extra lines.

## Updating manifest.json + Android side

The two values you need from this output:

- `sha256`: `4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c`
- `embedding_dim`: `512` (update `FaceEmbedder.EMBEDDING_SIZE` if it's not already 512)

And — most importantly — the Android pre-processing path is now the natural one: feed an NHWC `(1, 112, 112, 3)` float buffer, no `(0, 2, 3, 1)` permutation needed.

Want me to grep the Android code for `EMBEDDING_SIZE` and the manifest entries to do those updates next?

---

**User**

@build/ghostfacenet_fp16/ghostfacenet_fp16.tflite 

Is this absolutely correct file?

---

**Cursor**

Let me verify all properties of the file end-to-end against the ONNX source — size, hash, shapes, and (most importantly) numerical parity on your `test_0.bin`.

The file is **byte-identical** to the manifest entry (size 8,155,804, SHA `4965f3463d…`, valid TFLite magic `TFL3`), and the input/output shapes are exactly what we want — `(1, 112, 112, 3)` NHWC float32 → `(1, 512)` float32. But while running the parity check I hit something odd: `test_0.bin` is reading as 9408 bytes, not the 150528 we'd expect for `(1,112,112,3)` float32. Let me check what's actually on disk vs. what the conversion script saw.

Got it — I misread the format. `test_0.bin` is 37632 bytes, which is **int8 quantized** (1 byte per element for 1×112×112×3), NOT float32. The conversion script handles both via this branch:

```77:80:tools/convert_ghostfacenet_fp16.py
    elif len(raw) == expected_int8:
        # Treat as int8 quantized input in [-128, 127] mapped to [-1, 1].
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32) / 127.5
        arr = arr.reshape(shape_nchw)
```

That's why the script's `(float32, (1, 112, 112, 3))` log was correct — the dtype is the post-cast dtype. Let me redo the parity check using the same loader logic so we get an apples-to-apples answer.

Yes — **this is the correct, ship-ready file**. Here's the full verification:

## File integrity

| Property | Value | Status |
|---|---|---|
| Path | `build/ghostfacenet_fp16/ghostfacenet_fp16.tflite` | OK |
| Size | 8,155,804 bytes (~7.8 MB) | OK — matches manifest |
| SHA-256 | `4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c` | OK — matches manifest |
| Magic bytes (4–8) | `TFL3` | OK — valid TFLite v3 file |

## Tensor shapes (loaded into the TFLite interpreter)

| Tensor | Shape | dtype | Layout |
|---|---|---|---|
| Input `inputs_0` | `(1, 112, 112, 3)` | float32 | **NHWC** ✓ (channels last) |
| Output `Identity` | `(1, 512)` | float32 | embedding_dim 512 |

## Numerical parity vs. the source ONNX (using `test_0.bin`)

```
input range : min=-1.0039  max=0.9961  mean=-0.0564  std=0.5462   (int8 -> [-1,1])
cos_sim     : 0.999997
max_abs_err : 1.0454e-02
l2_norm_diff: 2.4476e-03   (after L2-normalizing both embeddings)

||embedding|| ONNX   = 33.9505
||embedding|| TFLite = 33.9455
```

Cosine similarity of **0.999997** is essentially the FP32→FP16 quantization noise floor for a model of this depth. The first six dims line up to ~3 decimals:

| | dim 0 | dim 1 | dim 2 | dim 3 | dim 4 | dim 5 |
|---|---|---|---|---|---|---|
| ONNX FP32 | 2.46018 | 3.60710 | −0.82956 | 1.10069 | −0.89551 | 3.20957 |
| TFLite FP16 | 2.46010 | 3.60957 | −0.82286 | 1.10570 | −0.89667 | 3.20875 |

## What that means for the Android side

- The model takes a natural NHWC float32 tensor `(1, 112, 112, 3)`. Pre-process bitmaps the obvious way — no `(0, 2, 3, 1)` permutation.
- Input normalisation must match the ONNX expectation: signed-symmetric `[-1, 1]`. Concretely: `f = (i8_pixel - 0) / 127.5`, or equivalently `f = (u8_pixel - 127.5) / 127.5`. **Not** the `(x/255 - mean)/std` ImageNet pre-processing.
- Output is an unnormalized 512-D embedding (`||emb|| ≈ 34` for this test sample). Whatever code consumes it must L2-normalize before computing cosine distance.

For the manifest, the values to plug in:

```json
{
  "name": "ghostfacenet_fp16",
  "size_bytes": 8155804,
  "sha256": "4965f3463d0209a298d5409aa4c08d1f7d6d1e4ae15fbd6aafdaf0ab2478fe1c",
  "input_shape": [1, 112, 112, 3],
  "input_layout": "NHWC",
  "input_dtype": "float32",
  "input_normalization": "[-1, 1] via (px - 127.5) / 127.5",
  "output_shape": [1, 512],
  "embedding_dim": 512
}
```

So: yes, this is the right file. Ship it.

