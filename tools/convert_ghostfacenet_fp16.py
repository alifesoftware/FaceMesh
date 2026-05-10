#!/usr/bin/env python3
"""
Convert GhostFaceNet ONNX (FP32, NCHW) -> TFLite (FP16, NHWC) and validate parity.

Usage
-----
    pip install -r tools/requirements.txt
    python tools/convert_ghostfacenet_fp16.py \
        --onnx /path/to/ghostface_fp32.onnx \
        --out  /path/to/output_dir

What it does
------------
 1. Loads the ONNX model and prints input / output specs.
 2. Runs ONNX inference on a deterministic synthetic input AND on
    `quantized_input/test_0.bin` (if you pass --test-bin).
 3. Calls `onnx2tf` to convert ONNX (NCHW) -> TF SavedModel (NHWC).
 4. Calls TFLiteConverter to produce a Float16-quantized TFLite.
 5. Loads the TFLite, prints its input/output specs (so we can confirm
    NHWC + dtype + dim).
 6. Runs TFLite inference on the same input and compares to ONNX:
     - Cosine similarity (should be > 0.999)
     - Max absolute error
     - L2 distance on L2-normalised embeddings (the actual production metric)
 7. Prints the SHA-256 of the resulting .tflite file (paste into manifest.json).

Anything < 0.999 cosine similarity warrants investigation before shipping.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- helpers ----


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2_normalize(a), _l2_normalize(b)))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_test_input(path: Path | None, shape_nchw: tuple[int, int, int, int]) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    raw = path.read_bytes()
    expected_fp32 = int(np.prod(shape_nchw)) * 4
    expected_int8 = int(np.prod(shape_nchw))
    if len(raw) == expected_fp32:
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape_nchw).copy()
    elif len(raw) == expected_int8:
        # Treat as int8 quantized input in [-128, 127] mapped to [-1, 1].
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32) / 127.5
        arr = arr.reshape(shape_nchw)
    else:
        print(f"  ! Could not interpret {path} (size={len(raw)}); using synthetic input only.")
        return None
    print(f"  OK loaded test input from {path} ({arr.dtype}, {arr.shape})")
    return arr


def _synthetic_input(shape_nchw: tuple[int, int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed=20260509)
    return rng.uniform(low=-1.0, high=1.0, size=shape_nchw).astype(np.float32)


# ---------------------------------------------------------------- pipeline ---


def inspect_onnx(onnx_path: Path) -> tuple[str, tuple[int, int, int, int], int]:
    import onnx

    model = onnx.load(str(onnx_path))
    inputs = model.graph.input
    outputs = model.graph.output
    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    in_name = inputs[0].name
    in_shape = tuple(d.dim_value for d in inputs[0].type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value for d in outputs[0].type.tensor_type.shape.dim)

    _print_section("ONNX model")
    print(f"  input :  name='{in_name}'  shape={in_shape}")
    print(f"  output: name='{outputs[0].name}'  shape={out_shape}")

    if len(in_shape) != 4:
        raise SystemExit(f"Unexpected input rank: {in_shape}")
    embedding_dim = int(out_shape[-1])
    return in_name, in_shape, embedding_dim


def run_onnx(onnx_path: Path, in_name: str, x_nchw: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {in_name: x_nchw})[0]
    return np.asarray(out).reshape(-1).astype(np.float32)


def _strip_entry_nhwc_to_nchw_transpose(onnx_path: Path, out_path: Path) -> bool:
    """
    Many Keras->ONNX exports wrap the model in an entry `Transpose(perm=[0,3,1,2])`
    so the body can run in ONNX-native NCHW. Feeding that to `onnx2tf` confuses
    its layout heuristics and yields a TFLite with a garbled input shape such as
    (1, 112, 3, 112). We fix this by promoting the post-Transpose NCHW tensor to
    be the model input directly. `onnx2tf` then sees a normal NCHW model and
    produces a clean NHWC TFLite with input shape (1, 112, 112, 3).

    Returns True iff the surgery was applied (and `out_path` was written).
    """
    import onnx
    import onnx_graphsurgeon as gs

    g = gs.import_onnx(onnx.load(str(onnx_path)))
    if len(g.inputs) != 1:
        return False
    old_in = g.inputs[0]
    entry = next(
        (n for n in g.nodes if n.op == "Transpose" and old_in in n.inputs),
        None,
    )
    if entry is None or list(entry.attrs.get("perm", [])) != [0, 3, 1, 2]:
        return False

    # Promote the Transpose's output (NCHW) to be the new graph input.
    # Rename first to avoid a transient duplicate-name with the old NHWC input.
    nchw = entry.outputs[0]
    nchw.name = old_in.name + "__nchw_in"
    g.inputs = [nchw]
    entry.outputs.clear()  # detach so cleanup() drops the orphaned Transpose
    g.cleanup().toposort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(gs.export_onnx(g), str(out_path))
    return True


def _ensure_onnx2tf_calib_data(workdir: Path) -> Path | None:
    """
    onnx2tf unconditionally fetches a 20x128x128x3 float32 calibration .npy from
    a Wasabi S3 URL when the model has all-NHWC 4D inputs. If the network is
    flaky or behind a proxy, that download yields garbage and `np.load` fails
    with a misleading "pickled data" error. Pre-place a deterministic synthetic
    file so the conversion is reproducible without a network round-trip.
    """
    fname = "calibration_image_sample_data_20x128x128x3_float32.npy"
    target = workdir / fname
    if target.exists():
        return target
    rng = np.random.default_rng(20260509)
    arr = rng.uniform(0.0, 1.0, size=(20, 128, 128, 3)).astype(np.float32)
    np.save(target, arr)
    return target


def convert_with_onnx2tf(
    onnx_path: Path,
    out_dir: Path,
    in_name: str,
    in_shape: tuple[int, ...],
) -> Path:
    """
    Convert ONNX into TF SavedModel + Float16 TFLite via the `onnx2tf` CLI.

    Two preprocessing steps make the result actually correct for Keras-exported
    GhostFaceNet ONNX:
      1) Strip the entry NHWC->NCHW Transpose so onnx2tf sees a normal NCHW
         graph and can do its standard NCHW->NHWC conversion (yielding a
         TFLite with proper NHWC input).
      2) Pre-place onnx2tf's calibration .npy so we don't depend on a flaky
         network fetch from Wasabi S3 at conversion time.
    """
    if shutil.which("onnx2tf") is None:
        raise SystemExit("onnx2tf is not on PATH. Install with: pip install onnx2tf")

    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) Surgery: promote NCHW tensor to be the input.
    surgery_path = out_dir / "_intermediate_nchw.onnx"
    if _strip_entry_nhwc_to_nchw_transpose(onnx_path, surgery_path):
        print(f"  (stripped entry NHWC->NCHW Transpose; using {surgery_path.name})")
        effective_onnx = surgery_path
    else:
        print("  (no entry NHWC->NCHW Transpose detected; using ONNX as-is)")
        effective_onnx = onnx_path

    # (2) Pre-place calibration data for onnx2tf in a scratch cwd so the
    # workspace doesn't accumulate a 3.8 MB .npy and we don't depend on its
    # network fetch succeeding.
    scratch_cwd = out_dir / "_onnx2tf_scratch"
    scratch_cwd.mkdir(parents=True, exist_ok=True)
    calib = _ensure_onnx2tf_calib_data(scratch_cwd)
    if calib is not None:
        print(f"  (pre-placed calibration data at: {calib})")

    # Use absolute paths since we're switching cwd for the subprocess.
    cmd = [
        "onnx2tf",
        "-i", str(effective_onnx.resolve()),
        "-o", str(out_dir.resolve()),
        "-osd",                        # output SavedModel
        "-cotof",                      # check tf vs onnx output diff (built-in sanity)
        "-rtpo", "AveragePool",        # safer for some Ghost ops; harmless if unused
    ]
    print()
    print(f"$ (cd {scratch_cwd} && " + " ".join(cmd) + ")")
    result = subprocess.run(cmd, check=False, cwd=str(scratch_cwd))

    # onnx2tf's `-cotof` post-conversion check needs `sne4onnx`. If that step
    # fails AFTER both .tflite files have been written, we don't want to throw
    # the artifacts away -- this script does its own (more thorough) parity
    # check downstream. So: only treat the run as fatal if no FP16 .tflite
    # actually landed on disk.
    fp16_candidates = sorted(out_dir.glob("*float16*.tflite"))
    if result.returncode != 0 and not fp16_candidates:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.returncode != 0:
        print(f"  ! onnx2tf exited {result.returncode} but FP16 TFLite was produced; continuing.")
    if fp16_candidates:
        # onnx2tf names the FP16 artifact after the (possibly intermediate)
        # input ONNX, e.g. `_intermediate_nchw_float16.tflite`. Rename to a
        # stable, ship-ready filename so the manifest path doesn't leak our
        # scratch naming.
        produced = fp16_candidates[0]
        target = out_dir / "ghostfacenet_fp16.tflite"
        if produced != target:
            if target.exists():
                target.unlink()
            produced.rename(target)
        print(f"  OK FP16 TFLite ready: {target.name}")
        return target

    # Manual SavedModel -> TFLite FP16 fallback.
    import tensorflow as tf

    saved_model_dir = out_dir
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    target = out_dir / "ghostfacenet_fp16.tflite"
    target.write_bytes(tflite_bytes)
    print(f"  OK manual FP16 conversion produced: {target.name}")
    return target


def run_tflite(tflite_path: Path, x_nchw: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    in_shape = tuple(in_details["shape"])
    if in_shape == x_nchw.shape:
        x_in = x_nchw  # model preserved NCHW
    elif in_shape == (1, x_nchw.shape[2], x_nchw.shape[3], x_nchw.shape[1]):
        x_in = np.transpose(x_nchw, (0, 2, 3, 1))  # NCHW -> NHWC
    else:
        raise SystemExit(f"TFLite input shape {in_shape} doesn't match expected {x_nchw.shape}")

    interpreter.set_tensor(in_details["index"], x_in.astype(in_details["dtype"]))
    interpreter.invoke()
    raw = interpreter.get_tensor(out_details["index"]).reshape(-1).astype(np.float32)
    return raw, in_details, out_details


# ---------------------------------------------------------------- main ------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path, help="Path to ghostface_fp32.onnx")
    parser.add_argument("--out", default=Path("build/ghostfacenet_fp16"), type=Path, help="Output dir")
    parser.add_argument("--test-bin", type=Path, default=None, help="Optional quantized_input/test_0.bin")
    args = parser.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX file not found: {args.onnx}")
    args.out.mkdir(parents=True, exist_ok=True)

    in_name, in_shape, embedding_dim = inspect_onnx(args.onnx)

    # Build inputs (NCHW).
    test_input = _load_test_input(args.test_bin, in_shape)
    synth_input = _synthetic_input(in_shape)

    _print_section("Converting ONNX -> TFLite (FP16, NHWC)")
    tflite_path = convert_with_onnx2tf(args.onnx, args.out, in_name, in_shape)

    _print_section("TFLite model details")
    _, in_details, out_details = run_tflite(tflite_path, synth_input)
    in_tflite_shape = tuple(int(d) for d in in_details["shape"])
    out_tflite_shape = tuple(int(d) for d in out_details["shape"])
    print(f"  input :  shape={in_tflite_shape}  dtype={in_details['dtype'].__name__}")
    print(f"  output: shape={out_tflite_shape}  dtype={out_details['dtype'].__name__}")
    if len(in_tflite_shape) == 4 and in_tflite_shape[-1] == 3:
        layout = "NHWC"
    elif len(in_tflite_shape) == 4 and in_tflite_shape[1] == 3:
        layout = "NCHW"
    else:
        layout = f"unexpected ({in_tflite_shape}) -- onnx2tf may have done a partial transpose"
    print(f"  layout: {layout}")
    print(f"  embedding_dim: {int(out_details['shape'][-1])}")

    _print_section("Parity check: ONNX FP32 vs TFLite FP16")
    for label, x_nchw in (("synthetic", synth_input), ("test_0.bin", test_input)):
        if x_nchw is None:
            print(f"  -- {label}: skipped (not provided)")
            continue
        onnx_out = run_onnx(args.onnx, in_name, x_nchw)
        tflite_out, _, _ = run_tflite(tflite_path, x_nchw)
        if onnx_out.size != tflite_out.size:
            print(f"  !! {label}: size mismatch (onnx={onnx_out.size}, tflite={tflite_out.size})")
            continue
        cos = _cosine_similarity(onnx_out, tflite_out)
        max_abs = float(np.max(np.abs(onnx_out - tflite_out)))
        l2_norm_diff = float(np.linalg.norm(_l2_normalize(onnx_out) - _l2_normalize(tflite_out)))
        verdict = "OK" if cos > 0.999 else "WARN"
        print(f"  [{verdict}] {label:<14} cos_sim={cos:.6f}  max_abs={max_abs:.6e}  l2_diff_norm={l2_norm_diff:.6e}")

    _print_section("Manifest values")
    print(f"  file path : {tflite_path}")
    print(f"  size_bytes: {tflite_path.stat().st_size}")
    print(f"  sha256    : {_sha256(tflite_path)}")
    print(f"  embedding_dim: {int(out_details['shape'][-1])}  <- update PipelineConfig.Embedder.embeddingDim if changed")


if __name__ == "__main__":
    sys.exit(main())
