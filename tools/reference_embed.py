#!/usr/bin/env python
"""
Reference embedder for the FaceMesh pipeline (O4: RGB-vs-BGR + FP precision parity).

Runs one or more 112x112 face crops through *every* GhostFaceNet variant we ship
(`ghostface_fp32.onnx`, `ghostface_fp16.tflite`, `ghostface_w8a8.tflite`) under both
RGB and BGR channel orderings, prints embeddings + a similarity matrix, and flags
which channel order is canonical.

Usage:
    source .venv/bin/activate
    python tools/reference_embed.py FACE.jpg [FACE2.jpg [FACE3.jpg ...]]

Recommended inputs to settle the channel-order question:
    - 2 photos of the *same* person (any pose/lighting OK), and
    - 1 photo of a *different* person.

The script then computes:
    - same-person cosine in each (model, channel_order)
    - different-person cosine in each (model, channel_order)

The (model, channel_order) where same-person sim is HIGH (>=0.6) and
different-person sim is LOW (<=0.4) is canonical for that model; the other
ordering will look noisy / "garbage in, garbage out".

Cross-precision parity (FP32 vs FP16 vs W8A8) is reported as a side benefit -
each pair should agree at >=0.99 cosine when fed the same input. If parity is
poor, that's a separate red flag (FP16 conversion or W8A8 calibration drift).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# Silence TF's noisy banner; do this BEFORE the import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import onnxruntime as ort  # noqa: E402

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required (`pip install Pillow`). The convert_ghostfacenet_fp16 venv "
        "ships it transitively via tensorflow; if it's missing, your venv may be split."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

INPUT_SIZE = 112        # GhostFaceNet input edge
EMBEDDING_DIM = 512     # GhostFaceNet output dim


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_face(path: Path, target: int = INPUT_SIZE) -> np.ndarray:
    """Load `path` and resize to (target, target, 3) uint8 RGB.

    We do a center-square-crop *before* resizing so we don't squash faces if the
    user provides a non-square image. For Phase A this is sufficient -- proper
    face alignment is a Phase B concern.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)  # H, W, 3 in RGB
    assert arr.shape == (target, target, 3), arr.shape
    return arr


# ---------------------------------------------------------------------------
# Channel-order + normalisation
# ---------------------------------------------------------------------------

def to_input(rgb_u8: np.ndarray, channel_order: str) -> np.ndarray:
    """Convert HWC uint8 RGB -> NHWC float32 in [-1, 1] under the requested order.

    `rgb_u8` is *always* read as RGB by Pillow; the BGR path explicitly flips
    channels before normalising so we feed the model what it would have gotten
    from cv2.imread had the trainer used OpenCV.
    """
    if channel_order == "RGB":
        chw = rgb_u8
    elif channel_order == "BGR":
        chw = rgb_u8[..., ::-1]
    else:
        raise ValueError(f"unknown channel_order={channel_order!r}")
    f = chw.astype(np.float32)
    f = (f - 127.5) / 127.5
    return f[None, ...]  # (1, H, W, 3) NHWC


# ---------------------------------------------------------------------------
# Embedder runners
# ---------------------------------------------------------------------------

@dataclass
class Embedder:
    name: str           # e.g. "FP32_ONNX"
    runner: Callable[[np.ndarray], np.ndarray]   # NHWC float32 -> (1, 512) float32

def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2(a), l2(b)))


def build_onnx_embedder(path: Path) -> Embedder:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_name = inp.name
    in_shape = inp.shape  # may be ['N', 3, 112, 112] or ['N', 112, 112, 3] etc.
    # Detect layout: GhostFaceNet ONNX is usually NCHW (Keras->ONNX export).
    # Heuristic: if the second dim is 3, it's NCHW.
    is_nchw = (len(in_shape) == 4 and in_shape[1] == 3)

    def run(nhwc: np.ndarray) -> np.ndarray:
        if is_nchw:
            x = np.transpose(nhwc, (0, 3, 1, 2)).astype(np.float32)
        else:
            x = nhwc.astype(np.float32)
        out = sess.run(None, {in_name: x})[0]  # (1, 512)
        return out

    layout = "NCHW" if is_nchw else "NHWC"
    return Embedder(name=f"FP32_ONNX[{layout}]", runner=run)


def build_tflite_embedder(path: Path, label: str) -> Embedder:
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_dtype = inp["dtype"]
    in_shape = tuple(inp["shape"])
    # GhostFaceNet TFLite ships NHWC ([1, 112, 112, 3]).
    assert in_shape[-1] == 3, f"unexpected TFLite input shape {in_shape}"

    in_quant = inp.get("quantization", (0.0, 0))   # (scale, zero_point)

    def run(nhwc: np.ndarray) -> np.ndarray:
        x = nhwc
        if in_dtype == np.int8:
            scale, zp = in_quant
            if scale == 0:
                # Some W8A8 exports keep the input in float and quantise inside
                # the graph; fall back to feeding float and let the interpreter
                # re-quantise. If this branch runs the model probably won't
                # accept int8 anyway, so escalate.
                raise RuntimeError(
                    f"TFLite input '{label}' is int8 but quantisation scale=0; "
                    "model may need a different input pipeline."
                )
            qx = np.round(x / scale + zp).clip(-128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], qx)
        elif in_dtype == np.uint8:
            scale, zp = in_quant
            qx = np.round(x / scale + zp).clip(0, 255).astype(np.uint8)
            interp.set_tensor(inp["index"], qx)
        else:
            interp.set_tensor(inp["index"], x.astype(np.float32))
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out["dtype"] != np.float32:
            scale, zp = out.get("quantization", (1.0, 0))
            y = (y.astype(np.float32) - zp) * scale
        return y

    return Embedder(name=label, runner=run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_vec(v: np.ndarray, n: int = 8) -> str:
    return ", ".join(f"{x:+.4f}" for x in v.flatten()[:n])


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, nargs="+", help="One or more face image paths.")
    p.add_argument("--fp32-onnx", type=Path, default=MODELS_DIR / "ghostface_fp32.onnx")
    p.add_argument("--fp16-tflite", type=Path, default=MODELS_DIR / "ghostface_fp16.tflite")
    p.add_argument("--w8a8-tflite", type=Path, default=MODELS_DIR / "ghostface_w8a8.tflite")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to dump full results as JSON for downstream tooling.")
    args = p.parse_args(argv)

    # ----- 1. load images -----
    print(f"# Loading {len(args.images)} image(s)")
    faces = []
    for path in args.images:
        rgb = load_face(path)
        faces.append((path, rgb))
        print(f"   loaded {path}: shape={rgb.shape} dtype={rgb.dtype} mean={rgb.mean():.1f}")
    print()

    # ----- 2. build embedders -----
    print("# Building embedders")
    embedders: List[Embedder] = []
    for path in (args.fp32_onnx, args.fp16_tflite, args.w8a8_tflite):
        if not path.exists():
            print(f"   ! missing {path}; skipping")
            continue
        try:
            if path.suffix == ".onnx":
                e = build_onnx_embedder(path)
            else:
                label = "FP16_TFLITE" if "fp16" in path.name else "W8A8_TFLITE"
                e = build_tflite_embedder(path, label)
            embedders.append(e)
            print(f"   built {e.name} <- {path.name}")
        except Exception as exc:
            print(f"   ! failed to build {path.name}: {exc}")
    print()

    # ----- 3. run every (embedder, channel_order, image) combination -----
    print("# Embedding")
    # results[embedder_name][order] = list of (path, raw_embedding, l2_embedding)
    results: Dict[str, Dict[str, List[Tuple[Path, np.ndarray, np.ndarray]]]] = {}
    for emb in embedders:
        results[emb.name] = {}
        for order in ("RGB", "BGR"):
            row: List[Tuple[Path, np.ndarray, np.ndarray]] = []
            for path, rgb in faces:
                nhwc = to_input(rgb, order)
                v = emb.runner(nhwc).reshape(-1)
                assert v.shape == (EMBEDDING_DIM,), v.shape
                row.append((path, v, l2(v)))
            results[emb.name][order] = row
            v0 = row[0][1]
            print(f"   {emb.name:<24} {order} norm={np.linalg.norm(v0):.4f} "
                  f"first8={fmt_vec(v0)}")
    print()

    # ----- 4. cross-precision parity per channel order (sanity) -----
    if len(embedders) >= 2:
        print("# Cross-precision parity (per channel order, per image)")
        print("#   Same input through different precisions of the same model should match")
        print("#   at >= 0.99 cosine. Lower means FP16 conversion or W8A8 calibration drift.")
        for order in ("RGB", "BGR"):
            print(f"   [{order}]")
            for i in range(len(faces)):
                ref = results[embedders[0].name][order][i][2]
                row = []
                for emb in embedders[1:]:
                    other = results[emb.name][order][i][2]
                    row.append(f"{emb.name}={cos(ref, other):+.4f}")
                print(f"      img{i} ({faces[i][0].name:<32}) {embedders[0].name} vs " + ", ".join(row))
        print()

    # ----- 5. pairwise similarity matrix per (embedder, order) -----
    if len(faces) >= 2:
        print("# Pairwise cosine matrices (per embedder, per channel order)")
        print("#   For SAME-person inputs: expect >= 0.6 (ideally 0.7-0.85) under canonical order")
        print("#   For DIFFERENT-person:   expect <= 0.4 (ideally 0.1-0.3) under canonical order")
        for emb in embedders:
            for order in ("RGB", "BGR"):
                print(f"   [{emb.name} / {order}]")
                row = results[emb.name][order]
                # Print matrix
                header = "      idx " + " ".join(f"   {i:>2}  " for i in range(len(faces)))
                print(header)
                for i in range(len(faces)):
                    cells = []
                    for j in range(len(faces)):
                        cells.append(f"{cos(row[i][2], row[j][2]):+.4f}")
                    print(f"      {i:>3} " + " ".join(f"{c:>7}" for c in cells))
        print()

    # ----- 6. inferred channel order (when >= 2 images) -----
    if len(faces) >= 2:
        print("# Channel-order inference")
        print("#   Highest off-diagonal cosine averaged across images is the canonical order")
        print("#   for that model (assuming user-supplied inputs are real face photos).")
        for emb in embedders:
            avgs = {}
            for order in ("RGB", "BGR"):
                row = results[emb.name][order]
                offdiag = [cos(row[i][2], row[j][2])
                           for i in range(len(faces)) for j in range(len(faces)) if i != j]
                avgs[order] = sum(offdiag) / len(offdiag) if offdiag else 0.0
            best = max(avgs, key=avgs.get)
            margin = avgs[best] - avgs["BGR" if best == "RGB" else "RGB"]
            verdict = "STRONG" if margin > 0.10 else "WEAK" if margin > 0.02 else "AMBIGUOUS"
            print(f"   {emb.name:<24} avg(RGB)={avgs['RGB']:+.4f}  avg(BGR)={avgs['BGR']:+.4f}  "
                  f"-> canonical={best}  margin={margin:+.4f}  [{verdict}]")
        print()
        print("# Reminder: these averages mix same-person and different-person pairs.")
        print("#   For the cleanest signal, label inputs explicitly e.g. with files named")
        print("#   `personA_1.jpg personA_2.jpg personB_1.jpg` and inspect the matrices above:")
        print("#     same-person cells should be HIGH under canonical order, LOW otherwise;")
        print("#     different-person cells should be LOW under canonical order regardless.")
        print()

    # ----- 7. optional JSON dump -----
    if args.json is not None:
        payload = {
            "images": [str(p) for p, _ in faces],
            "embedders": [e.name for e in embedders],
            "embeddings": {
                emb.name: {
                    order: [
                        {"path": str(path), "embedding": v.tolist(), "embedding_l2": vl.tolist()}
                        for path, v, vl in results[emb.name][order]
                    ]
                    for order in ("RGB", "BGR")
                }
                for emb in embedders
            },
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"# JSON dump -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
