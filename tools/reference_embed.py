#!/usr/bin/env python
"""
Reference embedder for the FaceMesh pipeline (O4: RGB-vs-BGR + FP precision parity).

Runs one or more face images through *every* GhostFaceNet variant we ship
(`ghostface_fp32.onnx`, `ghostface_fp16.tflite`, `ghostface_w8a8.tflite`) under both
RGB and BGR channel orderings, prints embeddings + a similarity matrix, and flags
which channel order is canonical.

Two preprocessing modes:

  1. Default (loose) -- center-square-crop + bilinear resize to 112x112. Use this
     ONLY with images that are already tightly cropped face shots (face fills
     >= 70% of the frame, eyes roughly centered). Anything looser will produce
     noise: GhostFaceNet was trained on landmark-aligned crops where eyes/nose/
     mouth land at fixed canonical positions, and a loose center-crop won't put
     them there.

  2. --detect-and-align (recommended) -- mirrors the Android pipeline end-to-end:
        EXIF transpose -> downsample (max long edge 1280)
        -> BlazeFace short-range -> highest-score face
        -> 4-point ArcFace affine warp to 112x112
     This is the only mode that gives clean signal on raw phone photos.

Usage:
    source .venv/bin/activate
    # Tight pre-cropped 112x112 face PNGs:
    python tools/reference_embed.py FACE.jpg [FACE2.jpg ...]
    # Raw phone photos (preferred):
    python tools/reference_embed.py --detect-and-align IMG.jpg [IMG2.jpg ...]

Recommended inputs to settle the channel-order question:
    - 2 photos of the *same* person (any pose/lighting OK), and
    - 1 photo of a *different* person.

The (model, channel_order) where same-person sim is HIGH (>=0.6) and
different-person sim is LOW (<=0.4) is canonical for that model; the other
ordering will look noisy / "garbage in, garbage out".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Silence TF's noisy banner; do this BEFORE the import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import onnxruntime as ort  # noqa: E402

try:
    from PIL import Image, ImageOps
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
# BlazeFace (Python port of both detector variants from the Android pipeline)
# ---------------------------------------------------------------------------

# These constants mirror PipelineConfig.Detector and PipelineConfig.Filters
# defaults so that running this script on the same image as the Android app
# yields the same detected faces and landmarks.
DETECTOR_VARIANT_SHORT_RANGE: str = "short_range"
DETECTOR_VARIANT_FULL_RANGE: str = "full_range"
DETECTOR_VARIANTS: Tuple[str, ...] = (DETECTOR_VARIANT_SHORT_RANGE, DETECTOR_VARIANT_FULL_RANGE)

# Short-range model contract (mirrors PipelineConfig.Detector.ShortRange).
BLAZEFACE_SR_INPUT_SIZE: int = 128
BLAZEFACE_SR_NUM_ANCHORS: int = 896
BLAZEFACE_SR_REG_STRIDE: int = 16

# Full-range model contract (mirrors PipelineConfig.Detector.FullRange).
BLAZEFACE_FR_INPUT_SIZE: int = 192
BLAZEFACE_FR_NUM_ANCHORS: int = 2304
BLAZEFACE_FR_REG_STRIDE: int = 16

# Backward-compat aliases pointing at the short-range constants. Older callers
# (and some downstream tooling) reference the un-prefixed names.
BLAZEFACE_INPUT_SIZE: int = BLAZEFACE_SR_INPUT_SIZE
BLAZEFACE_NUM_ANCHORS: int = BLAZEFACE_SR_NUM_ANCHORS
BLAZEFACE_REG_STRIDE: int = BLAZEFACE_SR_REG_STRIDE

BLAZEFACE_DEFAULT_SCORE_THRESHOLD: float = 0.55
BLAZEFACE_NMS_IOU_THRESHOLD: float = 0.30
DECODE_MAX_LONG_EDGE: int = 1280   # mirrors PipelineConfig.Decode.maxLongEdgePx

# ArcFace canonical landmark template (mirrors PipelineConfig.Aligner.canonicalLandmarkTemplate).
# Order: rightEye, leftEye, noseTip, mouthCenter; positions are relative to a 112x112 frame.
ARCFACE_TEMPLATE: List[Tuple[float, float]] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (56.1396, 92.2848),
]

LANDMARK_NAMES: Tuple[str, ...] = (
    "rightEye", "leftEye", "noseTip", "mouthCenter", "rightEarTragion", "leftEarTragion",
)


def _generate_blazeface_anchors_short_range() -> np.ndarray:
    """Mirrors `BlazeFaceShortRangeAnchors.FRONT_128`: 16x16x2 + 8x8x6 = 896 anchors."""
    out: List[Tuple[float, float]] = []
    for grid_size, anchors_per_cell in ((16, 2), (8, 6)):
        for y in range(grid_size):
            for x in range(grid_size):
                cx = (x + 0.5) / grid_size
                cy = (y + 0.5) / grid_size
                for _ in range(anchors_per_cell):
                    out.append((cx, cy))
    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape == (BLAZEFACE_SR_NUM_ANCHORS, 2), arr.shape
    return arr


def _generate_blazeface_anchors_full_range() -> np.ndarray:
    """Mirrors `BlazeFaceFullRangeAnchors.GRID_192`: a single 48x48x1 grid (stride 4) -> 2304."""
    out: List[Tuple[float, float]] = []
    grid_size = 48
    for y in range(grid_size):
        for x in range(grid_size):
            cx = (x + 0.5) / grid_size
            cy = (y + 0.5) / grid_size
            out.append((cx, cy))
    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape == (BLAZEFACE_FR_NUM_ANCHORS, 2), arr.shape
    return arr


def _generate_blazeface_anchors() -> np.ndarray:
    """Backward-compat alias: the short-range anchor table."""
    return _generate_blazeface_anchors_short_range()


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid; the same shape as Kotlin's BlazeFaceDecoder.sigmoid
    # but vectorised for the entire 896-anchor classifier output.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out.astype(np.float32)


def _letterbox_to(rgb_arr: np.ndarray, size: int) -> Tuple[np.ndarray, float, float, float]:
    """Letterbox-resize the source array into a `size x size x 3` canvas.

    Returns `(canvas_uint8, scale, pad_x, pad_y)` so the caller can unproject
    detector outputs back to source pixel coordinates.

    Mirrors `BlazeFaceDetector.prepareInput` exactly: aspect-preserving resize
    followed by symmetric black padding.
    """
    h, w = rgb_arr.shape[:2]
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    img = Image.fromarray(rgb_arr)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, (int(round(pad_x)), int(round(pad_y))))
    return np.asarray(canvas, dtype=np.uint8), scale, pad_x, pad_y


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il = max(al, bl); it = max(at, bt)
    ir = min(ar, br); ib = min(ab, bb)
    if ir <= il or ib <= it:
        return 0.0
    inter = (ir - il) * (ib - it)
    union = (ar - al) * (ab - at) + (br - bl) * (bb - bt) - inter
    return float(inter / union) if union > 0 else 0.0


def _weighted_nms(candidates: List[Dict], iou_threshold: float) -> List[Dict]:
    """Two-pass IoU-based weighted NMS, matching `BlazeFaceDecoder.weightedNms`.

    Each candidate is a dict with keys: bbox=(l,t,r,b), landmarks={name:(x,y)}, score.
    Returns the merged list, with each output's bbox + landmarks score-weight-averaged
    over its overlap group and score taken from the top member.
    """
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda c: -c["score"])
    kept: List[Dict] = []
    while sorted_c:
        anchor = sorted_c.pop(0)
        group = [anchor]
        remaining = []
        for other in sorted_c:
            if _iou(anchor["bbox"], other["bbox"]) > iou_threshold:
                group.append(other)
            else:
                remaining.append(other)
        sorted_c = remaining
        if len(group) == 1:
            kept.append(anchor)
            continue
        total = sum(c["score"] for c in group)
        if total == 0:
            kept.append(anchor)
            continue
        bbox = tuple(sum(c["bbox"][i] * c["score"] for c in group) / total for i in range(4))
        lm = {}
        for nm in anchor["landmarks"]:
            lm[nm] = (
                sum(c["landmarks"][nm][0] * c["score"] for c in group) / total,
                sum(c["landmarks"][nm][1] * c["score"] for c in group) / total,
            )
        kept.append({"bbox": bbox, "landmarks": lm, "score": anchor["score"]})
    return kept


def _build_blazeface_interp(path: Path) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def _resolve_blazeface_output_indices(
    interp: tf.lite.Interpreter,
    reg_stride: int = BLAZEFACE_SR_REG_STRIDE,
) -> Tuple[int, int]:
    """Return (regressors_idx, classifications_idx) into get_output_details().

    Name-based lookup with a shape-based fallback. Both BlazeFace variants
    name their outputs with substrings 'reg' and 'class'/'classif' (e.g. the
    short-range model exports `regressors` / `classificators`, the full-range
    model exports `reshaped_regressor_face_4` / `reshaped_classifier_face_4`).
    """
    out = interp.get_output_details()
    reg_idx = next((i for i, o in enumerate(out) if "reg" in o["name"].lower()), None)
    cls_idx = next(
        (i for i, o in enumerate(out) if "class" in o["name"].lower()),
        None,
    )
    if reg_idx is None or cls_idx is None:
        for i, o in enumerate(out):
            tail = int(o["shape"][-1])
            if tail == reg_stride and reg_idx is None:
                reg_idx = i
            elif tail == 1 and cls_idx is None:
                cls_idx = i
    if reg_idx is None or cls_idx is None:
        names = [o["name"] for o in out]
        raise RuntimeError(f"Could not resolve BlazeFace outputs from {names}")
    return reg_idx, cls_idx


def _run_blazeface(
    interp: tf.lite.Interpreter,
    rgb_arr: np.ndarray,
    variant: str = DETECTOR_VARIANT_SHORT_RANGE,
    score_threshold: float = BLAZEFACE_DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = BLAZEFACE_NMS_IOU_THRESHOLD,
) -> List[Dict]:
    """Run BlazeFace over `rgb_arr` (HxWx3 uint8 RGB), dispatching on variant.

    Returns a list of detected faces (post-NMS) in source pixel coordinates.
    Each face dict has keys: bbox=(l,t,r,b), landmarks={name:(x,y)}, score.

    Mirrors the per-variant detectors `BlazeFaceShortRangeDetector` /
    `BlazeFaceFullRangeDetector` on the Android side -- same input size, same
    anchor table, same regressor unpacking, same letterbox + [-1,1] norm.
    """
    if variant == DETECTOR_VARIANT_SHORT_RANGE:
        input_size = BLAZEFACE_SR_INPUT_SIZE
        num_anchors = BLAZEFACE_SR_NUM_ANCHORS
        reg_stride = BLAZEFACE_SR_REG_STRIDE
        anchors = _generate_blazeface_anchors_short_range()
    elif variant == DETECTOR_VARIANT_FULL_RANGE:
        input_size = BLAZEFACE_FR_INPUT_SIZE
        num_anchors = BLAZEFACE_FR_NUM_ANCHORS
        reg_stride = BLAZEFACE_FR_REG_STRIDE
        anchors = _generate_blazeface_anchors_full_range()
    else:
        raise ValueError(f"unknown detector variant {variant!r}; use one of {DETECTOR_VARIANTS}")

    src_h, src_w = rgb_arr.shape[:2]
    letterboxed, scale, pad_x, pad_y = _letterbox_to(rgb_arr, input_size)
    f = (letterboxed.astype(np.float32) - 127.5) / 127.5  # [-1, 1] like Kotlin
    nhwc = f[None, ...]

    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], nhwc)
    interp.invoke()

    reg_idx, cls_idx = _resolve_blazeface_output_indices(interp, reg_stride)
    out = interp.get_output_details()
    regs = interp.get_tensor(out[reg_idx]["index"]).reshape(-1)
    cls = interp.get_tensor(out[cls_idx]["index"]).reshape(-1)
    if regs.size != num_anchors * reg_stride:
        raise RuntimeError(
            f"unexpected regressor size {regs.size}, expected {num_anchors * reg_stride} "
            f"(variant={variant})"
        )
    if cls.size != num_anchors:
        raise RuntimeError(
            f"unexpected classifier size {cls.size}, expected {num_anchors} (variant={variant})"
        )
    regs = regs.reshape(num_anchors, reg_stride)

    scores = _sigmoid_np(cls)
    keep = scores >= score_threshold

    candidates: List[Dict] = []

    def to_src(x_in: float, y_in: float) -> Tuple[float, float]:
        # Letterbox -> source: shift by padding then divide by scale.
        return ((x_in - pad_x) / scale, (y_in - pad_y) / scale)

    for i in np.where(keep)[0]:
        anchor_cx = float(anchors[i, 0])
        anchor_cy = float(anchors[i, 1])
        # Box decode: same math as BlazeFace*Decoder.kt.
        cx_input = anchor_cx * input_size + float(regs[i, 0])
        cy_input = anchor_cy * input_size + float(regs[i, 1])
        w_input = float(regs[i, 2])
        h_input = float(regs[i, 3])

        l_in = cx_input - w_input / 2.0
        t_in = cy_input - h_input / 2.0
        r_in = cx_input + w_input / 2.0
        b_in = cy_input + h_input / 2.0

        l_src, t_src = to_src(l_in, t_in)
        r_src, b_src = to_src(r_in, b_in)

        l_src = max(0.0, min(float(src_w), l_src))
        t_src = max(0.0, min(float(src_h), t_src))
        r_src = max(0.0, min(float(src_w), r_src))
        b_src = max(0.0, min(float(src_h), b_src))

        if r_src <= l_src or b_src <= t_src:
            continue

        landmarks: Dict[str, Tuple[float, float]] = {}
        for li, nm in enumerate(LANDMARK_NAMES):
            base = 4 + li * 2
            lx_input = anchor_cx * input_size + float(regs[i, base])
            ly_input = anchor_cy * input_size + float(regs[i, base + 1])
            lm_x, lm_y = to_src(lx_input, ly_input)
            lm_x = max(0.0, min(float(src_w), lm_x))
            lm_y = max(0.0, min(float(src_h), lm_y))
            landmarks[nm] = (lm_x, lm_y)

        candidates.append({
            "bbox": (l_src, t_src, r_src, b_src),
            "landmarks": landmarks,
            "score": float(scores[i]),
        })

    return _weighted_nms(candidates, iou_threshold)


# ---------------------------------------------------------------------------
# 4-point perspective solve + ArcFace affine warp via PIL
# ---------------------------------------------------------------------------

def _similarity_2d_lstsq(
    src: Sequence[Tuple[float, float]],
    dst: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Least-squares fit of a 2D similarity transform mapping `src` points to `dst` points.

    A similarity transform has 4 DOF (uniform scale, rotation, 2D translation):

        [dx]   [ a  -b ] [sx]   [tx]
        [dy] = [ b   a ] [sy] + [ty]

    For N >= 2 point pairs this is overdetermined; we solve via least squares.
    Returns a 3x3 affine matrix (perspective row is `[0, 0, 1]`) suitable for use
    with PIL.Image.PERSPECTIVE coefficients.

    This is the canonical ArcFace-family alignment (matches InsightFace's
    `cv2.estimateAffinePartial2D` recipe). Unlike the 4-point perspective fit it
    cannot match arbitrary 4-landmark configurations exactly, but it never
    introduces shear or perspective distortion of the interior face pixels.
    """
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError(f"need >=2 matching point pairs, got src={len(src)} dst={len(dst)}")
    n = len(src)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src, dst)):
        A[2 * i] = [sx, -sy, 1.0, 0.0]
        A[2 * i + 1] = [sy, sx, 0.0, 1.0]
        b[2 * i] = dx
        b[2 * i + 1] = dy
    soln, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, tx, ty = soln
    return np.array([
        [a, -b_, tx],
        [b_,  a, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _perspective_4pt_homography(
    src: Sequence[Tuple[float, float]],
    dst: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Solve the 8-DOF perspective transform that maps `src` points to `dst` points.

    Returns a 3x3 homography H such that `dst_h = H @ [src_x, src_y, 1]^T` (in
    homogeneous coordinates, with `h22 = 1`).

    Matches Android's `Matrix.setPolyToPoly(src, dst, count=4)` which fits a
    perspective transform from 4 source points to 4 destination points. Provided
    as an opt-in option (--alignment-mode perspective) for diagnostic parity
    with the Android pipeline; not recommended for production use because the
    8-DOF fit is numerically unstable when the underlying transform is closer
    to a similarity (the typical case for face alignment).
    """
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("perspective_4pt requires exactly 4 source and 4 destination points")
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        sx, sy = src[i]
        dx, dy = dst[i]
        A[2 * i] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy]
        A[2 * i + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy]
        b[2 * i] = dx
        b[2 * i + 1] = dy
    try:
        h = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ], dtype=np.float64)


ALIGNMENT_MODE_SIMILARITY: str = "similarity"
ALIGNMENT_MODE_PERSPECTIVE: str = "perspective"
ALIGNMENT_MODES: Tuple[str, ...] = (ALIGNMENT_MODE_SIMILARITY, ALIGNMENT_MODE_PERSPECTIVE)


def _warp_to_arcface(
    rgb_arr: np.ndarray,
    landmarks: Dict[str, Tuple[float, float]],
    output_size: int = INPUT_SIZE,
    mode: str = ALIGNMENT_MODE_SIMILARITY,
) -> np.ndarray:
    """Warp the source RGB array to the canonical 112x112 ArcFace pose.

    `mode = 'similarity'` (default, recommended) fits a 4-DOF similarity
    transform (rotation + uniform scale + translation) by least squares - the
    canonical ArcFace alignment recipe. Doesn't match the 4 landmarks exactly
    but never distorts the interior of the face.

    `mode = 'perspective'` fits an 8-DOF perspective transform that hits the 4
    landmarks exactly, mirroring Android's `Matrix.setPolyToPoly(src, dst, 4)`.
    Useful for parity testing with the on-device pipeline; produces unstable /
    distorted crops when the underlying ground-truth transform is closer to a
    similarity (the typical case).
    """
    src_pts = [
        landmarks["rightEye"],
        landmarks["leftEye"],
        landmarks["noseTip"],
        landmarks["mouthCenter"],
    ]
    dst_pts = ARCFACE_TEMPLATE
    # PIL's Image.PERSPECTIVE wants the INVERSE transform (output -> input). The
    # cleanest way to get that is to solve the system with src and dst swapped:
    # given (output_pixel -> input_pixel) sample pairs, solve for the matrix that
    # maps any output pixel to its source location.
    if mode == ALIGNMENT_MODE_SIMILARITY:
        h_inv = _similarity_2d_lstsq(dst_pts, src_pts)
    elif mode == ALIGNMENT_MODE_PERSPECTIVE:
        h_inv = _perspective_4pt_homography(dst_pts, src_pts)
    else:
        raise ValueError(f"unknown alignment mode {mode!r}; use one of {ALIGNMENT_MODES}")
    coeffs = (
        h_inv[0, 0], h_inv[0, 1], h_inv[0, 2],
        h_inv[1, 0], h_inv[1, 1], h_inv[1, 2],
        h_inv[2, 0], h_inv[2, 1],
    )
    src_img = Image.fromarray(rgb_arr)
    warped = src_img.transform(
        (output_size, output_size),
        Image.PERSPECTIVE,
        coeffs,
        Image.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(warped, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Image loading: simple center-crop OR full BlazeFace + ArcFace align
# ---------------------------------------------------------------------------

def load_face(path: Path, target: int = INPUT_SIZE) -> np.ndarray:
    """Center-square-crop + bilinear resize. Caller should pass already-aligned faces."""
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    assert arr.shape == (target, target, 3), arr.shape
    return arr


def detect_and_align_face(
    path: Path,
    blazeface_interp: tf.lite.Interpreter,
    score_threshold: float = BLAZEFACE_DEFAULT_SCORE_THRESHOLD,
    max_long_edge: int = DECODE_MAX_LONG_EDGE,
    alignment_mode: str = ALIGNMENT_MODE_SIMILARITY,
    detector_variant: str = DETECTOR_VARIANT_SHORT_RANGE,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Mirror the Android pipeline up to the embedder input -- single highest-score face.

    Returns (aligned_face_uint8, info). On detection failure aligned_face is None
    and info["reason"] tells you why. For the multi-face flavour (the actual
    Android `FaceProcessor.process` behaviour), use [detect_and_align_all_faces].
    """
    aligned_list, info = detect_and_align_all_faces(
        path,
        blazeface_interp,
        score_threshold=score_threshold,
        max_long_edge=max_long_edge,
        alignment_mode=alignment_mode,
        detector_variant=detector_variant,
    )
    if not aligned_list:
        return None, info
    # Pick the highest-score face for the single-face caller.
    best_idx = int(np.argmax([f["score"] for f in info["faces"]]))
    return aligned_list[best_idx], {
        **info["faces"][best_idx],
        "src_size": info["src_size"],
        "n_detected": len(aligned_list),
        "alignment_mode": alignment_mode,
        "detector_variant": detector_variant,
        "reason": "aligned",
    }


def detect_and_align_all_faces(
    path: Path,
    blazeface_interp: tf.lite.Interpreter,
    score_threshold: float = BLAZEFACE_DEFAULT_SCORE_THRESHOLD,
    max_long_edge: int = DECODE_MAX_LONG_EDGE,
    alignment_mode: str = ALIGNMENT_MODE_SIMILARITY,
    detector_variant: str = DETECTOR_VARIANT_SHORT_RANGE,
) -> Tuple[List[np.ndarray], Dict]:
    """Run the full Android pipeline (per FaceProcessor.process) for ALL faces.

    Returns (aligned_face_list, info_dict). `info_dict["faces"]` is a list of
    per-face dicts {bbox, landmarks, score} parallel to `aligned_face_list`.
    Returns ([], info_dict) when no face survives detection.
    """
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    w0, h0 = img.size
    # Mirror BitmapDecoder's max-long-edge cap.
    longest = max(w0, h0)
    if longest > max_long_edge:
        s = max_long_edge / float(longest)
        img = img.resize((int(round(w0 * s)), int(round(h0 * s))), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    h_src, w_src = rgb.shape[:2]

    faces = _run_blazeface(
        blazeface_interp,
        rgb,
        variant=detector_variant,
        score_threshold=score_threshold,
    )
    if not faces:
        return [], {
            "reason": "no_faces_detected",
            "src_size": (w_src, h_src),
            "score_threshold": score_threshold,
            "detector_variant": detector_variant,
            "faces": [],
        }
    aligned_list: List[np.ndarray] = [
        _warp_to_arcface(rgb, f["landmarks"], mode=alignment_mode) for f in faces
    ]
    return aligned_list, {
        "reason": "aligned",
        "src_size": (w_src, h_src),
        "n_detected": len(faces),
        "alignment_mode": alignment_mode,
        "detector_variant": detector_variant,
        "faces": faces,
    }


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


def to_input(rgb_u8: np.ndarray, channel_order: str) -> np.ndarray:
    """Convert HWC uint8 RGB -> NHWC float32 in [-1, 1] under the requested order."""
    if channel_order == "RGB":
        chw = rgb_u8
    elif channel_order == "BGR":
        chw = rgb_u8[..., ::-1]
    else:
        raise ValueError(f"unknown channel_order={channel_order!r}")
    f = chw.astype(np.float32)
    f = (f - 127.5) / 127.5
    return f[None, ...]


def build_onnx_embedder(path: Path) -> Embedder:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_name = inp.name
    in_shape = inp.shape
    is_nchw = (len(in_shape) == 4 and in_shape[1] == 3)

    def run(nhwc: np.ndarray) -> np.ndarray:
        if is_nchw:
            x = np.transpose(nhwc, (0, 3, 1, 2)).astype(np.float32)
        else:
            x = nhwc.astype(np.float32)
        out = sess.run(None, {in_name: x})[0]
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
    assert in_shape[-1] == 3, f"unexpected TFLite input shape {in_shape}"

    in_quant = inp.get("quantization", (0.0, 0))

    def run(nhwc: np.ndarray) -> np.ndarray:
        x = nhwc
        if in_dtype == np.int8:
            scale, zp = in_quant
            if scale == 0:
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
    p.add_argument(
        "--detect-and-align",
        action="store_true",
        help="Run BlazeFace + ArcFace 4-point warp before embedding (recommended).",
    )
    p.add_argument(
        "--blazeface-tflite",
        type=Path,
        default=MODELS_DIR / "face_detection_short_range.tflite",
        help="BlazeFace short-range model used by --detect-and-align.",
    )
    p.add_argument(
        "--blazeface-score-threshold",
        type=float,
        default=BLAZEFACE_DEFAULT_SCORE_THRESHOLD,
        help=f"Score threshold for BlazeFace (default {BLAZEFACE_DEFAULT_SCORE_THRESHOLD}).",
    )
    p.add_argument(
        "--alignment-mode",
        choices=ALIGNMENT_MODES,
        default=ALIGNMENT_MODE_SIMILARITY,
        help="Warp model: 'similarity' (4-DOF, recommended; canonical ArcFace alignment) "
             "or 'perspective' (8-DOF, mirrors Android's setPolyToPoly(4); diagnostic only).",
    )
    p.add_argument(
        "--detector-variant",
        choices=DETECTOR_VARIANTS,
        default=DETECTOR_VARIANT_SHORT_RANGE,
        help="BlazeFace variant: 'short_range' (128x128, 896 anchors; matches "
             "face_detection_short_range.tflite) or 'full_range' (192x192, 2304 anchors; "
             "matches face_detection_full_range.tflite). Pass --blazeface-tflite alongside "
             "to point at the matching model file.",
    )
    p.add_argument(
        "--save-aligned-dir",
        type=Path,
        default=None,
        help="If set, write each aligned 112x112 face crop as a PNG into this directory "
             "(only used with --detect-and-align). Lets you visually verify the alignment.",
    )
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to dump full results as JSON for downstream tooling.")
    args = p.parse_args(argv)

    # ----- 1. load + (optionally) detect-and-align -----
    blaze_interp = None
    if args.detect_and_align:
        if not args.blazeface_tflite.exists():
            print(f"! BlazeFace model not found at {args.blazeface_tflite}; aborting", file=sys.stderr)
            return 2
        print(f"# Loading BlazeFace from {args.blazeface_tflite.name}")
        blaze_interp = _build_blazeface_interp(args.blazeface_tflite)

    print(f"# Loading {len(args.images)} image(s) "
          f"(mode={'detect-and-align' if args.detect_and_align else 'center-crop'})")
    faces: List[Tuple[Path, np.ndarray]] = []
    skipped: List[Tuple[Path, Dict]] = []
    save_dir = args.save_aligned_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        if args.detect_and_align:
            aligned, info = detect_and_align_face(
                path,
                blaze_interp,
                score_threshold=args.blazeface_score_threshold,
                alignment_mode=args.alignment_mode,
                detector_variant=args.detector_variant,
            )
            if aligned is None:
                print(f"   ! skip {path.name}: {info.get('reason')} (src={info.get('src_size')})")
                skipped.append((path, info))
                continue
            lm = info["landmarks"]
            print(
                f"   aligned {path.name}: src={info['src_size']} "
                f"score={info['score']:+.3f} n_det={info['n_detected']} "
                f"mode={info['alignment_mode']}"
            )
            print(
                f"      bbox=({info['bbox'][0]:.1f},{info['bbox'][1]:.1f},"
                f"{info['bbox'][2]:.1f},{info['bbox'][3]:.1f}) "
                f"R-eye=({lm['rightEye'][0]:.1f},{lm['rightEye'][1]:.1f}) "
                f"L-eye=({lm['leftEye'][0]:.1f},{lm['leftEye'][1]:.1f}) "
                f"nose=({lm['noseTip'][0]:.1f},{lm['noseTip'][1]:.1f}) "
                f"mouth=({lm['mouthCenter'][0]:.1f},{lm['mouthCenter'][1]:.1f})"
            )
            print(f"      aligned shape={aligned.shape} mean={aligned.mean():.1f}")
            if save_dir is not None:
                out_png = save_dir / f"{path.stem}_aligned.png"
                Image.fromarray(aligned).save(out_png)
                print(f"      -> saved {out_png}")
            faces.append((path, aligned))
        else:
            rgb = load_face(path)
            faces.append((path, rgb))
            print(f"   loaded {path}: shape={rgb.shape} dtype={rgb.dtype} mean={rgb.mean():.1f}")

    if not faces:
        print("# No usable faces; aborting")
        return 3
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
            "mode": "detect-and-align" if args.detect_and_align else "center-crop",
            "images": [str(p) for p, _ in faces],
            "skipped": [{"path": str(p), "info": _serialise_info(i)} for p, i in skipped],
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


def _serialise_info(info: Dict) -> Dict:
    """JSON-safe view of detect_and_align_face's info dict."""
    out: Dict = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = {kk: list(vv) for kk, vv in v.items()}
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
