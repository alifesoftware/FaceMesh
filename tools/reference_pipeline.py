#!/usr/bin/env python
"""
End-to-end FaceMesh pipeline runner for offline experimentation on macOS.

Mirrors the on-device clusterify pipeline exactly:

    one or more folders of source photos
        -> per image: EXIF transpose + downsample (max 1280 long edge)
        -> BlazeFace detection (short-range OR full-range)
        -> per face: ArcFace 4-point similarity warp to 112x112
        -> GhostFaceNet FP16 embedding -> L2 normalise
        -> DBSCAN over cosine distance with eps + minPts (defaults match
           PipelineConfig.Clustering)
        -> per-cluster summary, with ground-truth folder labels for evaluation

Defaults match the production Android app as of commits c3a3267 (full-range
detector default) + f0445a4 (similarity-transform alignment) so the script's
output should reproduce on-device clustering decisions for any photo set.
Use this to:

  - Reproduce a clustering bug seen on-device, with full visibility.
  - Sweep eps / minPts to find the right thresholds for a given photo
    library before changing the production defaults.
  - Compare detector variants (short vs full range) on the same images.
  - Compare alignment modes (similarity vs perspective) on the same images.

Usage:
    source .venv/bin/activate
    python tools/reference_pipeline.py --folder ~/photos/me --folder ~/photos/wife
    python tools/reference_pipeline.py --folder ~/Desktop/test_photos --eps 0.45 --out /tmp/run

    python tools/reference_pipeline.py --folder ./me --folder ./spouse --dedupe-by-content --out ./run

Tuning DBSCAN (optional `--eps`; default matches the app):

    Cosine *distance* is d = 1 - cos_similarity between L2-normalised embeddings.
    Larger ``--eps`` merges more aggressively (fewer clusters / less noise).

    Typical sweep when results feel too fragmented: 0.35, 0.40, 0.45, 0.50, 0.55.

Optional: pass `--label me` etc. alongside each `--folder` to attach a custom
ground-truth label; default label is the folder's basename.

When evaluating clustering quality on camera rolls mirrored into two dirs
(me vs spouse), WhatsApp/Google Photos often duplicates the exact same bytes
into both trees. Same pixels → identical embeddings → extra ``minPts=2`` blobs
(pairwise cosine distance 0) that are not extra identities.

Use ``--dedupe-by-content`` to keep the first path encountered for each
SHA-256 (``--folder`` order decides which copy is canonical).

The script does NOT modify the source photos. With `--out DIR` it writes an
output bundle: per-cluster subdirectories of *symlinks* to the source photos,
all aligned 112x112 PNGs, and a `report.json` with the full run metadata
(detector variant, eps, every face's source path, embedding dim, cosine
distances within each cluster, etc.).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Silence TF's noisy banner before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Heavy imports come from the existing reference_embed.py module so this script
# stays a thin orchestration layer over the same building blocks. We import
# under-prefixed names where needed; Python's "private" convention isn't
# enforced and the alternative (extracting a shared lib) would be a lot of
# churn for the same outcome.
import reference_embed as re_mod  # noqa: E402

from reference_embed import (  # noqa: E402
    ALIGNMENT_MODE_SIMILARITY,
    ALIGNMENT_MODES,
    BLAZEFACE_DEFAULT_SCORE_THRESHOLD,
    DETECTOR_VARIANT_FULL_RANGE,
    DETECTOR_VARIANT_SHORT_RANGE,
    DETECTOR_VARIANTS,
    EMBEDDING_DIM,
    Embedder,
    build_onnx_embedder,
    build_tflite_embedder,
    cos,
    detect_and_align_all_faces,
    l2,
    to_input,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# DBSCAN defaults mirror PipelineConfig.Clustering on the Kotlin side.
DBSCAN_DEFAULT_EPS: float = 0.50
DBSCAN_DEFAULT_MIN_PTS: int = 2

# Match phase default (PipelineConfig.Match.defaultThreshold). Used only in the
# optional "filter against centroids" pass at the end of the pipeline.
MATCH_DEFAULT_THRESHOLD: float = 0.65

# Image extensions we'll consider when walking a folder.
IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FaceRecord:
    """A single detected face's full provenance + embedding."""
    source_path: Path
    folder_label: str
    face_index_in_image: int
    bbox: Tuple[float, float, float, float]
    score: float
    embedding: np.ndarray         # raw, 512-d (matches GhostFaceNet output)
    embedding_l2: np.ndarray      # L2-normalised (matches what Android stores)


@dataclass
class ClusterAssignment:
    cluster_id: int               # 0..N-1; -1 = noise
    members: List[int] = field(default_factory=list)  # indices into face_records


# ---------------------------------------------------------------------------
# Folder walk
# ---------------------------------------------------------------------------

def discover_images(folder: Path) -> List[Path]:
    """Recursively list image files under `folder`, sorted for determinism."""
    if not folder.is_dir():
        raise SystemExit(f"--folder {folder} is not a directory")
    out: List[Path] = []
    for ext in IMAGE_EXTS:
        out.extend(folder.rglob(f"*{ext}"))
        out.extend(folder.rglob(f"*{ext.upper()}"))
    # rglob can yield duplicates on case-insensitive filesystems; dedupe.
    seen: set = set()
    dedup: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(p)
    dedup.sort()
    return dedup


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Full-file SHA-256 for byte-identical duplicate detection."""
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def flatten_image_jobs(folder_paths: List[Path], folder_labels: List[str]) -> List[Tuple[Path, str]]:
    """Expand folders to (path, label) pairs: folder walking order × sorted paths."""
    items: List[Tuple[Path, str]] = []
    for folder, label in zip(folder_paths, folder_labels):
        for path in discover_images(folder):
            items.append((path, label))
    return items


def dedupe_image_jobs_by_content(
    items: Sequence[Tuple[Path, str]],
) -> Tuple[List[Tuple[Path, str]], Dict[str, Any]]:
    """Drop later copies whose file bytes hash the same as an earlier kept path.

    Preserves traversal order from ``items``: the first occurrence of each
    digest wins (so ``--folder A --folder B`` keeps A's copy when both hold
    identical ``PXL_….jpg``).
    Returns (unique_work_items, report_dict for JSON / logs).
    """
    seen_digest: Dict[str, Tuple[Path, str]] = {}
    unique: List[Tuple[Path, str]] = []
    skipped_rows: List[Dict[str, str]] = []
    for path, label in items:
        digest = sha256_file(path)
        if digest in seen_digest:
            canon_path, canon_label = seen_digest[digest]
            skipped_rows.append({
                "sha256": digest,
                "skipped_path": str(path.resolve()),
                "skipped_label": label,
                "kept_path": str(canon_path.resolve()),
                "kept_label": canon_label,
            })
            continue
        seen_digest[digest] = (path, label)
        unique.append((path, label))
    report: Dict[str, Any] = {
        "enabled": True,
        "paths_before": len(items),
        "paths_after": len(unique),
        "skipped_count": len(skipped_rows),
        "skipped": skipped_rows,
    }
    return unique, report


# ---------------------------------------------------------------------------
# DBSCAN -- direct port of app/src/.../ml/cluster/Dbscan.kt
# ---------------------------------------------------------------------------

UNVISITED: int = -2
NOISE: int = -1


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - dot for L2-normalised inputs.

    Mirrors EmbeddingMath.cosineDistance: clamps to [0, 2] for safety.
    """
    return float(np.clip(1.0 - float(np.dot(a, b)), 0.0, 2.0))


def dbscan(
    points: List[np.ndarray],
    eps: float,
    min_pts: int,
) -> List[int]:
    """Density-based clustering over cosine distance.

    Returns a per-point label list: 0..N-1 for cluster ids, NOISE (-1) for
    outliers. Algorithm mirrors `Dbscan.run` in the Kotlin app: BFS expansion
    with the standard noise-promotion-to-border-point rule.
    """
    n = len(points)
    labels = [UNVISITED] * n
    if n == 0:
        return labels

    def region_query(idx: int) -> List[int]:
        origin = points[idx]
        return [j for j in range(n) if cosine_distance(origin, points[j]) <= eps]

    cluster_id = 0
    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        neighbours = region_query(i)
        if len(neighbours) < min_pts:
            labels[i] = NOISE
            continue
        labels[i] = cluster_id
        seeds = list(neighbours)
        while seeds:
            q = seeds.pop(0)
            if q == i:
                continue
            if labels[q] == NOISE:
                labels[q] = cluster_id
            if labels[q] != UNVISITED:
                continue
            labels[q] = cluster_id
            q_neighbours = region_query(q)
            if len(q_neighbours) >= min_pts:
                seeds.extend(q_neighbours)
        cluster_id += 1
    return labels


def mean_and_normalize(vectors: List[np.ndarray]) -> np.ndarray:
    """Component-wise mean of L2-normed vectors, then L2-normalise. Mirrors
    `EmbeddingMath.meanAndNormalize` -- this is how Android computes a
    cluster centroid for the match phase."""
    if not vectors:
        raise ValueError("cannot average empty list")
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    return l2(mean)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def build_embedder(name: str, fp32_path: Path, fp16_path: Path, w8a8_path: Path) -> Embedder:
    """Build the requested embedder (matches reference_embed.py's choice set)."""
    name = name.lower()
    if name in ("fp32", "onnx", "fp32_onnx"):
        return build_onnx_embedder(fp32_path)
    if name in ("fp16", "fp16_tflite"):
        return build_tflite_embedder(fp16_path, "FP16_TFLITE")
    if name in ("w8a8", "int8", "w8a8_tflite"):
        return build_tflite_embedder(w8a8_path, "W8A8_TFLITE")
    raise ValueError(f"unknown embedder {name!r}; use one of fp32 / fp16 / w8a8")


def run_pipeline(
    work_items: Sequence[Tuple[Path, str]],
    all_folder_labels: Sequence[str],
    detector_path: Path,
    embedder: Embedder,
    detector_variant: str,
    alignment_mode: str,
    score_threshold: float,
    channel_order: str,
    eps: float,
    min_pts: int,
    show_progress: bool = True,
) -> Tuple[List[FaceRecord], Dict[str, Any]]:
    """Run detect+align+embed for each (path, folder_label) work item.

    ``work_items`` is usually built from ``flatten_image_jobs`` and optionally
    ``dedupe_image_jobs_by_content``. Pure data extraction: no clustering yet.
    """
    # Build BlazeFace interpreter once.
    interp = re_mod._build_blazeface_interp(detector_path)  # noqa: SLF001

    records: List[FaceRecord] = []
    per_folder_stats: Dict[str, Dict[str, int]] = {}
    for lbl in all_folder_labels:
        per_folder_stats.setdefault(lbl, {"images": 0, "faces": 0, "zero_face_images": 0})

    total_images = len(work_items)
    for path, label in work_items:
        per_folder_stats[label]["images"] += 1

    print(f"# {total_images} image(s) in work queue across "
          f"{len(set(all_folder_labels))} folder label(s); detector={detector_variant} "
          f"alignment={alignment_mode} eps={eps} minPts={min_pts}")
    print()

    started_at = time.time()
    for image_counter, (path, label) in enumerate(work_items, start=1):
        t0 = time.time()
        try:
            aligned_list, info = detect_and_align_all_faces(
                path,
                interp,
                score_threshold=score_threshold,
                alignment_mode=alignment_mode,
                detector_variant=detector_variant,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"   ! [{image_counter}/{total_images}] {path.name} ERROR: {exc}")
            continue
        n = len(aligned_list)
        if n == 0:
            per_folder_stats[label]["zero_face_images"] += 1
            if show_progress:
                print(f"   . [{image_counter}/{total_images}] {label}/{path.name} "
                      f"-> 0 faces ({(time.time() - t0) * 1000:.0f}ms)")
            continue
        for face_idx, (aligned, face_info) in enumerate(zip(aligned_list, info["faces"])):
            nhwc = to_input(aligned, channel_order)
            v = embedder.runner(nhwc).reshape(-1).astype(np.float32)
            assert v.shape == (EMBEDDING_DIM,), v.shape
            records.append(FaceRecord(
                source_path=path,
                folder_label=label,
                face_index_in_image=face_idx,
                bbox=tuple(face_info["bbox"]),
                score=float(face_info["score"]),
                embedding=v,
                embedding_l2=l2(v),
            ))
            per_folder_stats[label]["faces"] += 1
        if show_progress:
            print(f"   + [{image_counter}/{total_images}] {label}/{path.name} "
                  f"-> {n} face(s) ({(time.time() - t0) * 1000:.0f}ms)")

    elapsed = time.time() - started_at
    stats = {
        "elapsed_sec": round(elapsed, 2),
        "per_folder": per_folder_stats,
        "total_faces": len(records),
        "total_images": total_images,
        "detector_variant": detector_variant,
        "alignment_mode": alignment_mode,
        "channel_order": channel_order,
        "score_threshold": score_threshold,
    }
    print()
    print(f"# Detection summary ({elapsed:.2f}s):")
    for label, s in per_folder_stats.items():
        non_zero = s["images"] - s["zero_face_images"]
        rate = (non_zero / s["images"] * 100) if s["images"] else 0.0
        avg_faces_per_hit = (s["faces"] / non_zero) if non_zero else 0.0
        print(f"   {label:<24} images={s['images']:>4}  faces={s['faces']:>4}  "
              f"zero-face-images={s['zero_face_images']:>3} ({rate:.0f}% hit)  "
              f"avg-faces-per-hit={avg_faces_per_hit:.2f}")
    print(f"   {'TOTAL':<24} images={total_images:>4}  faces={len(records):>4}")
    print()
    return records, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_clustering_report(
    records: List[FaceRecord],
    labels: List[int],
    eps: float,
    min_pts: int,
) -> Dict[str, Any]:
    """Print + return a structured report grouping faces by DBSCAN cluster id."""
    n_clusters = max(labels) + 1 if any(label >= 0 for label in labels) else 0
    cluster_to_indices: Dict[int, List[int]] = {cid: [] for cid in range(n_clusters)}
    noise_indices: List[int] = []
    for i, label in enumerate(labels):
        if label == NOISE:
            noise_indices.append(i)
        else:
            cluster_to_indices[label].append(i)

    print(f"# DBSCAN -> {n_clusters} cluster(s), {len(noise_indices)} noise (eps={eps} minPts={min_pts})")
    print()

    cluster_payload: List[Dict[str, Any]] = []
    for cid in sorted(cluster_to_indices.keys()):
        members = cluster_to_indices[cid]
        if not members:
            continue
        # Per-folder counts inside this cluster (this is the "ground-truth"
        # signal: a cluster of "you" should be majority-from-the-me-folder).
        folder_counts: Dict[str, int] = {}
        for idx in members:
            folder_counts[records[idx].folder_label] = folder_counts.get(records[idx].folder_label, 0) + 1
        # Tightness: mean off-diagonal cosine similarity within the cluster.
        sims: List[float] = []
        for i_a, idx_a in enumerate(members):
            for idx_b in members[i_a + 1:]:
                sims.append(cos(records[idx_a].embedding_l2, records[idx_b].embedding_l2))
        mean_sim = float(np.mean(sims)) if sims else 1.0
        min_sim = float(np.min(sims)) if sims else 1.0
        max_sim = float(np.max(sims)) if sims else 1.0
        print(f"   cluster_{cid:<2}  size={len(members):>3}  by-folder=" +
              ", ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(folder_counts.items(), key=lambda kv: -kv[1])) +
              f"   sim[mean/min/max]={mean_sim:.3f}/{min_sim:.3f}/{max_sim:.3f}")
        for idx in members:
            r = records[idx]
            print(f"      - {r.folder_label}/{r.source_path.name} "
                  f"face_idx={r.face_index_in_image} score={r.score:.2f}")
        cluster_payload.append({
            "cluster_id": cid,
            "size": len(members),
            "by_folder": folder_counts,
            "similarity": {"mean": mean_sim, "min": min_sim, "max": max_sim},
            "members": [
                {
                    "source_path": str(records[idx].source_path),
                    "folder_label": records[idx].folder_label,
                    "face_index_in_image": records[idx].face_index_in_image,
                    "bbox": list(records[idx].bbox),
                    "score": records[idx].score,
                }
                for idx in members
            ],
        })

    if noise_indices:
        print()
        print(f"   noise  ({len(noise_indices)} singleton face(s) -- did not cluster with anyone)")
        # Group by folder for the "are these faces I expected to cluster?" answer.
        noise_by_folder: Dict[str, List[int]] = {}
        for idx in noise_indices:
            noise_by_folder.setdefault(records[idx].folder_label, []).append(idx)
        for lbl, idxs in sorted(noise_by_folder.items()):
            print(f"      [{lbl}] {len(idxs)} face(s):")
            for idx in idxs:
                r = records[idx]
                print(f"         - {r.source_path.name} face_idx={r.face_index_in_image} "
                      f"score={r.score:.2f}")

    print()
    return {
        "n_clusters": n_clusters,
        "n_noise": len(noise_indices),
        "clusters": cluster_payload,
        "noise_indices": [
            {
                "source_path": str(records[idx].source_path),
                "folder_label": records[idx].folder_label,
                "face_index_in_image": records[idx].face_index_in_image,
                "score": records[idx].score,
            }
            for idx in noise_indices
        ],
    }


def write_output_bundle(
    out_dir: Path,
    records: List[FaceRecord],
    labels: List[int],
    aligned_thumbs: List[Optional[np.ndarray]],
    cluster_report: Dict[str, Any],
    run_stats: Dict[str, Any],
) -> None:
    """Materialise the run as a directory of symlinks + aligned thumbnails + JSON."""
    from PIL import Image  # local import: only need this when --out is set
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    for i, (rec, aligned) in enumerate(zip(records, aligned_thumbs)):
        if aligned is None:
            continue
        # Filename: "0034_<folder>_<source-stem>_face<n>.png"
        name = f"{i:04d}_{rec.folder_label}_{rec.source_path.stem}_face{rec.face_index_in_image}.png"
        Image.fromarray(aligned).save(aligned_dir / name)

    # Per-cluster directories of symlinks back to the original source photo.
    n_clusters = max(labels) + 1 if any(label >= 0 for label in labels) else 0
    for cid in range(n_clusters):
        cdir = out_dir / f"cluster_{cid:02d}"
        cdir.mkdir(exist_ok=True)
        for i, label in enumerate(labels):
            if label != cid:
                continue
            rec = records[i]
            link = cdir / f"{rec.folder_label}_{rec.source_path.name}"
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(rec.source_path.resolve())
            except OSError:
                # Fall back to copy if the FS doesn't support symlinks.
                import shutil
                shutil.copy2(rec.source_path, link)
    noise_dir = out_dir / "noise"
    noise_dir.mkdir(exist_ok=True)
    for i, label in enumerate(labels):
        if label != NOISE:
            continue
        rec = records[i]
        link = noise_dir / f"{rec.folder_label}_{rec.source_path.name}"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(rec.source_path.resolve())
        except OSError:
            import shutil
            shutil.copy2(rec.source_path, link)

    # Machine-readable JSON dump.
    report_path = out_dir / "report.json"
    payload: Dict[str, Any] = {
        "run": run_stats,
        "clustering": cluster_report,
        "records": [
            {
                "index": i,
                "source_path": str(rec.source_path),
                "folder_label": rec.folder_label,
                "face_index_in_image": rec.face_index_in_image,
                "bbox": list(rec.bbox),
                "score": rec.score,
                "label": labels[i],
                "embedding_first8": rec.embedding_l2[:8].tolist(),
            }
            for i, rec in enumerate(records)
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"# Output bundle written to {out_dir}")
    print(f"   report.json     ({report_path.stat().st_size:,} bytes)")
    print(f"   aligned/        ({len(records)} aligned 112x112 PNG(s))")
    print(f"   cluster_NN/     ({n_clusters} cluster dir(s) of symlinks)")
    print(f"   noise/          ({sum(1 for l in labels if l == NOISE)} noise symlink(s))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLUSTERING_TUNING_EPILOG = """
DBSCAN uses cosine distance d = 1 - cos_similarity between L2-normalised
embeddings (same as Android). Larger --eps merges neighbours more aggressively.

Suggested values to compare on the SAME inputs (reuse --out dirs or --quiet):

  Strict (more clusters / more noise — split identities easier):
      --eps 0.28  --eps 0.30  --eps 0.35

  Around app default — matches PipelineClustering.defaultEps / manifest baseline:
      --eps 0.50

  Tighter-than-default diagnostics:
      --eps 0.38  --eps 0.40  --eps 0.42

  Looser — if still noisy at 0.50; watch for relative merges:
      --eps 0.55  --eps 0.60

  --min-pts tweaks (default %(min)d):
      --min-pts 2   two faces suffice to seed a cluster (current app default).
      --min-pts 3   drop tiny accidental pairs; tends to raise noise singletons.

Run the same folders with different --eps; compare stdout cluster counts plus
noise/ symlink counts under --out, and skim aligned/*.png inside each cluster.
""".strip() % {"min": DBSCAN_DEFAULT_MIN_PTS}


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLUSTERING_TUNING_EPILOG,
    )
    p.add_argument(
        "--folder",
        action="append",
        type=Path,
        required=True,
        help="A folder of source photos. Pass multiple --folder flags for multi-class "
             "ground-truth labelling. Ground-truth label defaults to the folder basename; "
             "override with --label.",
    )
    p.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional ground-truth label per --folder, in the same order. If omitted the "
             "folder basename is used. Pass once per folder.",
    )
    p.add_argument(
        "--detector-variant",
        choices=DETECTOR_VARIANTS,
        default=DETECTOR_VARIANT_FULL_RANGE,
        help="BlazeFace variant. Default mirrors the on-device default (full_range as of the "
             "May 2026 PipelineConfig flip).",
    )
    p.add_argument(
        "--blazeface-tflite",
        type=Path,
        default=None,
        help="Override path to the BlazeFace .tflite. Defaults to the matching variant under "
             "models/ (face_detection_short_range.tflite or face_detection_full_range.tflite).",
    )
    p.add_argument(
        "--blazeface-score-threshold",
        type=float,
        default=BLAZEFACE_DEFAULT_SCORE_THRESHOLD,
        help=f"BlazeFace sigmoid score threshold. Default {BLAZEFACE_DEFAULT_SCORE_THRESHOLD}.",
    )
    p.add_argument(
        "--alignment-mode",
        choices=ALIGNMENT_MODES,
        default=ALIGNMENT_MODE_SIMILARITY,
        help="ArcFace warp model. Default 'similarity' matches FaceAligner on the device.",
    )
    p.add_argument(
        "--embedder",
        choices=("fp32", "fp16", "w8a8"),
        default="fp16",
        help="GhostFaceNet variant. Default fp16 mirrors the .tflite shipped on-device.",
    )
    p.add_argument(
        "--channel-order",
        choices=("RGB", "BGR"),
        default="RGB",
        help="Pixel channel order fed to the embedder. RGB matches the on-device pipeline.",
    )
    p.add_argument(
        "--fp32-onnx",
        type=Path,
        default=MODELS_DIR / "ghostface_fp32.onnx",
    )
    p.add_argument(
        "--fp16-tflite",
        type=Path,
        default=MODELS_DIR / "ghostface_fp16.tflite",
    )
    p.add_argument(
        "--w8a8-tflite",
        type=Path,
        default=MODELS_DIR / "ghostface_w8a8.tflite",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=DBSCAN_DEFAULT_EPS,
        metavar="D",
        help=(
            "DBSCAN neighbourhood radius as cosine DISTANCE (= 1 - cosine similarity "
            "for unit vectors). Default %(default)g matches PipelineConfig.Clustering "
            "and the in-app slider default; see epilog below for values to sweep."
        ),
    )
    p.add_argument(
        "--min-pts",
        type=int,
        default=DBSCAN_DEFAULT_MIN_PTS,
        metavar="N",
        help=(
            "DBSCAN minPts (minimum neighbours to seed/expand a cluster). "
            "Default %(default)s matches PipelineConfig; try 3 if tiny spurious pairs appear."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory: writes per-cluster symlink dirs + aligned PNGs + report.json",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image progress lines.",
    )
    p.add_argument(
        "--dedupe-by-content",
        action="store_true",
        help="SHA-256 all inputs and skip later copies identical to an earlier file. "
             "Keeps first path in traversal order (--folder flags first→last, sorted paths "
             "within each folder). Removes artificial pairwise clusters when the same byte "
             "exact image lives in multiple trees; skips re-reading bytes on each run besides "
             "this pre-pass.",
    )
    args = p.parse_args(argv)

    # Resolve labels.
    folder_labels: List[str]
    if args.label:
        if len(args.label) != len(args.folder):
            raise SystemExit(
                f"--label provided {len(args.label)} time(s) but --folder provided "
                f"{len(args.folder)} time(s); they must match in count and order"
            )
        folder_labels = list(args.label)
    else:
        folder_labels = [f.name for f in args.folder]

    # Resolve detector path.
    detector_path = args.blazeface_tflite
    if detector_path is None:
        detector_path = MODELS_DIR / (
            "face_detection_full_range.tflite"
            if args.detector_variant == DETECTOR_VARIANT_FULL_RANGE
            else "face_detection_short_range.tflite"
        )
    if not detector_path.exists():
        raise SystemExit(f"BlazeFace model not found at {detector_path}")

    # Build embedder.
    print(f"# Building {args.embedder.upper()} embedder")
    embedder = build_embedder(args.embedder, args.fp32_onnx, args.fp16_tflite, args.w8a8_tflite)
    print(f"   built {embedder.name}")

    work_items = flatten_image_jobs(args.folder, folder_labels)
    dedupe_report: Optional[Dict[str, Any]] = None
    if args.dedupe_by_content:
        t_dup = time.time()
        work_items, dedupe_report = dedupe_image_jobs_by_content(work_items)
        dt_dup = time.time() - t_dup
        print(
            f"# Dedupe-by-content (SHA-256): {dedupe_report['paths_before']} paths -> "
            f"{dedupe_report['paths_after']} unique ({dedupe_report['skipped_count']} "
            f"skipped) in {dt_dup:.2f}s"
        )
        if dedupe_report["skipped_count"] and not args.quiet:
            preview = dedupe_report["skipped"][:5]
            for row in preview:
                print(f"      skip [{row['skipped_label']}]{Path(row['skipped_path']).name} "
                      f"-> keep [{row['kept_label']}]{Path(row['kept_path']).name}")
            if dedupe_report["skipped_count"] > len(preview):
                print(f"      ... ({dedupe_report['skipped_count'] - len(preview)} more in report.json)")
        print()

    # Run detection + embedding.
    records, run_stats = run_pipeline(
        work_items,
        folder_labels,
        detector_path=detector_path,
        embedder=embedder,
        detector_variant=args.detector_variant,
        alignment_mode=args.alignment_mode,
        score_threshold=args.blazeface_score_threshold,
        channel_order=args.channel_order,
        eps=args.eps,
        min_pts=args.min_pts,
        show_progress=not args.quiet,
    )
    if dedupe_report is not None:
        run_stats["dedupe_by_content"] = dedupe_report

    if not records:
        print("# No faces detected; aborting before clustering")
        return 0

    # Cluster.
    print(f"# DBSCAN over {len(records)} embedding(s) (eps={args.eps} minPts={args.min_pts})")
    t0 = time.time()
    labels = dbscan([r.embedding_l2 for r in records], eps=args.eps, min_pts=args.min_pts)
    print(f"   done in {(time.time() - t0) * 1000:.0f}ms")
    print()

    cluster_report = print_clustering_report(records, labels, args.eps, args.min_pts)

    # Optional output bundle.
    if args.out is not None:
        # Re-emit the aligned thumbnails so they can be saved alongside the clusters.
        # We do this here (vs. caching during run_pipeline) to keep the hot path lean
        # when --out isn't passed; the second pass is cheap on aligned 112x112 buffers
        # we'd otherwise have to hold in memory for the whole detection sweep.
        from PIL import Image, ImageOps  # noqa: F401  (used by re_mod indirectly)
        # We don't have the aligned bitmaps anymore (we threw them away after embedding).
        # Re-run alignment for each record so --out gives a consistent visual artifact set.
        # This costs ~10ms per face on a Pixel-class CPU; for typical N <= 200 faces that's
        # well under 5s and is a one-shot cost only when --out is requested.
        interp = re_mod._build_blazeface_interp(detector_path)  # noqa: SLF001
        aligned_thumbs: List[Optional[np.ndarray]] = []
        seen_paths: Dict[Path, List[np.ndarray]] = {}
        for rec in records:
            if rec.source_path not in seen_paths:
                aligned_list, _info = detect_and_align_all_faces(
                    rec.source_path,
                    interp,
                    score_threshold=args.blazeface_score_threshold,
                    alignment_mode=args.alignment_mode,
                    detector_variant=args.detector_variant,
                )
                seen_paths[rec.source_path] = aligned_list
            faces_for_path = seen_paths[rec.source_path]
            if rec.face_index_in_image < len(faces_for_path):
                aligned_thumbs.append(faces_for_path[rec.face_index_in_image])
            else:
                aligned_thumbs.append(None)
        write_output_bundle(args.out, records, labels, aligned_thumbs, cluster_report, run_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
