package com.alifesoftware.facemesh.ml

import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig
import kotlin.math.abs

/**
 * Applies SPEC \u00a76.2 false-positive heuristics on top of the BlazeFace candidate set.
 *
 *   1. Confidence threshold (already applied by [BlazeFaceDecoder] but re-applied here so
 *      callers can tune).
 *   2. Geometric sanity: eye-to-eye distance vs box width must look human.
 *   3. Size-outlier: keep faces within +/- sizeBandFraction of the median width when between
 *      two and (groupPhotoSizeBandSkipThreshold - 1) faces are present. Skipped entirely for
 *      group photos (n >= groupPhotoSizeBandSkipThreshold) where wide variance in face size is
 *      the norm and the median ceases to be a reliable "real face" anchor.
 */
object FaceFilters {

    private const val TAG: String = "FaceMesh.Filters"

    fun apply(
        faces: List<DetectedFace>,
        confidenceThreshold: Float = PipelineConfig.Filters.confidenceThreshold,
        eyeWidthRatioMin: Float = PipelineConfig.Filters.eyeWidthRatioMin,
        eyeWidthRatioMax: Float = PipelineConfig.Filters.eyeWidthRatioMax,
        sizeBandFraction: Float = PipelineConfig.Filters.sizeBandFraction,
        groupPhotoSizeBandSkipThreshold: Int = PipelineConfig.Filters.groupPhotoSizeBandSkipThreshold,
    ): List<DetectedFace> {
        Log.i(
            TAG,
            "apply: input=${faces.size} confTh=$confidenceThreshold eyeRatio=[$eyeWidthRatioMin..$eyeWidthRatioMax] " +
                "sizeBand=$sizeBandFraction groupSkip>=$groupPhotoSizeBandSkipThreshold",
        )
        if (faces.isEmpty()) {
            Log.i(TAG, "apply: empty input -> early return empty")
            return emptyList()
        }

        val byScore = ArrayList<DetectedFace>(faces.size)
        faces.forEachIndexed { i, f ->
            if (f.score >= confidenceThreshold) {
                byScore += f
            } else {
                Log.i(
                    TAG,
                    "apply: face[$i] DROP confidence score=${"%.3f".format(f.score)} < $confidenceThreshold " +
                        "bbox=${f.boundingBox}",
                )
            }
        }
        Log.i(TAG, "apply: afterConfidence=${byScore.size} (dropped ${faces.size - byScore.size})")

        val byGeometry = ArrayList<DetectedFace>(byScore.size)
        byScore.forEachIndexed { i, f ->
            val w = f.boundingBox.width()
            if (w <= 1f) {
                Log.i(TAG, "apply: face[$i] DROP geometry width<=1 (w=$w) bbox=${f.boundingBox}")
                return@forEachIndexed
            }
            val ratio = f.landmarks.eyeDistance() / w
            if (ratio !in eyeWidthRatioMin..eyeWidthRatioMax) {
                Log.i(
                    TAG,
                    "apply: face[$i] DROP geometry eyeDist/width=${"%.3f".format(ratio)} " +
                        "out of [$eyeWidthRatioMin..$eyeWidthRatioMax] " +
                        "(eyeDist=${"%.2f".format(f.landmarks.eyeDistance())} w=${"%.2f".format(w)})",
                )
                return@forEachIndexed
            }
            Log.i(
                TAG,
                "apply: face[$i] PASS geometry eyeDist/width=${"%.3f".format(ratio)} " +
                    "score=${"%.3f".format(f.score)}",
            )
            byGeometry += f
        }
        Log.i(TAG, "apply: afterGeometry=${byGeometry.size} (dropped ${byScore.size - byGeometry.size})")

        if (byGeometry.size <= 1) {
            Log.i(TAG, "apply: skipping size-band filter (n<=1); early return ${byGeometry.size}")
            return byGeometry
        }
        if (byGeometry.size >= groupPhotoSizeBandSkipThreshold) {
            Log.i(
                TAG,
                "apply: skipping size-band filter (n=${byGeometry.size} >= " +
                    "groupPhotoSizeBandSkipThreshold=$groupPhotoSizeBandSkipThreshold); " +
                    "treating as group photo, early return ${byGeometry.size}",
            )
            return byGeometry
        }

        val widths = byGeometry.map { it.boundingBox.width() }.sorted()
        val median = widths[widths.size / 2]
        val band = median * sizeBandFraction
        Log.i(
            TAG,
            "apply: size-band stage widths=${widths.map { "%.1f".format(it) }} median=$median " +
                "band=±$band",
        )
        val kept = ArrayList<DetectedFace>(byGeometry.size)
        byGeometry.forEachIndexed { i, f ->
            val w = f.boundingBox.width()
            val deviation = abs(w - median)
            if (deviation <= band) {
                Log.i(
                    TAG,
                    "apply: face[$i] PASS size-band w=${"%.1f".format(w)} dev=${"%.1f".format(deviation)} <= $band",
                )
                kept += f
            } else {
                Log.i(
                    TAG,
                    "apply: face[$i] DROP size-band w=${"%.1f".format(w)} dev=${"%.1f".format(deviation)} > $band " +
                        "(median=${"%.1f".format(median)})",
                )
            }
        }
        Log.i(
            TAG,
            "apply: medianWidth=$median band=±$band afterSizeBand=${kept.size} " +
                "(dropped ${byGeometry.size - kept.size})",
        )
        return kept
    }
}
