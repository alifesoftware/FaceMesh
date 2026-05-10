package com.alifesoftware.facemesh.ml

import android.util.Log
import kotlin.math.abs

/**
 * Applies SPEC \u00a76.2 false-positive heuristics on top of the BlazeFace candidate set.
 *
 *   1. Confidence threshold (already applied by [BlazeFaceDecoder] but re-applied here so
 *      callers can tune).
 *   2. Geometric sanity: eye-to-eye distance vs box width must look human.
 *   3. Size-outlier: keep faces within \u00b1 sizeBandFraction of the median width when more than
 *      one face is present.
 */
object FaceFilters {

    private const val TAG: String = "FaceMesh.Filters"

    fun apply(
        faces: List<DetectedFace>,
        confidenceThreshold: Float = 0.75f,
        eyeWidthRatioMin: Float = 0.25f,
        eyeWidthRatioMax: Float = 0.65f,
        sizeBandFraction: Float = 1.0f, // \u00b1 100% \u2192 \u00b1 2sigma typical; tune empirically
    ): List<DetectedFace> {
        Log.i(
            TAG,
            "apply: input=${faces.size} confTh=$confidenceThreshold eyeRatio=[$eyeWidthRatioMin..$eyeWidthRatioMax] " +
                "sizeBand=$sizeBandFraction",
        )
        val byScore = faces.filter { it.score >= confidenceThreshold }
        Log.i(TAG, "apply: afterConfidence=${byScore.size} (dropped ${faces.size - byScore.size})")
        val byGeometry = byScore.filter { it.passesGeometry(eyeWidthRatioMin, eyeWidthRatioMax) }
        Log.i(TAG, "apply: afterGeometry=${byGeometry.size} (dropped ${byScore.size - byGeometry.size})")
        if (byGeometry.size <= 1) {
            Log.i(TAG, "apply: skipping size-band filter (n<=1); returning ${byGeometry.size}")
            return byGeometry
        }

        val widths = byGeometry.map { it.boundingBox.width() }.sorted()
        val median = widths[widths.size / 2]
        val band = median * sizeBandFraction
        val kept = byGeometry.filter { abs(it.boundingBox.width() - median) <= band }
        Log.i(
            TAG,
            "apply: medianWidth=$median band=±$band afterSizeBand=${kept.size} " +
                "(dropped ${byGeometry.size - kept.size})",
        )
        return kept
    }

    private fun DetectedFace.passesGeometry(min: Float, max: Float): Boolean {
        val w = boundingBox.width()
        if (w <= 1f) return false
        val ratio = landmarks.eyeDistance() / w
        return ratio in min..max
    }
}
