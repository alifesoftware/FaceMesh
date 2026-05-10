package com.alifesoftware.facemesh.ml

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

    fun apply(
        faces: List<DetectedFace>,
        confidenceThreshold: Float = 0.75f,
        eyeWidthRatioMin: Float = 0.25f,
        eyeWidthRatioMax: Float = 0.65f,
        sizeBandFraction: Float = 1.0f, // \u00b1 100% \u2192 \u00b1 2sigma typical; tune empirically
    ): List<DetectedFace> {
        val byScore = faces.filter { it.score >= confidenceThreshold }
        val byGeometry = byScore.filter { it.passesGeometry(eyeWidthRatioMin, eyeWidthRatioMax) }
        if (byGeometry.size <= 1) return byGeometry

        val widths = byGeometry.map { it.boundingBox.width() }.sorted()
        val median = widths[widths.size / 2]
        val band = median * sizeBandFraction
        return byGeometry.filter { abs(it.boundingBox.width() - median) <= band }
    }

    private fun DetectedFace.passesGeometry(min: Float, max: Float): Boolean {
        val w = boundingBox.width()
        if (w <= 1f) return false
        val ratio = landmarks.eyeDistance() / w
        return ratio in min..max
    }
}
