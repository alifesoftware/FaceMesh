package com.alifesoftware.facemesh.ml

import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig
import com.alifesoftware.facemesh.config.PipelineConfig.Filters.SizeBandMode
import kotlin.math.abs

/**
 * Applies SPEC §6.2 false-positive heuristics on top of the BlazeFace candidate set.
 *
 *   1. Confidence threshold (already applied by [BlazeFaceDecoder] but re-applied here so
 *      callers can tune).
 *   2. Geometric sanity: eye-to-eye distance vs box width must look human.
 *   3. Size-outlier: dispatched on [SizeBandMode]:
 *      - [SizeBandMode.DISABLED]              -> stage skipped (production default).
 *      - [SizeBandMode.EXTREME_OUTLIERS_ONLY] -> drop faces with `width / median` outside
 *        `[extremeOutlierMinRatio, extremeOutlierMaxRatio]`.
 *      - [SizeBandMode.SYMMETRIC_BAND]        -> drop faces with `|width - median| >
 *        sizeBandFraction * median`.
 *      All three modes are no-ops for n <= 1 (no median) and for n >= [groupPhotoSizeBandSkipThreshold]
 *      (group-photo signal).
 */
object FaceFilters {

    private const val TAG: String = "FaceMesh.Filters"

    fun apply(
        faces: List<DetectedFace>,
        confidenceThreshold: Float = PipelineConfig.Filters.confidenceThreshold,
        eyeWidthRatioMin: Float = PipelineConfig.Filters.eyeWidthRatioMin,
        eyeWidthRatioMax: Float = PipelineConfig.Filters.eyeWidthRatioMax,
        sizeBandMode: SizeBandMode = PipelineConfig.Filters.sizeBandMode,
        sizeBandFraction: Float = PipelineConfig.Filters.sizeBandFraction,
        extremeOutlierMinRatio: Float = PipelineConfig.Filters.extremeOutlierMinRatio,
        extremeOutlierMaxRatio: Float = PipelineConfig.Filters.extremeOutlierMaxRatio,
        groupPhotoSizeBandSkipThreshold: Int = PipelineConfig.Filters.groupPhotoSizeBandSkipThreshold,
    ): List<DetectedFace> {
        Log.i(
            TAG,
            "apply: input=${faces.size} confTh=$confidenceThreshold eyeRatio=[$eyeWidthRatioMin..$eyeWidthRatioMax] " +
                "sizeBandMode=$sizeBandMode sizeBandFraction=$sizeBandFraction " +
                "extremeOutlierRatio=[$extremeOutlierMinRatio..$extremeOutlierMaxRatio] " +
                "groupSkip>=$groupPhotoSizeBandSkipThreshold",
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

        // ----- Size-outlier stage dispatch -----
        if (sizeBandMode == SizeBandMode.DISABLED) {
            Log.i(TAG, "apply: size-band stage DISABLED; passing ${byGeometry.size} face(s) through")
            return byGeometry
        }
        if (byGeometry.size <= 1) {
            Log.i(TAG, "apply: skipping size-band stage (n<=1; no median); early return ${byGeometry.size}")
            return byGeometry
        }
        if (byGeometry.size >= groupPhotoSizeBandSkipThreshold) {
            Log.i(
                TAG,
                "apply: skipping size-band stage (n=${byGeometry.size} >= " +
                    "groupPhotoSizeBandSkipThreshold=$groupPhotoSizeBandSkipThreshold); " +
                    "treating as group photo, early return ${byGeometry.size}",
            )
            return byGeometry
        }

        val widths = byGeometry.map { it.boundingBox.width() }.sorted()
        val median = widths[widths.size / 2]
        return when (sizeBandMode) {
            SizeBandMode.SYMMETRIC_BAND ->
                applySymmetricBand(byGeometry, median, sizeBandFraction)
            SizeBandMode.EXTREME_OUTLIERS_ONLY ->
                applyExtremeOutliersOnly(byGeometry, median, extremeOutlierMinRatio, extremeOutlierMaxRatio)
            // Already returned above for DISABLED; this branch is unreachable but keeps the
            // `when` exhaustive without a fall-through `else`.
            SizeBandMode.DISABLED -> byGeometry
        }
    }

    private fun applySymmetricBand(
        faces: List<DetectedFace>,
        median: Float,
        sizeBandFraction: Float,
    ): List<DetectedFace> {
        val band = median * sizeBandFraction
        Log.i(
            TAG,
            "apply: SYMMETRIC_BAND widths=${faces.map { "%.1f".format(it.boundingBox.width()) }} " +
                "median=$median band=±$band",
        )
        val kept = ArrayList<DetectedFace>(faces.size)
        faces.forEachIndexed { i, f ->
            val w = f.boundingBox.width()
            val deviation = abs(w - median)
            if (deviation <= band) {
                Log.i(
                    TAG,
                    "apply: face[$i] PASS SYMMETRIC_BAND w=${"%.1f".format(w)} dev=${"%.1f".format(deviation)} <= $band",
                )
                kept += f
            } else {
                Log.i(
                    TAG,
                    "apply: face[$i] DROP SYMMETRIC_BAND w=${"%.1f".format(w)} dev=${"%.1f".format(deviation)} > $band " +
                        "(median=${"%.1f".format(median)})",
                )
            }
        }
        Log.i(
            TAG,
            "apply: SYMMETRIC_BAND median=$median band=±$band afterSizeBand=${kept.size} " +
                "(dropped ${faces.size - kept.size})",
        )
        return kept
    }

    private fun applyExtremeOutliersOnly(
        faces: List<DetectedFace>,
        median: Float,
        minRatio: Float,
        maxRatio: Float,
    ): List<DetectedFace> {
        val minWidth = median * minRatio
        val maxWidth = median * maxRatio
        Log.i(
            TAG,
            "apply: EXTREME_OUTLIERS_ONLY widths=${faces.map { "%.1f".format(it.boundingBox.width()) }} " +
                "median=$median acceptable=[${"%.1f".format(minWidth)}..${"%.1f".format(maxWidth)}]",
        )
        val kept = ArrayList<DetectedFace>(faces.size)
        faces.forEachIndexed { i, f ->
            val w = f.boundingBox.width()
            if (w < minWidth) {
                Log.i(
                    TAG,
                    "apply: face[$i] DROP EXTREME_OUTLIERS_ONLY w=${"%.1f".format(w)} < " +
                        "minWidth=${"%.1f".format(minWidth)} (=${"%.2f".format(minRatio)} * median=$median)",
                )
            } else if (w > maxWidth) {
                Log.i(
                    TAG,
                    "apply: face[$i] DROP EXTREME_OUTLIERS_ONLY w=${"%.1f".format(w)} > " +
                        "maxWidth=${"%.1f".format(maxWidth)} (=${"%.2f".format(maxRatio)} * median=$median)",
                )
            } else {
                Log.i(
                    TAG,
                    "apply: face[$i] PASS EXTREME_OUTLIERS_ONLY w=${"%.1f".format(w)} " +
                        "in [${"%.1f".format(minWidth)}..${"%.1f".format(maxWidth)}]",
                )
                kept += f
            }
        }
        Log.i(
            TAG,
            "apply: EXTREME_OUTLIERS_ONLY median=$median range=[${"%.1f".format(minWidth)}.." +
                "${"%.1f".format(maxWidth)}] afterSizeBand=${kept.size} (dropped ${faces.size - kept.size})",
        )
        return kept
    }
}
