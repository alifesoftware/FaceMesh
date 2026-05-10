package com.alifesoftware.facemesh.ml

import android.graphics.PointF
import android.graphics.RectF

/**
 * One face detected by BlazeFace inside a source bitmap.
 *
 * All coordinates are in **pixels** of the source bitmap (post-EXIF, post-downsample).
 */
data class DetectedFace(
    val boundingBox: RectF,
    val landmarks: FaceLandmarks,
    val score: Float,
)

/**
 * BlazeFace's six landmarks. Indices match the model's regressor output:
 *   0 = right eye, 1 = left eye, 2 = nose tip, 3 = mouth center,
 *   4 = right ear tragion, 5 = left ear tragion.
 *
 * "Right" / "Left" are from the subject's perspective (so right-eye is on the image's left side).
 */
data class FaceLandmarks(
    val rightEye: PointF,
    val leftEye: PointF,
    val noseTip: PointF,
    val mouthCenter: PointF,
    val rightEarTragion: PointF,
    val leftEarTragion: PointF,
) {
    fun eyeDistance(): Float {
        val dx = rightEye.x - leftEye.x
        val dy = rightEye.y - leftEye.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}
