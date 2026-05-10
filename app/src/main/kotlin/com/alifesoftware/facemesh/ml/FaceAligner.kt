package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log

/**
 * Affinely aligns a face crop into the canonical 112x112 pose used by GhostFaceNet
 * (SPEC \u00a76.3).
 *
 * Uses [Matrix.setPolyToPoly] with **4 source / 4 destination** points (the maximum the API
 * supports). The 4 anchor points are right-eye, left-eye, nose-tip, mouth-centre, mapped to a
 * canonical ArcFace-style template normalised to 112x112.
 */
class FaceAligner(
    private val outputSize: Int = OUTPUT_SIZE,
) {

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()

    fun align(source: Bitmap, face: DetectedFace): Bitmap {
        Log.i(
            TAG,
            "align: source=${source.width}x${source.height} -> ${outputSize}x${outputSize} " +
                "bbox=${face.boundingBox} eyeDist=${"%.2f".format(face.landmarks.eyeDistance())} " +
                "score=${"%.3f".format(face.score)}",
        )
        Log.i(
            TAG,
            "align: src landmarks " +
                "rightEye=(${"%.1f".format(face.landmarks.rightEye.x)},${"%.1f".format(face.landmarks.rightEye.y)}) " +
                "leftEye=(${"%.1f".format(face.landmarks.leftEye.x)},${"%.1f".format(face.landmarks.leftEye.y)}) " +
                "nose=(${"%.1f".format(face.landmarks.noseTip.x)},${"%.1f".format(face.landmarks.noseTip.y)}) " +
                "mouth=(${"%.1f".format(face.landmarks.mouthCenter.x)},${"%.1f".format(face.landmarks.mouthCenter.y)})",
        )
        Log.i(
            TAG,
            "align: dst canonical " +
                "rightEye=(${canonicalDestination[0]},${canonicalDestination[1]}) " +
                "leftEye=(${canonicalDestination[2]},${canonicalDestination[3]}) " +
                "nose=(${canonicalDestination[4]},${canonicalDestination[5]}) " +
                "mouth=(${canonicalDestination[6]},${canonicalDestination[7]})",
        )
        val src = floatArrayOf(
            face.landmarks.rightEye.x, face.landmarks.rightEye.y,
            face.landmarks.leftEye.x, face.landmarks.leftEye.y,
            face.landmarks.noseTip.x, face.landmarks.noseTip.y,
            face.landmarks.mouthCenter.x, face.landmarks.mouthCenter.y,
        )
        val dst = canonicalDestination
        matrix.reset()
        val mapped = matrix.setPolyToPoly(src, 0, dst, 0, 4)
        if (!mapped) {
            Log.w(
                TAG,
                "align: setPolyToPoly returned false (degenerate landmark set); proceeding with identity",
            )
        }

        val out = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawBitmap(source, matrix, paint)
        Log.i(TAG, "align: produced ${out.width}x${out.height} aligned crop polyMapOk=$mapped")
        return out
    }

    companion object {
        private const val TAG: String = "FaceMesh.Aligner"
        const val OUTPUT_SIZE: Int = 112

        /**
         * ArcFace canonical landmark template (5-point), reduced to the 4 we use:
         * right-eye, left-eye, nose, mouth-centre. Original mouth-corner pair (4,5) is averaged.
         *
         * Source: ArcFace authors' reference template at 112x112.
         *   right_eye   = (38.2946, 51.6963)
         *   left_eye    = (73.5318, 51.5014)
         *   nose        = (56.0252, 71.7366)
         *   mouth_center = mean( (41.5493, 92.3655), (70.7299, 92.2041) ) = (56.1396, 92.2848)
         */
        private val canonicalDestination: FloatArray = floatArrayOf(
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            56.0252f, 71.7366f,
            56.1396f, 92.2848f,
        )
    }
}
