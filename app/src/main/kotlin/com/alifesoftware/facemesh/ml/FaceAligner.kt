package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig

/**
 * Affinely aligns a face crop into the canonical 112x112 pose used by GhostFaceNet
 * (SPEC \u00a76.3).
 *
 * Uses [Matrix.setPolyToPoly] with **4 source / 4 destination** points (the maximum the API
 * supports). The 4 anchor points are right-eye, left-eye, nose-tip, mouth-centre, mapped to a
 * canonical ArcFace-style template normalised to 112x112.
 */
class FaceAligner(
    private val outputSize: Int = PipelineConfig.Aligner.outputSize,
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
        val canonical = PipelineConfig.Aligner.canonicalLandmarkTemplate
        Log.i(
            TAG,
            "align: dst canonical " +
                "rightEye=(${canonical[0]},${canonical[1]}) " +
                "leftEye=(${canonical[2]},${canonical[3]}) " +
                "nose=(${canonical[4]},${canonical[5]}) " +
                "mouth=(${canonical[6]},${canonical[7]})",
        )
        val src = floatArrayOf(
            face.landmarks.rightEye.x, face.landmarks.rightEye.y,
            face.landmarks.leftEye.x, face.landmarks.leftEye.y,
            face.landmarks.noseTip.x, face.landmarks.noseTip.y,
            face.landmarks.mouthCenter.x, face.landmarks.mouthCenter.y,
        )
        val dst = PipelineConfig.Aligner.canonicalLandmarkTemplate
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
    }
}
