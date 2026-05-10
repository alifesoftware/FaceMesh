package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig
import kotlin.math.abs

/**
 * Affinely aligns a face crop into the canonical 112x112 pose used by GhostFaceNet
 * (SPEC §6.3).
 *
 * Fits a **2D similarity transform** (uniform scale + rotation + translation, 4 DOF) by
 * least squares over the 4 detected landmarks (right-eye, left-eye, nose-tip, mouth-centre)
 * mapped to the corresponding ArcFace canonical positions in
 * [PipelineConfig.Aligner.canonicalLandmarkTemplate].
 *
 * ## Why not [Matrix.setPolyToPoly] with `count = 4`?
 *
 * That API does an 8-DOF perspective fit which exactly matches 4 source landmarks to 4
 * destination landmarks. On real-world face photos the underlying ground-truth transform
 * is almost always a similarity — humans don't deform much in 2D projection — so the extra
 * 4 DOF latch onto BlazeFace landmark prediction noise and produce visibly distorted
 * aligned crops, including occasional fully degenerate "kaleidoscope" outputs when the
 * landmark configuration is ill-conditioned for an 8-DOF fit. Validated against
 * `tools/reference_embed.py --alignment-mode {similarity,perspective}`: same-person cosine
 * similarity rises from ~0.25 (perspective) to ~0.67 (similarity) on the Virat/SRK test
 * triplet.
 *
 * The similarity-only fit matches the canonical ArcFace alignment used by InsightFace's
 * `cv2.estimateAffinePartial2D`. With 4 source/destination point pairs the system has 8
 * equations in 4 unknowns; we form the 4x4 normal equations and solve via Gaussian
 * elimination with partial pivoting.
 */
class FaceAligner(
    private val outputSize: Int = PipelineConfig.Aligner.outputSize,
) {

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    // Reused per-call scratch buffer for the 9-float Android Matrix value array. align() is
    // called from a single Dispatchers.Default coroutine per FaceProcessor.process(), so
    // sharing this state across calls within an aligner instance is safe.
    private val scratchMatrixValues = FloatArray(9)

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

        matrix.reset()
        val solveOk = try {
            similarityFromLstsq(src, canonical, scratchMatrixValues)
            matrix.setValues(scratchMatrixValues)
            true
        } catch (e: Exception) {
            Log.w(
                TAG,
                "align: similarity-lstsq failed (${e.message}); proceeding with identity matrix",
                e,
            )
            false
        }
        if (solveOk) {
            // Diagnostic: print the recovered scale (sqrt(a^2 + b^2)) and rotation in degrees.
            // For a typical phone-photo face landmark at >= 80px eye distance, scale should be
            // ~0.4..0.05 (mapping eye distance to ~35 canonical px) and rotation < 30deg.
            val a = scratchMatrixValues[0]
            val b = scratchMatrixValues[3]
            val scale = kotlin.math.sqrt((a * a + b * b).toDouble())
            val rotDeg = Math.toDegrees(kotlin.math.atan2(b.toDouble(), a.toDouble()))
            Log.i(
                TAG,
                "align: matrix solved=$solveOk scale=${"%.4f".format(scale)} " +
                    "rotation=${"%.2f".format(rotDeg)}deg " +
                    "values=[a=${"%.4f".format(a)} -b=${"%.4f".format(scratchMatrixValues[1])} " +
                    "tx=${"%.2f".format(scratchMatrixValues[2])} | " +
                    "b=${"%.4f".format(b)} a=${"%.4f".format(scratchMatrixValues[4])} " +
                    "ty=${"%.2f".format(scratchMatrixValues[5])}]",
            )
        }

        val out = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawBitmap(source, matrix, paint)
        Log.i(TAG, "align: produced ${out.width}x${out.height} aligned crop solveOk=$solveOk")
        return out
    }

    companion object {
        private const val TAG: String = "FaceMesh.Aligner"

        /**
         * Fit a 4-DOF similarity transform (uniform scale + rotation + translation) mapping
         * `srcPts` to `dstPts` by least squares, and write the resulting 3x3 affine matrix
         * into `outMatrixValues` in Android's row-major `[a, b, c, d, e, f, g, h, i]` order
         * (i.e. the format consumed by [Matrix.setValues]).
         *
         *   - `srcPts`, `dstPts`: flat `FloatArray`s of N (x, y) pairs (length 2N each).
         *     Must contain at least 2 pairs.
         *   - `outMatrixValues`: required length 9; rewritten on every call. Third row is set
         *     to `(0, 0, 1)`.
         *
         * The output matrix has the form
         *
         * ```
         *   [  a  -b  tx ]
         *   [  b   a  ty ]
         *   [  0   0   1 ]
         * ```
         *
         * where `a = scale*cos(rotation)` and `b = scale*sin(rotation)`. For 4 source-
         * destination point pairs the system has 8 equations in 4 unknowns; we form the 4x4
         * normal equations `A^T A x = A^T b` and solve via Gaussian elimination with partial
         * pivoting.
         *
         * Throws if the system is singular (e.g. all source points coincide).
         *
         * Internal so [FaceAlignerTest] can exercise the math directly without going through
         * [Bitmap]/[Canvas].
         */
        @JvmStatic
        internal fun similarityFromLstsq(
            srcPts: FloatArray,
            dstPts: FloatArray,
            outMatrixValues: FloatArray,
        ): FloatArray {
            require(srcPts.size == dstPts.size && srcPts.size % 2 == 0 && srcPts.size >= 4) {
                "need >=2 matching point pairs (4 floats each); got src=${srcPts.size} dst=${dstPts.size}"
            }
            require(outMatrixValues.size == 9) {
                "outMatrixValues must have length 9; got ${outMatrixValues.size}"
            }
            val n = srcPts.size / 2
            val ata = Array(4) { DoubleArray(4) }
            val atb = DoubleArray(4)
            for (i in 0 until n) {
                val sx = srcPts[2 * i].toDouble()
                val sy = srcPts[2 * i + 1].toDouble()
                val dx = dstPts[2 * i].toDouble()
                val dy = dstPts[2 * i + 1].toDouble()
                // For each pair (sx, sy) -> (dx, dy), the model
                //   dx = a*sx - b*sy + tx
                //   dy = b*sx + a*sy + ty
                // contributes two rows to A:
                //   [  sx, -sy, 1, 0 ] * [a, b, tx, ty]^T = dx
                //   [  sy,  sx, 0, 1 ] * [a, b, tx, ty]^T = dy
                accumulateRow(ata, atb, sx, -sy, 1.0, 0.0, dx)
                accumulateRow(ata, atb, sy, sx, 0.0, 1.0, dy)
            }
            val sol = solve4x4Gaussian(ata, atb)
            val a = sol[0].toFloat()
            val b = sol[1].toFloat()
            val tx = sol[2].toFloat()
            val ty = sol[3].toFloat()
            outMatrixValues[0] = a
            outMatrixValues[1] = -b
            outMatrixValues[2] = tx
            outMatrixValues[3] = b
            outMatrixValues[4] = a
            outMatrixValues[5] = ty
            outMatrixValues[6] = 0f
            outMatrixValues[7] = 0f
            outMatrixValues[8] = 1f
            return outMatrixValues
        }

        private fun accumulateRow(
            ata: Array<DoubleArray>,
            atb: DoubleArray,
            r0: Double, r1: Double, r2: Double, r3: Double,
            target: Double,
        ) {
            atb[0] += r0 * target
            atb[1] += r1 * target
            atb[2] += r2 * target
            atb[3] += r3 * target
            // Update the upper triangle then mirror; the matrix is symmetric so this saves
            // a few multiplies. Keeping it simple here since 4x4 is tiny.
            ata[0][0] += r0 * r0; ata[0][1] += r0 * r1; ata[0][2] += r0 * r2; ata[0][3] += r0 * r3
            ata[1][0] += r1 * r0; ata[1][1] += r1 * r1; ata[1][2] += r1 * r2; ata[1][3] += r1 * r3
            ata[2][0] += r2 * r0; ata[2][1] += r2 * r1; ata[2][2] += r2 * r2; ata[2][3] += r2 * r3
            ata[3][0] += r3 * r0; ata[3][1] += r3 * r1; ata[3][2] += r3 * r2; ata[3][3] += r3 * r3
        }

        /**
         * Solve a 4x4 linear system `A x = b` via Gaussian elimination with partial pivoting.
         * Throws if A is singular (smallest pivot < 1e-12).
         */
        private fun solve4x4Gaussian(A: Array<DoubleArray>, b: DoubleArray): DoubleArray {
            // Augmented matrix [A | b] of shape 4x5; modified in place.
            val m = Array(4) { i ->
                DoubleArray(5).also { row ->
                    for (j in 0 until 4) row[j] = A[i][j]
                    row[4] = b[i]
                }
            }
            for (k in 0 until 4) {
                // Partial pivoting: pick the row in [k..3] with the largest |m[i][k]|.
                var maxRow = k
                var maxVal = abs(m[k][k])
                for (i in k + 1 until 4) {
                    val v = abs(m[i][k])
                    if (v > maxVal) {
                        maxVal = v
                        maxRow = i
                    }
                }
                check(maxVal >= 1e-12) {
                    "singular matrix in similarity solve (max pivot $maxVal at column $k)"
                }
                if (maxRow != k) {
                    val tmp = m[k]
                    m[k] = m[maxRow]
                    m[maxRow] = tmp
                }
                val pivot = m[k][k]
                for (i in k + 1 until 4) {
                    val f = m[i][k] / pivot
                    for (j in k until 5) {
                        m[i][j] -= f * m[k][j]
                    }
                }
            }
            // Back-substitution.
            val x = DoubleArray(4)
            for (i in 3 downTo 0) {
                var s = m[i][4]
                for (j in i + 1 until 4) s -= m[i][j] * x[j]
                x[i] = s / m[i][i]
            }
            return x
        }
    }
}
