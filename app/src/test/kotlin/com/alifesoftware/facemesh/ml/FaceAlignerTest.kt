package com.alifesoftware.facemesh.ml

import com.alifesoftware.facemesh.config.PipelineConfig
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-math tests for [FaceAligner.similarityFromLstsq] -- the 4-DOF similarity transform
 * fit that replaced the previous 8-DOF perspective fit. Validates round-trip recovery on
 * canonical synthetic transforms (identity, translation, scale, rotation) and on a
 * realistic face-landmark configuration drawn from a real-world test photo.
 *
 * The function under test is allocation-friendly (writes the result into a caller-provided
 * 9-float buffer) so we test it directly rather than going through Bitmap/Canvas, which
 * keeps these unit tests fast and Robolectric-light.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class FaceAlignerTest {

    private val canonical: FloatArray = PipelineConfig.Aligner.canonicalLandmarkTemplate

    /** Apply a 3x3 affine matrix (in Android setValues order) to a (x, y) point. */
    private fun applyMatrix(values: FloatArray, x: Float, y: Float): Pair<Float, Float> {
        val xx = values[0] * x + values[1] * y + values[2]
        val yy = values[3] * x + values[4] * y + values[5]
        return xx to yy
    }

    @Test
    fun `identity case -- src equals dst yields identity matrix`() {
        val out = FloatArray(9)
        FaceAligner.similarityFromLstsq(canonical.copyOf(), canonical.copyOf(), out)
        assertEquals("a (scale*cos)",  1f, out[0], 1e-4f)
        assertEquals("-b (-scale*sin)", 0f, out[1], 1e-4f)
        assertEquals("tx",              0f, out[2], 1e-3f)
        assertEquals("b (scale*sin)",   0f, out[3], 1e-4f)
        assertEquals("a (= out[0])",    1f, out[4], 1e-4f)
        assertEquals("ty",              0f, out[5], 1e-3f)
        assertEquals("perspective[0]",  0f, out[6], 0f)
        assertEquals("perspective[1]",  0f, out[7], 0f)
        assertEquals("perspective[2]",  1f, out[8], 0f)
    }

    @Test
    fun `pure translation -- shifted src recovers the inverse translation`() {
        // Source = canonical shifted by (+10, +20). The fit should produce a transform that
        // *reverses* the shift, i.e. translation (-10, -20) with scale 1 and zero rotation.
        val src = FloatArray(canonical.size) { i ->
            canonical[i] + if (i % 2 == 0) 10f else 20f
        }
        val out = FloatArray(9)
        FaceAligner.similarityFromLstsq(src, canonical, out)
        assertEquals(1f,  out[0], 1e-3f)
        assertEquals(0f,  out[1], 1e-3f)
        assertEquals(-10f, out[2], 1e-2f)
        assertEquals(0f,  out[3], 1e-3f)
        assertEquals(1f,  out[4], 1e-3f)
        assertEquals(-20f, out[5], 1e-2f)
    }

    @Test
    fun `pure scale -- 2x source recovers 0_5x scale`() {
        val src = FloatArray(canonical.size) { i -> canonical[i] * 2f }
        val out = FloatArray(9)
        FaceAligner.similarityFromLstsq(src, canonical, out)
        assertEquals(0.5f, out[0], 1e-3f)  // scale*cos
        assertEquals(0f,   out[1], 1e-3f)  // -scale*sin
        assertEquals(0f,   out[2], 1e-2f)  // tx (small because canonical isn't centred at origin)
        assertEquals(0f,   out[3], 1e-3f)
        assertEquals(0.5f, out[4], 1e-3f)
        assertEquals(0f,   out[5], 1e-2f)
    }

    @Test
    fun `round trip -- known similarity applied to canonical recovers within ~1px`() {
        // Build a synthetic source by transforming the canonical template through a known
        // similarity (rotate 15 degrees, scale 1.5, translate (+30, -20)). The lstsq fit
        // must invert it: applying the resulting matrix to src should recover canonical
        // within numerical precision.
        val theta = (15.0 * Math.PI / 180.0)
        val s = 1.5f
        val tx = 30f
        val ty = -20f
        val cosT = (s * cos(theta)).toFloat()
        val sinT = (s * sin(theta)).toFloat()
        val src = FloatArray(canonical.size)
        for (i in 0 until canonical.size / 2) {
            val x = canonical[2 * i]
            val y = canonical[2 * i + 1]
            // forward: src = R(theta) * scale * canonical + (tx, ty)
            src[2 * i]     = cosT * x - sinT * y + tx
            src[2 * i + 1] = sinT * x + cosT * y + ty
        }
        val out = FloatArray(9)
        FaceAligner.similarityFromLstsq(src, canonical, out)
        for (i in 0 until canonical.size / 2) {
            val (mx, my) = applyMatrix(out, src[2 * i], src[2 * i + 1])
            assertEquals("recovered x[$i]", canonical[2 * i],     mx, 0.5f)
            assertEquals("recovered y[$i]", canonical[2 * i + 1], my, 0.5f)
        }
    }

    @Test
    fun `realistic Virat-like landmarks produce a well-conditioned scale and small residuals`() {
        // Approximation of the virat_head_01 source landmarks (498x576 source) used during
        // the cross-pipeline parity work. Eye distance in source is ~136 px; ArcFace
        // canonical eye distance is ~35.24 px; expected scale = 35.24 / 136 ~= 0.259.
        val src = floatArrayOf(
            156.8f, 259.8f,  // right eye
            293.1f, 266.3f,  // left eye
            205.9f, 335.6f,  // nose tip
            209.9f, 407.3f,  // mouth centre
        )
        val out = FloatArray(9)
        FaceAligner.similarityFromLstsq(src, canonical, out)
        val a = out[0]
        val b = out[3]
        val recoveredScale = sqrt((a * a + b * b).toDouble()).toFloat()
        assertEquals("recovered scale", 0.259f, recoveredScale, 0.02f)

        // The 4 source landmarks aren't *exactly* a similarity transform of the canonical
        // template (real faces drift slightly from the InsightFace ideal), so an exact fit
        // is impossible. But the RMS residual after applying the matrix should be small --
        // typical real-world InsightFace alignments report a few pixels.
        var sumSq = 0f
        for (i in 0 until 4) {
            val (mx, my) = applyMatrix(out, src[2 * i], src[2 * i + 1])
            val ex = mx - canonical[2 * i]
            val ey = my - canonical[2 * i + 1]
            sumSq += ex * ex + ey * ey
        }
        val rms = sqrt((sumSq / 4f).toDouble()).toFloat()
        assertEquals("RMS landmark residual must be small", 0f, rms, 5f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `mismatched src and dst sizes throw`() {
        FaceAligner.similarityFromLstsq(FloatArray(8), FloatArray(6), FloatArray(9))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `wrong out buffer size throws`() {
        FaceAligner.similarityFromLstsq(FloatArray(8), FloatArray(8), FloatArray(8))
    }

    @Test(expected = IllegalStateException::class)
    fun `singular configuration -- all source points coincide -- throws`() {
        // 4 source points all at the same location is a degenerate input: scale and rotation
        // are unrecoverable. Solver should detect the singular pivot and throw.
        val src = FloatArray(8) { 50f }
        FaceAligner.similarityFromLstsq(src, canonical.copyOf(), FloatArray(9))
    }
}
