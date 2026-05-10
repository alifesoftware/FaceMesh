package com.alifesoftware.facemesh.ml

import android.graphics.PointF
import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class FaceFiltersTest {

    private fun face(
        left: Float, top: Float, right: Float, bottom: Float,
        eyeDistance: Float = (right - left) * 0.4f,
        score: Float = 0.9f,
    ): DetectedFace {
        val cy = (top + bottom) / 2f
        val cx = (left + right) / 2f
        val rightEye = PointF(cx - eyeDistance / 2f, cy)
        val leftEye = PointF(cx + eyeDistance / 2f, cy)
        return DetectedFace(
            boundingBox = RectF(left, top, right, bottom),
            landmarks = FaceLandmarks(
                rightEye = rightEye,
                leftEye = leftEye,
                noseTip = PointF(cx, cy),
                mouthCenter = PointF(cx, cy + eyeDistance / 2f),
                rightEarTragion = PointF(left, cy),
                leftEarTragion = PointF(right, cy),
            ),
            score = score,
        )
    }

    @Test
    fun `low confidence faces are filtered out`() {
        val low = face(0f, 0f, 100f, 100f, score = 0.5f)
        val high = face(0f, 0f, 100f, 100f, score = 0.9f)
        val out = FaceFilters.apply(listOf(low, high), confidenceThreshold = 0.7f)
        assertEquals(1, out.size)
        assertEquals(0.9f, out.first().score, 0f)
    }

    @Test
    fun `geometric outlier with too-narrow eye distance is rejected`() {
        // Eye distance / width = 5 / 100 = 0.05, well below the default min 0.25.
        val skinny = face(0f, 0f, 100f, 100f, eyeDistance = 5f)
        val out = FaceFilters.apply(listOf(skinny))
        assertTrue(out.isEmpty())
    }

    @Test
    fun `single accepted face passes all filters`() {
        val good = face(0f, 0f, 100f, 100f) // eyeDistance = 40 -> ratio 0.4
        val out = FaceFilters.apply(listOf(good))
        assertEquals(1, out.size)
    }

    @Test
    fun `size outlier dropped when band fraction is tight`() {
        val a = face(0f, 0f, 100f, 100f)
        val b = face(0f, 0f, 100f, 100f)
        val tiny = face(0f, 0f, 30f, 30f)
        val out = FaceFilters.apply(
            listOf(a, b, tiny),
            sizeBandFraction = 0.3f,
        )
        // tiny (30 wide) deviates by 70 from median 100; band 30 -> filtered out.
        assertEquals(2, out.size)
        assertFalse(out.any { it.boundingBox.width() == 30f })
    }
}
