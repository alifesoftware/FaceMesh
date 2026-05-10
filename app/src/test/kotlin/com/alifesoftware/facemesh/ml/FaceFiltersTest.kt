package com.alifesoftware.facemesh.ml

import android.graphics.PointF
import android.graphics.RectF
import com.alifesoftware.facemesh.config.PipelineConfig.Filters.SizeBandMode
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
    fun `SYMMETRIC_BAND drops a tight-band size outlier`() {
        val a = face(0f, 0f, 100f, 100f)
        val b = face(0f, 0f, 100f, 100f)
        val tiny = face(0f, 0f, 30f, 30f)
        val out = FaceFilters.apply(
            listOf(a, b, tiny),
            sizeBandMode = SizeBandMode.SYMMETRIC_BAND,
            sizeBandFraction = 0.3f,
        )
        // tiny (30 wide) deviates by 70 from median 100; band = 0.3 * 100 = 30 -> filtered out.
        assertEquals(2, out.size)
        assertFalse(out.any { it.boundingBox.width() == 30f })
    }

    @Test
    fun `DISABLED mode passes all faces regardless of size variance`() {
        val tiny = face(0f, 0f, 10f, 10f)
        val small = face(0f, 0f, 100f, 100f)
        val huge = face(0f, 0f, 1000f, 1000f)
        // Default mode is DISABLED in PipelineConfig.Filters.sizeBandMode.
        val out = FaceFilters.apply(listOf(tiny, small, huge))
        assertEquals("DISABLED mode must not cull on size", 3, out.size)
    }

    @Test
    fun `EXTREME_OUTLIERS_ONLY drops absurdly small faces`() {
        // Trio with one face at 5 px (way below 0.25 * median) plus two normal 100 px faces.
        val absurd = face(0f, 0f, 5f, 5f)
        val a = face(0f, 0f, 100f, 100f)
        val b = face(0f, 0f, 100f, 100f)
        val out = FaceFilters.apply(
            listOf(absurd, a, b),
            sizeBandMode = SizeBandMode.EXTREME_OUTLIERS_ONLY,
            extremeOutlierMinRatio = 0.25f,
            extremeOutlierMaxRatio = 4.0f,
        )
        assertEquals(2, out.size)
        assertFalse(out.any { it.boundingBox.width() == 5f })
    }

    @Test
    fun `EXTREME_OUTLIERS_ONLY drops absurdly large faces`() {
        val a = face(0f, 0f, 100f, 100f)
        val b = face(0f, 0f, 100f, 100f)
        val giant = face(0f, 0f, 800f, 800f) // 8x median
        val out = FaceFilters.apply(
            listOf(a, b, giant),
            sizeBandMode = SizeBandMode.EXTREME_OUTLIERS_ONLY,
            extremeOutlierMinRatio = 0.25f,
            extremeOutlierMaxRatio = 4.0f,
        )
        assertEquals(2, out.size)
        assertFalse(out.any { it.boundingBox.width() == 800f })
    }

    @Test
    fun `EXTREME_OUTLIERS_ONLY keeps reasonable size variance untouched`() {
        // A near subject at 200 px coexisting with two distant subjects at 60 px - the exact
        // case that motivated retiring the symmetric band as the default.
        val near = face(0f, 0f, 200f, 200f)
        val far1 = face(0f, 0f, 60f, 60f)
        val far2 = face(0f, 0f, 60f, 60f)
        val out = FaceFilters.apply(
            listOf(near, far1, far2),
            sizeBandMode = SizeBandMode.EXTREME_OUTLIERS_ONLY,
            extremeOutlierMinRatio = 0.25f,
            extremeOutlierMaxRatio = 4.0f,
        )
        // median = 60, band = [15..240]. All three (60, 60, 200) survive.
        assertEquals(3, out.size)
    }

    @Test
    fun `groupPhotoSizeBandSkipThreshold short-circuits regardless of mode`() {
        // Four faces -> any active mode should be skipped to avoid culling group photos.
        val faces = (0 until 4).map { face(0f, 0f, (it + 1) * 50f, (it + 1) * 50f) }
        val out = FaceFilters.apply(
            faces,
            sizeBandMode = SizeBandMode.SYMMETRIC_BAND,
            sizeBandFraction = 0.1f, // would be brutal if applied
            groupPhotoSizeBandSkipThreshold = 4,
        )
        assertEquals("group-photo skip must short-circuit the active mode", 4, out.size)
    }
}
