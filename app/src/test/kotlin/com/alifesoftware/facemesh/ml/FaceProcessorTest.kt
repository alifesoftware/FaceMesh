package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class FaceProcessorTest {

    private fun blankBitmap(w: Int, h: Int): Bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

    @Test
    fun `cropDisplayThumbnail produces a square crop with padding`() {
        val source = blankBitmap(1000, 1000)
        val bbox = RectF(400f, 400f, 600f, 600f) // 200 x 200 face in the middle
        val out = FaceProcessor.cropDisplayThumbnail(
            source = source,
            bbox = bbox,
            paddingFraction = 0.30f,
            maxOutputDim = 256,
        )
        assertNotNull(out)
        // Padded face = 200 * 1.6 = 320 px, downscaled to 256.
        assertEquals(256, out!!.width)
        assertEquals(256, out.height)
    }

    @Test
    fun `cropDisplayThumbnail downscales to maxOutputDim`() {
        val source = blankBitmap(2000, 2000)
        val bbox = RectF(800f, 800f, 1200f, 1200f) // 400 px face -> padded 640 -> capped 256
        val out = FaceProcessor.cropDisplayThumbnail(source, bbox, paddingFraction = 0.30f, maxOutputDim = 256)
        assertNotNull(out)
        assertEquals(256, out!!.width)
        assertEquals(256, out.height)
    }

    @Test
    fun `cropDisplayThumbnail keeps result square even when face is near top-left edge`() {
        val source = blankBitmap(1000, 1000)
        // Face hugs the top-left corner; padded square would clip past (0,0).
        val bbox = RectF(20f, 20f, 200f, 200f) // 180 px -> padded 288
        val out = FaceProcessor.cropDisplayThumbnail(source, bbox, paddingFraction = 0.30f, maxOutputDim = 256)
        assertNotNull(out)
        assertEquals(out!!.width, out.height) // strictly square
    }

    @Test
    fun `cropDisplayThumbnail keeps result square even when face is near bottom-right edge`() {
        val source = blankBitmap(1000, 1000)
        val bbox = RectF(800f, 800f, 980f, 980f)
        val out = FaceProcessor.cropDisplayThumbnail(source, bbox, paddingFraction = 0.30f, maxOutputDim = 256)
        assertNotNull(out)
        assertEquals(out!!.width, out.height)
    }

    @Test
    fun `cropDisplayThumbnail caps padded crop to source dimensions`() {
        val source = blankBitmap(300, 300)
        val bbox = RectF(50f, 50f, 250f, 250f) // 200 px face in a 300 px image
        val out = FaceProcessor.cropDisplayThumbnail(source, bbox, paddingFraction = 0.50f, maxOutputDim = 512)
        assertNotNull(out)
        // Padded crop would be 200 * 2 = 400 px, but source is only 300 -> clamped.
        assertTrue(out!!.width <= source.width)
        assertTrue(out.height <= source.height)
        assertEquals(out.width, out.height)
    }

    @Test
    fun `cropDisplayThumbnail returns null for tiny degenerate sources`() {
        val source = blankBitmap(2, 2)
        val bbox = RectF(0f, 0f, 1f, 1f)
        val out = FaceProcessor.cropDisplayThumbnail(source, bbox, paddingFraction = 0.30f, maxOutputDim = 256)
        assertEquals(null, out)
    }

    /**
     * Regression for the use-after-free corner case: when the padded square crop covers the
     * entire source bitmap, `Bitmap.createBitmap(source, 0, 0, w, h)` returns the source
     * itself per the documented Android contract. FaceProcessor.process recycles the source
     * in finally{}, which would invalidate any FaceRecord.displayCrop that aliased it. The
     * defensive copy in cropDisplayThumbnail must guarantee the returned bitmap survives.
     */
    @Test
    fun `cropDisplayThumbnail returns a bitmap that is not the source even when crop covers everything`() {
        val source = blankBitmap(300, 300)
        val bbox = RectF(50f, 50f, 250f, 250f) // 200 px face -> padded 400 -> clamped to 300x300
        val out = FaceProcessor.cropDisplayThumbnail(source, bbox, paddingFraction = 0.50f, maxOutputDim = 512)
        assertNotNull(out)
        // Critical: must be a *different* Bitmap object so recycling source does not kill it.
        assertNotSame("displayCrop must not alias source bitmap", source, out)
        // Sanity: shape unchanged (no rescale because 300 <= 512).
        assertEquals(300, out!!.width)
        assertEquals(300, out.height)
        // And recycling source must NOT recycle the returned crop.
        source.recycle()
        assertFalse("displayCrop must remain valid after source.recycle()", out.isRecycled)
    }
}
