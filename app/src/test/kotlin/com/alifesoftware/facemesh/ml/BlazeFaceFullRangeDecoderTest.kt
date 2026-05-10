package com.alifesoftware.facemesh.ml

import com.alifesoftware.facemesh.config.PipelineConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Mirror of [BlazeFaceShortRangeDecoderTest] for the full-range variant. The tests are
 * intentionally duplicated rather than parameterised because the two decoders are
 * deliberately isolated implementations - if a future change drifts one decoder's math, we
 * want the other's test to keep passing untouched.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class BlazeFaceFullRangeDecoderTest {

    @Test
    fun `sigmoid is monotonic and bounded`() {
        val a = BlazeFaceFullRangeDecoder.sigmoid(-100f)
        val b = BlazeFaceFullRangeDecoder.sigmoid(0f)
        val c = BlazeFaceFullRangeDecoder.sigmoid(100f)
        assertTrue(a in 0f..1f)
        assertTrue(b in 0f..1f)
        assertTrue(c in 0f..1f)
        assertTrue("monotonic", a < b && b < c)
        assertEquals(0.5f, b, 1e-6f)
    }

    @Test
    fun `decode filters out low-confidence anchors`() {
        val anchors = arrayOf(BlazeFaceFullRangeAnchors.Anchor(0.5f, 0.5f))
        val decoder = BlazeFaceFullRangeDecoder(
            inputSize = 100,
            anchors = anchors,
            scoreThreshold = 0.9f,
        )
        val regs = FloatArray(PipelineConfig.Detector.FullRange.regStride)
        val logits = floatArrayOf(0f) // sigmoid(0) = 0.5 < 0.9
        val out = decoder.decode(regs, logits, sourceWidth = 100, sourceHeight = 100)
        assertEquals(0, out.size)
    }

    @Test
    fun `decode emits a face when score exceeds threshold and box is non-degenerate`() {
        val anchors = arrayOf(BlazeFaceFullRangeAnchors.Anchor(0.5f, 0.5f))
        val decoder = BlazeFaceFullRangeDecoder(
            inputSize = 100,
            anchors = anchors,
            scoreThreshold = 0.5f,
        )
        val regs = FloatArray(PipelineConfig.Detector.FullRange.regStride).apply {
            this[2] = 60f
            this[3] = 60f
            // 6 landmark (x, y) pairs starting at index 4.
            this[4] = -10f; this[5] = -10f
            this[6] = 10f;  this[7] = -10f
            this[8] = 0f;   this[9] = 5f
            this[10] = 0f;  this[11] = 15f
            this[12] = -20f; this[13] = 0f
            this[14] = 20f;  this[15] = 0f
        }
        val logits = floatArrayOf(5f) // sigmoid(5) ~ 0.993 > 0.5
        val out = decoder.decode(regs, logits, sourceWidth = 100, sourceHeight = 100)
        assertEquals(1, out.size)
        val face = out[0]
        assertEquals(20f, face.boundingBox.left, 1e-3f)
        assertEquals(80f, face.boundingBox.right, 1e-3f)
        assertEquals(20f, face.boundingBox.top, 1e-3f)
        assertEquals(80f, face.boundingBox.bottom, 1e-3f)
        assertEquals(20f, face.landmarks.eyeDistance(), 1e-3f)
    }

    @Test
    fun `decode performs NMS on heavily overlapping detections`() {
        val anchors = arrayOf(
            BlazeFaceFullRangeAnchors.Anchor(0.5f, 0.5f),
            BlazeFaceFullRangeAnchors.Anchor(0.5f, 0.5f),
        )
        val decoder = BlazeFaceFullRangeDecoder(
            inputSize = 100,
            anchors = anchors,
            scoreThreshold = 0.5f,
            iouThreshold = 0.30f,
        )
        val perAnchor = FloatArray(PipelineConfig.Detector.FullRange.regStride).apply {
            this[2] = 60f
            this[3] = 60f
        }
        val regs = perAnchor + perAnchor
        val logits = floatArrayOf(4f, 3f)
        val out = decoder.decode(regs, logits, sourceWidth = 100, sourceHeight = 100)
        assertEquals("NMS should fuse identical boxes", 1, out.size)
    }
}
