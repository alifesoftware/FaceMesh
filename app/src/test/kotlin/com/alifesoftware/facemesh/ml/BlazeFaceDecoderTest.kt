package com.alifesoftware.facemesh.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class BlazeFaceDecoderTest {

    @Test
    fun `sigmoid is monotonic and bounded`() {
        val a = BlazeFaceDecoder.sigmoid(-100f)
        val b = BlazeFaceDecoder.sigmoid(0f)
        val c = BlazeFaceDecoder.sigmoid(100f)
        assertTrue(a in 0f..1f)
        assertTrue(b in 0f..1f)
        assertTrue(c in 0f..1f)
        assertTrue("monotonic", a < b && b < c)
        assertEquals(0.5f, b, 1e-6f)
    }

    @Test
    fun `decode filters out low-confidence anchors`() {
        // Synthetic single-anchor decoder so we control inputs precisely.
        val anchors = arrayOf(BlazeFaceAnchors.Anchor(0.5f, 0.5f))
        val decoder = BlazeFaceDecoder(
            inputSize = 100,
            anchors = anchors,
            scoreThreshold = 0.9f,
        )
        val regs = FloatArray(BlazeFaceDecoder.REG_STRIDE) // box centred, w/h = 0
        val logits = floatArrayOf(0f) // sigmoid(0) = 0.5 < 0.9
        val out = decoder.decode(regs, logits, sourceWidth = 100, sourceHeight = 100)
        assertEquals(0, out.size)
    }

    @Test
    fun `decode emits a face when score exceeds threshold and box is non-degenerate`() {
        val anchors = arrayOf(BlazeFaceAnchors.Anchor(0.5f, 0.5f))
        val decoder = BlazeFaceDecoder(
            inputSize = 100,
            anchors = anchors,
            scoreThreshold = 0.5f,
        )
        val regs = FloatArray(BlazeFaceDecoder.REG_STRIDE).apply {
            // Centre offset = 0, w = 60, h = 60, so the box is (20, 20, 80, 80) in input space.
            this[2] = 60f
            this[3] = 60f
            // Landmarks: right eye at (-10, -10) offset, left eye at (10, -10), etc.
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
        // Eye distance derived from landmarks = sqrt(20^2 + 0^2) = 20
        assertEquals(20f, face.landmarks.eyeDistance(), 1e-3f)
    }

    @Test
    fun `decode performs NMS on heavily overlapping detections`() {
        // Two anchors at the same centre; both regress to same box.
        val anchors = arrayOf(
            BlazeFaceAnchors.Anchor(0.5f, 0.5f),
            BlazeFaceAnchors.Anchor(0.5f, 0.5f),
        )
        val decoder = BlazeFaceDecoder(
            inputSize = 100,
            anchors = anchors,
            scoreThreshold = 0.5f,
            iouThreshold = 0.30f,
        )
        val perAnchor = FloatArray(BlazeFaceDecoder.REG_STRIDE).apply {
            this[2] = 60f
            this[3] = 60f
        }
        val regs = perAnchor + perAnchor // both anchors emit the same box
        val logits = floatArrayOf(4f, 3f)
        val out = decoder.decode(regs, logits, sourceWidth = 100, sourceHeight = 100)
        assertEquals("NMS should fuse identical boxes", 1, out.size)
    }
}
