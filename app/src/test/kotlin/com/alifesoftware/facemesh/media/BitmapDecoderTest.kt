package com.alifesoftware.facemesh.media

import org.junit.Assert.assertEquals
import org.junit.Test

class BitmapDecoderTest {

    @Test
    fun `computeInSampleSize is 1 for already-small images`() {
        val sample = BitmapDecoder.computeInSampleSize(width = 800, height = 600, maxDim = 1280)
        assertEquals(1, sample)
    }

    @Test
    fun `computeInSampleSize doubles until largest side fits`() {
        // 4096 longest side, target 1280 -> need to halve at least twice (1024 fits).
        val sample = BitmapDecoder.computeInSampleSize(width = 4096, height = 3072, maxDim = 1280)
        assertEquals(4, sample)
    }

    @Test
    fun `computeInSampleSize grows by powers of two`() {
        val cases = mapOf(
            (1281 to 1000) to 2,
            (2560 to 1920) to 2,
            (5120 to 3840) to 4,
            (10240 to 7680) to 8,
        )
        for ((dims, expected) in cases) {
            val (w, h) = dims
            assertEquals("for $w x $h", expected, BitmapDecoder.computeInSampleSize(w, h, 1280))
        }
    }
}
