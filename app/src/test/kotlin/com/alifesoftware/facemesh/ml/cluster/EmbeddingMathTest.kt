package com.alifesoftware.facemesh.ml.cluster

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

class EmbeddingMathTest {

    @Test
    fun `l2NormalizeInPlace produces unit vector`() {
        val v = floatArrayOf(3f, 4f)
        EmbeddingMath.l2NormalizeInPlace(v)
        assertEquals(0.6f, v[0], 1e-6f)
        assertEquals(0.8f, v[1], 1e-6f)
        assertEquals(1f, sqrt(v[0] * v[0] + v[1] * v[1]), 1e-6f)
    }

    @Test
    fun `l2NormalizeInPlace zero vector stays zero`() {
        val v = FloatArray(8) // all zeros
        EmbeddingMath.l2NormalizeInPlace(v)
        v.forEach { assertEquals(0f, it, 0f) }
    }

    @Test
    fun `dot of unit vectors equals cosine similarity`() {
        val a = floatArrayOf(0.6f, 0.8f)
        val b = floatArrayOf(0.8f, 0.6f)
        // both already unit norm
        val expected = 0.6f * 0.8f + 0.8f * 0.6f
        assertEquals(expected, EmbeddingMath.dot(a, b), 1e-6f)
    }

    @Test
    fun `cosineDistance complements similarity`() {
        val a = floatArrayOf(1f, 0f)
        val b = floatArrayOf(0f, 1f)
        assertEquals(1f, EmbeddingMath.cosineDistance(a, b), 1e-6f)
        val c = floatArrayOf(1f, 0f)
        val d = floatArrayOf(1f, 0f)
        assertEquals(0f, EmbeddingMath.cosineDistance(c, d), 1e-6f)
    }

    @Test
    fun `meanAndNormalize centroid is unit vector and equidistant`() {
        val a = floatArrayOf(1f, 0f, 0f)
        val b = floatArrayOf(0f, 1f, 0f)
        val centroid = EmbeddingMath.meanAndNormalize(listOf(a, b))
        // (0.5, 0.5, 0) normalised -> (1/sqrt(2), 1/sqrt(2), 0)
        val s = (1f / sqrt(2.0f))
        assertEquals(s, centroid[0], 1e-6f)
        assertEquals(s, centroid[1], 1e-6f)
        assertEquals(0f, centroid[2], 1e-6f)
        // Symmetric distance to both members.
        val da = EmbeddingMath.cosineDistance(a, centroid)
        val db = EmbeddingMath.cosineDistance(b, centroid)
        assertTrue("expected symmetric", abs(da - db) < 1e-6f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `dot rejects mismatched sizes`() {
        EmbeddingMath.dot(floatArrayOf(1f), floatArrayOf(1f, 0f))
    }
}
