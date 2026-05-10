package com.alifesoftware.facemesh.ml.cluster

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class DbscanTest {

    /** Build a unit vector at angle [degrees] in 2D \u2014 great for cosine-distance fixtures. */
    private fun unit(degrees: Double): FloatArray {
        val r = Math.toRadians(degrees)
        return floatArrayOf(cos(r).toFloat(), sin(r).toFloat())
    }

    @Test
    fun `groups two tight clusters and marks an outlier as noise`() {
        // Cluster A: 5 vectors near 0 deg.
        val clusterA = (0..4).map { unit((-2 + it).toDouble()) }
        // Cluster B: 5 vectors near 90 deg.
        val clusterB = (0..4).map { unit((88 + it).toDouble()) }
        // Outlier at 45 deg \u2014 cosine distance to either centre is large enough at eps=0.05.
        val outlier = unit(45.0)
        val all = clusterA + clusterB + listOf(outlier)

        val labels = Dbscan(eps = 0.01f, minPts = 3).run(all)

        // First 5 share a label, next 5 share a label, last one is noise.
        val labelA = labels[0]
        val labelB = labels[5]
        assertNotEquals("clusters must differ", labelA, labelB)
        assertNotEquals("must not be noise", Dbscan.NOISE, labelA)
        assertNotEquals("must not be noise", Dbscan.NOISE, labelB)
        for (i in 0 until 5) assertEquals(labelA, labels[i])
        for (i in 5 until 10) assertEquals(labelB, labels[i])
        assertEquals(Dbscan.NOISE, labels[10])
    }

    @Test
    fun `single point with minPts greater than one is noise`() {
        val labels = Dbscan(eps = 0.5f, minPts = 2).run(listOf(unit(0.0)))
        assertEquals(Dbscan.NOISE, labels[0])
    }

    @Test
    fun `empty input returns empty array`() {
        val labels = Dbscan(eps = 0.5f, minPts = 1).run(emptyList())
        assertEquals(0, labels.size)
    }

    @Test
    fun `all points in one tight cluster get the same label`() {
        val pts = (0..9).map { unit(it * 0.05) }
        val labels = Dbscan(eps = 0.001f, minPts = 2).run(pts)
        val first = labels[0]
        labels.forEach { assertEquals(first, it) }
        assertNotEquals(Dbscan.NOISE, first)
    }
}
