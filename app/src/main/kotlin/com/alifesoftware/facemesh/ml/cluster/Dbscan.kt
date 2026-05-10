package com.alifesoftware.facemesh.ml.cluster

import android.os.SystemClock
import android.util.Log

/**
 * Density-based spatial clustering (DBSCAN) using cosine distance over L2-normalised face
 * embeddings (SPEC \u00a76.5).
 *
 *   \u2022 `eps`     \u2014 maximum cosine distance for two points to be considered neighbours.
 *   \u2022 `minPts`  \u2014 minimum number of neighbours (including self) to be a *core* point.
 *
 * Returns a label per input point: 0..N-1 for clusters, [NOISE] (-1) for outliers.
 */
class Dbscan(
    private val eps: Float,
    private val minPts: Int,
) {

    fun run(points: List<FloatArray>): IntArray {
        val n = points.size
        val labels = IntArray(n) { UNVISITED }
        if (n == 0) {
            Log.i(TAG, "run: empty input; returning empty labels (eps=$eps minPts=$minPts)")
            return labels
        }

        val started = SystemClock.elapsedRealtime()
        Log.i(TAG, "run: start n=$n eps=$eps minPts=$minPts")

        var clusterId = 0
        for (i in 0 until n) {
            if (labels[i] != UNVISITED) continue
            val neighbours = regionQuery(points, i)
            if (neighbours.size < minPts) {
                labels[i] = NOISE
                continue
            }
            // Expand cluster.
            labels[i] = clusterId
            val seeds = ArrayDeque<Int>()
            seeds.addAll(neighbours)
            while (seeds.isNotEmpty()) {
                val q = seeds.removeFirst()
                if (q == i) continue
                if (labels[q] == NOISE) {
                    labels[q] = clusterId
                }
                if (labels[q] != UNVISITED) continue
                labels[q] = clusterId
                val qNeighbours = regionQuery(points, q)
                if (qNeighbours.size >= minPts) {
                    seeds.addAll(qNeighbours)
                }
            }
            Log.i(TAG, "run: closed cluster id=$clusterId (members so far counted at end)")
            clusterId++
        }
        val noiseCount = labels.count { it == NOISE }
        val sizesByCluster = IntArray(clusterId)
        for (l in labels) if (l in 0 until clusterId) sizesByCluster[l]++
        val took = SystemClock.elapsedRealtime() - started
        Log.i(
            TAG,
            "run: done n=$n clusters=$clusterId noise=$noiseCount sizes=${sizesByCluster.toList()} " +
                "took=${took}ms",
        )
        return labels
    }

    private fun regionQuery(points: List<FloatArray>, index: Int): List<Int> {
        val origin = points[index]
        val out = ArrayList<Int>()
        for (j in points.indices) {
            if (EmbeddingMath.cosineDistance(origin, points[j]) <= eps) {
                out += j
            }
        }
        return out
    }

    companion object {
        private const val TAG: String = "FaceMesh.Dbscan"
        const val UNVISITED: Int = -2
        const val NOISE: Int = -1
    }
}
