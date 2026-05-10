package com.alifesoftware.facemesh.ml.cluster

import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig

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
        var corePoints = 0
        var borderPoints = 0
        var coreExpansions = 0
        for (i in 0 until n) {
            if (labels[i] != UNVISITED) {
                Log.i(TAG, "run: point[$i] SKIP already labelled=${labels[i]}")
                continue
            }
            val neighbours = regionQuery(points, i)
            if (neighbours.size < minPts) {
                labels[i] = NOISE
                Log.i(
                    TAG,
                    "run: point[$i] MARK noise neighbours=${neighbours.size} < minPts=$minPts",
                )
                continue
            }
            // Expand cluster.
            labels[i] = clusterId
            corePoints++
            Log.i(
                TAG,
                "run: point[$i] CORE seeding cluster=$clusterId neighbours=${neighbours.size}",
            )
            val seeds = ArrayDeque<Int>()
            seeds.addAll(neighbours)
            var clusterMembers = 1
            while (seeds.isNotEmpty()) {
                val q = seeds.removeFirst()
                if (q == i) continue
                if (labels[q] == NOISE) {
                    labels[q] = clusterId
                    clusterMembers++
                    borderPoints++
                    Log.i(
                        TAG,
                        "run: point[$q] PROMOTE noise->cluster=$clusterId (border, via core[$i])",
                    )
                }
                if (labels[q] != UNVISITED) continue
                labels[q] = clusterId
                clusterMembers++
                val qNeighbours = regionQuery(points, q)
                if (qNeighbours.size >= minPts) {
                    corePoints++
                    coreExpansions++
                    Log.i(
                        TAG,
                        "run: point[$q] CORE expanding cluster=$clusterId neighbours=${qNeighbours.size}",
                    )
                    seeds.addAll(qNeighbours)
                } else {
                    borderPoints++
                    Log.i(
                        TAG,
                        "run: point[$q] BORDER cluster=$clusterId neighbours=${qNeighbours.size} < minPts=$minPts",
                    )
                }
            }
            Log.i(TAG, "run: closed cluster id=$clusterId memberCount=$clusterMembers")
            clusterId++
        }
        val noiseCount = labels.count { it == NOISE }
        val sizesByCluster = IntArray(clusterId)
        for (l in labels) if (l in 0 until clusterId) sizesByCluster[l]++
        val took = SystemClock.elapsedRealtime() - started
        Log.i(
            TAG,
            "run: done n=$n clusters=$clusterId noise=$noiseCount " +
                "corePoints=$corePoints borderPoints=$borderPoints expansions=$coreExpansions " +
                "sizes=${sizesByCluster.toList()} took=${took}ms",
        )
        return labels
    }

    private fun regionQuery(points: List<FloatArray>, index: Int): List<Int> {
        val origin = points[index]
        val out = ArrayList<Int>()
        var minD = Float.POSITIVE_INFINITY
        var maxD = Float.NEGATIVE_INFINITY
        for (j in points.indices) {
            val d = EmbeddingMath.cosineDistance(origin, points[j])
            if (j != index) {
                if (d < minD) minD = d
                if (d > maxD) maxD = d
            }
            if (d <= eps) out += j
        }
        Log.i(
            TAG,
            "regionQuery: point[$index] neighbours=${out.size} (within eps=$eps) " +
                "distRange=[${"%.4f".format(minD)}..${"%.4f".format(maxD)}]",
        )
        return out
    }

    companion object {
        private const val TAG: String = "FaceMesh.Dbscan"

        /**
         * Re-exposed from [PipelineConfig.Clustering.unvisitedLabel] so the algorithm body can
         * use a short local name. Update the canonical value in PipelineConfig if changed.
         */
        const val UNVISITED: Int = PipelineConfig.Clustering.unvisitedLabel

        /**
         * Re-exposed from [PipelineConfig.Clustering.noiseLabel] (also used by external callers
         * such as [com.alifesoftware.facemesh.domain.ClusterifyUseCase] to filter out noise
         * points after clustering).
         */
        const val NOISE: Int = PipelineConfig.Clustering.noiseLabel
    }
}
