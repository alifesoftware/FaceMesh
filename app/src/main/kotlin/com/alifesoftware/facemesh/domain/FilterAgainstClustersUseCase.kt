package com.alifesoftware.facemesh.domain

import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.data.ClusterRepository
import com.alifesoftware.facemesh.ml.FaceProcessor
import com.alifesoftware.facemesh.ml.cluster.EmbeddingMath
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.yield

/**
 * Phase 2 of the SPEC pipeline (\u00a76.6): per-image cosine match against the centroids of the
 * user-checked clusters. An image is a Keeper iff at least one face in it scores
 * \u2265 [matchThreshold] against any selected centroid.
 */
class FilterAgainstClustersUseCase(
    private val processor: FaceProcessor,
    private val clusterRepository: ClusterRepository,
    private val preferences: AppPreferences,
) {

    sealed interface Event {
        data class Progress(val processed: Int, val total: Int) : Event
        data class Done(val keepers: List<Uri>) : Event
    }

    fun run(images: List<Uri>, selectedClusterIds: Set<String>): Flow<Event> = flow {
        Log.i(
            TAG,
            "run: invoked images=${images.size} selectedClusters=${selectedClusterIds.size}",
        )
        if (images.isEmpty() || selectedClusterIds.isEmpty()) {
            Log.i(TAG, "run: empty input -> emitting Done(0)")
            emit(Event.Done(emptyList()))
            return@flow
        }

        val matchThreshold = preferences.matchThreshold.first()
        val matchThresholdSource = preferences.matchThresholdSource.first()
        val centroids = clusterRepository.loadCentroidsForIds(selectedClusterIds).map { it.second }
        Log.i(
            TAG,
            "run: matchThreshold=$matchThreshold(source=$matchThresholdSource) " +
                "loadedCentroids=${centroids.size} (requested=${selectedClusterIds.size})",
        )
        if (centroids.isEmpty()) {
            Log.w(TAG, "run: no centroids resolved for selected cluster ids -> emitting Done(0)")
            emit(Event.Done(emptyList()))
            return@flow
        }

        val phaseStart = SystemClock.elapsedRealtime()
        val keepers = ArrayList<Uri>(images.size)
        images.forEachIndexed { index, uri ->
            yield()
            try {
                val faces = processor.process(uri, keepDisplayCrop = false)
                if (faces.isEmpty()) {
                    Log.i(
                        TAG,
                        "run: ${index + 1}/${images.size} uri=$uri DROP no faces detected",
                    )
                    emit(Event.Progress(processed = index + 1, total = images.size))
                    return@forEachIndexed
                }
                var bestScore = Float.NEGATIVE_INFINITY
                var bestFaceIdx = -1
                var bestCentroidIdx = -1
                var matchingCentroidIdx = -1
                var matchingFaceIdx = -1
                outer@ for ((fIdx, face) in faces.withIndex()) {
                    for ((cIdx, centroid) in centroids.withIndex()) {
                        val score = EmbeddingMath.dot(face.embedding, centroid)
                        Log.i(
                            TAG,
                            "run: ${index + 1}/${images.size} uri=$uri " +
                                "face[$fIdx] vs centroid[$cIdx] score=${"%.3f".format(score)} " +
                                "(threshold=$matchThreshold)",
                        )
                        if (score > bestScore) {
                            bestScore = score
                            bestFaceIdx = fIdx
                            bestCentroidIdx = cIdx
                        }
                        if (score >= matchThreshold) {
                            matchingFaceIdx = fIdx
                            matchingCentroidIdx = cIdx
                            break@outer
                        }
                    }
                }
                val isKeeper = matchingCentroidIdx >= 0
                if (isKeeper) {
                    Log.i(
                        TAG,
                        "run: ${index + 1}/${images.size} uri=$uri KEEP " +
                            "via face[$matchingFaceIdx] x centroid[$matchingCentroidIdx] " +
                            "(threshold=$matchThreshold first-match short-circuited)",
                    )
                    keepers += uri
                } else {
                    Log.i(
                        TAG,
                        "run: ${index + 1}/${images.size} uri=$uri DROP " +
                            "bestScore=${"%.3f".format(bestScore)} " +
                            "(face[$bestFaceIdx] vs centroid[$bestCentroidIdx]) < $matchThreshold",
                    )
                }
            } catch (e: Exception) {
                Log.w(TAG, "run: processor failed for uri=$uri (${index + 1}/${images.size}); skipping", e)
            }
            emit(Event.Progress(processed = index + 1, total = images.size))
        }

        Log.i(
            TAG,
            "run: complete keepers=${keepers.size}/${images.size} took=" +
                "${SystemClock.elapsedRealtime() - phaseStart}ms",
        )
        emit(Event.Done(keepers))
    }.flowOn(Dispatchers.Default)

    companion object {
        private const val TAG: String = "FaceMesh.Filter"
    }
}
