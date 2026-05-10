package com.alifesoftware.facemesh.domain

import android.net.Uri
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
        if (images.isEmpty() || selectedClusterIds.isEmpty()) {
            emit(Event.Done(emptyList()))
            return@flow
        }

        val matchThreshold = preferences.matchThreshold.first()
        val centroids = clusterRepository.loadCentroidsForIds(selectedClusterIds).map { it.second }
        if (centroids.isEmpty()) {
            emit(Event.Done(emptyList()))
            return@flow
        }

        val keepers = ArrayList<Uri>(images.size)
        images.forEachIndexed { index, uri ->
            yield()
            try {
                val faces = processor.process(uri, keepRepresentativeCrop = false)
                val isKeeper = faces.any { face ->
                    centroids.any { centroid ->
                        EmbeddingMath.dot(face.embedding, centroid) >= matchThreshold
                    }
                }
                if (isKeeper) keepers += uri
            } catch (_: Exception) {
                // Skip this image; the rest of the batch still proceeds.
            }
            emit(Event.Progress(processed = index + 1, total = images.size))
        }

        emit(Event.Done(keepers))
    }.flowOn(Dispatchers.Default)
}
