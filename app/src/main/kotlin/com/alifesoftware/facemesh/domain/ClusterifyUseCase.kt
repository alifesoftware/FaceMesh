package com.alifesoftware.facemesh.domain

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.data.ClusterImageEntity
import com.alifesoftware.facemesh.data.ClusterRepository
import com.alifesoftware.facemesh.domain.model.Cluster
import com.alifesoftware.facemesh.ml.FaceProcessor
import com.alifesoftware.facemesh.ml.cluster.Dbscan
import com.alifesoftware.facemesh.ml.cluster.EmbeddingMath
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.yield
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

/**
 * Phase 1 of the SPEC pipeline (\u00a76.1\u20136.5): turn N source URIs into M persisted clusters.
 *
 * Emits progress as it processes each source image so the UI can render the per-image counter.
 */
class ClusterifyUseCase(
    private val context: Context,
    private val processor: FaceProcessor,
    private val clusterRepository: ClusterRepository,
    private val preferences: AppPreferences,
) {

    sealed interface Event {
        data class Progress(val processed: Int, val total: Int) : Event
        data object NoFaces : Event
        data class Done(val clusters: List<Cluster>) : Event
    }

    fun run(sources: List<Uri>): Flow<Event> = flow {
        if (sources.isEmpty()) {
            emit(Event.NoFaces)
            return@flow
        }

        val eps = preferences.dbscanEps.first()
        val minPts = preferences.dbscanMinPts.first()
        val createdAt = System.currentTimeMillis()

        val records = ArrayList<FaceProcessor.FaceRecord>(sources.size * 2)
        sources.forEachIndexed { index, uri ->
            yield()
            try {
                records.addAll(processor.process(uri, keepRepresentativeCrop = true))
            } catch (e: Exception) {
                // Per-image failures are tolerated; the rest of the batch still proceeds.
            }
            emit(Event.Progress(processed = index + 1, total = sources.size))
        }

        if (records.isEmpty()) {
            emit(Event.NoFaces)
            return@flow
        }

        val dbscan = Dbscan(eps = eps, minPts = minPts)
        val labels = dbscan.run(records.map { it.embedding })

        val grouped = LinkedHashMap<Int, MutableList<FaceProcessor.FaceRecord>>()
        labels.forEachIndexed { idx, label ->
            if (label == Dbscan.NOISE) return@forEachIndexed
            grouped.getOrPut(label) { ArrayList() }.add(records[idx])
        }

        if (grouped.isEmpty()) {
            emit(Event.NoFaces)
            return@flow
        }

        val resultClusters = ArrayList<Cluster>(grouped.size)
        for ((_, members) in grouped) {
            val centroid = EmbeddingMath.meanAndNormalize(members.map { it.embedding })
            val rep = pickRepresentative(members)
            val repBitmap = members.firstOrNull { it.faceIndex == 0 && it.representativeCrop != null }?.representativeCrop
            val savedThumbUri = if (repBitmap != null) {
                saveRepresentative(repBitmap)
            } else {
                rep.sourceUri
            }
            val clusterId = UUID.randomUUID().toString()
            val cluster = Cluster(
                id = clusterId,
                representativeImageUri = savedThumbUri,
                faceCount = members.size,
            )
            val rows = members.map {
                ClusterImageEntity(
                    clusterId = clusterId,
                    imageUri = it.sourceUri.toString(),
                    faceIndex = it.faceIndex,
                    embedding = it.embedding,
                )
            }
            clusterRepository.saveCluster(cluster, centroid, rows, createdAt)
            resultClusters += cluster
        }

        // Recycle held representative crops we already persisted.
        records.forEach { it.representativeCrop?.recycle() }

        emit(Event.Done(resultClusters))
    }.flowOn(Dispatchers.Default)

    private fun pickRepresentative(members: List<FaceProcessor.FaceRecord>): FaceProcessor.FaceRecord =
        members.firstOrNull { it.faceIndex == 0 && it.representativeCrop != null }
            ?: members.first()

    /** Persists a representative thumbnail as PNG into private app files; returns its file:// Uri. */
    private fun saveRepresentative(bitmap: Bitmap): Uri {
        val dir = File(context.filesDir, REPRESENTATIVE_DIR).apply { mkdirs() }
        val out = File(dir, "${UUID.randomUUID()}.png")
        FileOutputStream(out).use { fos ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }
        return Uri.fromFile(out)
    }

    companion object {
        const val REPRESENTATIVE_DIR: String = "representatives"
    }
}
