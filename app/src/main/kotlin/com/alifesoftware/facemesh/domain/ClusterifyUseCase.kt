package com.alifesoftware.facemesh.domain

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import android.util.Log
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
        Log.i(TAG, "run: invoked with ${sources.size} source URI(s)")
        if (sources.isEmpty()) {
            Log.i(TAG, "run: empty source list -> emitting NoFaces")
            emit(Event.NoFaces)
            return@flow
        }

        val eps = preferences.dbscanEps.first()
        val minPts = preferences.dbscanMinPts.first()
        val createdAt = System.currentTimeMillis()
        Log.i(TAG, "run: config eps=$eps minPts=$minPts createdAt=$createdAt")
        val phaseStart = SystemClock.elapsedRealtime()

        val records = ArrayList<FaceProcessor.FaceRecord>(sources.size * 2)
        sources.forEachIndexed { index, uri ->
            yield()
            try {
                val before = records.size
                records.addAll(processor.process(uri, keepDisplayCrop = true))
                Log.i(
                    TAG,
                    "run: processed ${index + 1}/${sources.size} uri=$uri -> +${records.size - before} face(s) " +
                        "(running total=${records.size})",
                )
            } catch (e: Exception) {
                Log.w(TAG, "run: processor failed for uri=$uri (${index + 1}/${sources.size}); skipping", e)
            }
            emit(Event.Progress(processed = index + 1, total = sources.size))
        }

        Log.i(
            TAG,
            "run: processing phase done; total faces=${records.size} from ${sources.size} image(s) " +
                "in ${SystemClock.elapsedRealtime() - phaseStart}ms",
        )

        if (records.isEmpty()) {
            Log.i(TAG, "run: no faces collected -> emitting NoFaces")
            emit(Event.NoFaces)
            return@flow
        }

        val dbscan = Dbscan(eps = eps, minPts = minPts)
        val clusterStart = SystemClock.elapsedRealtime()
        val labels = dbscan.run(records.map { it.embedding })
        Log.i(TAG, "run: DBSCAN done in ${SystemClock.elapsedRealtime() - clusterStart}ms")

        val grouped = LinkedHashMap<Int, MutableList<FaceProcessor.FaceRecord>>()
        labels.forEachIndexed { idx, label ->
            if (label == Dbscan.NOISE) return@forEachIndexed
            grouped.getOrPut(label) { ArrayList() }.add(records[idx])
        }
        Log.i(
            TAG,
            "run: grouped into ${grouped.size} cluster(s) " +
                "(noise=${labels.count { it == Dbscan.NOISE }})",
        )

        if (grouped.isEmpty()) {
            Log.i(TAG, "run: only noise after DBSCAN -> emitting NoFaces")
            emit(Event.NoFaces)
            return@flow
        }

        val resultClusters = ArrayList<Cluster>(grouped.size)
        for ((label, members) in grouped) {
            val centroid = EmbeddingMath.meanAndNormalize(members.map { it.embedding })
            val rep = pickRepresentative(members)
            // FaceProcessor produces a natural display crop for every accepted face when
            // keepDisplayCrop=true, so any cluster member is a viable representative \u2014 not
            // just members where faceIndex == 0. The fallback to `rep.sourceUri` (the full
            // source photo) is now reached only on degenerate inputs.
            val savedThumbUri = rep.displayCrop?.let(::saveRepresentative) ?: rep.sourceUri
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
            Log.i(
                TAG,
                "run: persisted cluster label=$label id=$clusterId members=${members.size} " +
                    "thumbFromCrop=${rep.displayCrop != null} thumbUri=$savedThumbUri",
            )
        }

        // Recycle held display crops we already persisted.
        records.forEach { it.displayCrop?.recycle() }

        Log.i(
            TAG,
            "run: complete clusters=${resultClusters.size} totalTime=" +
                "${SystemClock.elapsedRealtime() - phaseStart}ms",
        )
        emit(Event.Done(resultClusters))
    }.flowOn(Dispatchers.Default)

    /**
     * Pick the most cluster-representative face. Heuristic for v1: prefer the largest
     * available natural display crop (largest face = closest to the camera = best chance of a
     * recognisable thumbnail), falling back to the first member if none have a crop.
     */
    private fun pickRepresentative(members: List<FaceProcessor.FaceRecord>): FaceProcessor.FaceRecord =
        members
            .filter { it.displayCrop != null }
            .maxByOrNull { (it.displayCrop?.width ?: 0) * (it.displayCrop?.height ?: 0) }
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
        private const val TAG: String = "FaceMesh.Clusterify"
        const val REPRESENTATIVE_DIR: String = "representatives"
    }
}
