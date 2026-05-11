package com.alifesoftware.facemesh.domain

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig
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
        val epsSource = preferences.dbscanEpsSource.first()
        val minPts = preferences.dbscanMinPts.first()
        val createdAt = System.currentTimeMillis()
        Log.i(
            TAG,
            "run: config eps=$eps(source=$epsSource) minPts=$minPts(source=MANIFEST_OR_DEFAULT) " +
                "createdAt=$createdAt",
        )
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

        val incrementalPref = preferences.incrementalClusterMergeIntoExisting.first()
        val persistedPairs = if (incrementalPref) {
            clusterRepository.loadPersistedClustersForIncrementalMerge()
        } else {
            emptyList()
        }
        val incrementalActive = incrementalPref && persistedPairs.isNotEmpty()
        val matchThreshold = preferences.matchThreshold.first()
        val matchSource = preferences.matchThresholdSource.first()
        val ambiguityMargin = PipelineConfig.IncrementalClusterify.centroidAssignmentAmbiguityMargin
        Log.i(
            TAG,
            "run: incrementalMergePref=$incrementalPref persistedClusterCount=${persistedPairs.size} " +
                "incrementalMergeActive=$incrementalActive " +
                "matchThreshold=${"%.3f".format(matchThreshold)}(source=$matchSource) " +
                "ambiguityMargin=${"%.3f".format(ambiguityMargin)}",
        )

        val mergedIntoExisting = LinkedHashMap<String, MutableList<FaceProcessor.FaceRecord>>()
        val orphansFromMerge: List<FaceProcessor.FaceRecord> = if (incrementalActive) {
            assignToPersistedCentroids(
                records = records,
                persistedPairs = persistedPairs,
                matchThreshold = matchThreshold,
                ambiguityMargin = ambiguityMargin,
                groupedByClusterOut = mergedIntoExisting,
            )
        } else {
            records
        }

        Log.i(
            TAG,
            "run: after incremental pass mergedClusterIds=${mergedIntoExisting.size} " +
                "orphansForDbscan=${orphansFromMerge.size} (incremental=$incrementalActive)",
        )

        val appendRecover = ArrayList<FaceProcessor.FaceRecord>()
        val updatedDomains = ArrayList<Cluster>(mergedIntoExisting.size)
        for ((clusterId, faces) in mergedIntoExisting.entries) {
            if (faces.isEmpty()) continue
            val rows = faces.map { r ->
                ClusterImageEntity(
                    clusterId = clusterId,
                    imageUri = r.sourceUri.toString(),
                    faceIndex = r.faceIndex,
                    embedding = r.embedding,
                )
            }
            val domain = clusterRepository.appendFacesAndRecomputeCentroid(clusterId, rows)
            if (domain != null) {
                updatedDomains += domain
                Log.i(
                    TAG,
                    "run: incremental MERGED id=$clusterId newFaces=${faces.size} totalFaceCount=${domain.faceCount}",
                )
            } else {
                Log.e(
                    TAG,
                    "run: incremental FAILED append id=$clusterId newFaces=${faces.size}; " +
                        "re-queuing for DBSCAN subset",
                )
                appendRecover.addAll(faces)
            }
        }

        val dbscanSubsetSize = orphansFromMerge.size + appendRecover.size
        Log.i(
            TAG,
            "run: DBSCAN subset faces=$dbscanSubsetSize " +
                "(unmatchedIncremental=${orphansFromMerge.size} recoveredFromFailedAppend=${appendRecover.size})",
        )
        val newFromDbscan = persistDbscanNewClusters(
            subset = orphansFromMerge + appendRecover,
            eps = eps,
            minPts = minPts,
            createdAt = createdAt,
        )

        val resultClusters = ArrayList<Cluster>(updatedDomains.size + newFromDbscan.size)
        resultClusters.addAll(updatedDomains)
        resultClusters.addAll(newFromDbscan)

        if (resultClusters.isEmpty()) {
            Log.i(TAG, "run: no incremental merges and DBSCAN yielded no clusters -> emitting NoFaces")
            records.forEach { it.displayCrop?.recycle() }
            emit(Event.NoFaces)
            return@flow
        }

        // Recycle held display crops we already persisted.
        records.forEach { it.displayCrop?.recycle() }

        Log.i(
            TAG,
            "run: complete updated=${updatedDomains.size} newDbscan=${newFromDbscan.size} " +
                "totalEmitted=${resultClusters.size} wallTime=${SystemClock.elapsedRealtime() - phaseStart}ms",
        )
        emit(Event.Done(resultClusters))
    }.flowOn(Dispatchers.Default)

    /**
     * Centroid-greedy assignment of new [records] into persisted clusters. Unmatched faces
     * are returned for DBSCAN. Side effect: populates [groupedByClusterOut] with merge lists.
     */
    private fun assignToPersistedCentroids(
        records: List<FaceProcessor.FaceRecord>,
        persistedPairs: List<Pair<String, FloatArray>>,
        matchThreshold: Float,
        ambiguityMargin: Float,
        groupedByClusterOut: MutableMap<String, MutableList<FaceProcessor.FaceRecord>>,
    ): List<FaceProcessor.FaceRecord> {
        val unmatched = ArrayList<FaceProcessor.FaceRecord>()
        records.forEachIndexed { rIdx, r ->
            val scored = persistedPairs.mapIndexed { cIdx, (id, cent) ->
                Triple(id, cIdx, EmbeddingMath.dot(r.embedding, cent))
            }.sortedByDescending { it.third }
            val best = scored[0]
            if (best.third < matchThreshold) {
                Log.i(
                    TAG,
                    "incrementalAssign: face[$rIdx] uri=${r.sourceUri} faceIdx=${r.faceIndex} " +
                        "DROP below threshold bestClusterId=${best.first} centroid[${best.second}] " +
                        "score=${"%.3f".format(best.third)} < ${"%.3f".format(matchThreshold)}",
                )
                unmatched += r
                return@forEachIndexed
            }
            val runnerDistinct = scored.firstOrNull { it.first != best.first }
            if (runnerDistinct != null && (best.third - runnerDistinct.third) < ambiguityMargin) {
                Log.i(
                    TAG,
                    "incrementalAssign: face[$rIdx] uri=${r.sourceUri} faceIdx=${r.faceIndex} " +
                        "DROP ambiguous bestId=${best.first} score=${"%.3f".format(best.third)} " +
                        "runnerId=${runnerDistinct.first} runnerScore=${"%.3f".format(runnerDistinct.third)} " +
                        "delta=${"%.3f".format(best.third - runnerDistinct.third)} < margin=$ambiguityMargin",
                )
                unmatched += r
                return@forEachIndexed
            }
            Log.i(
                TAG,
                "incrementalAssign: face[$rIdx] uri=${r.sourceUri} faceIdx=${r.faceIndex} PASS " +
                    "mergedInto id=${best.first} centroid[${best.second}] score=${"%.3f".format(best.third)} " +
                    "(threshold=$matchThreshold runnerUp=${runnerDistinct?.let { "${it.first}@${"%.3f".format(it.third)}" } ?: "none"})",
            )
            groupedByClusterOut.getOrPut(best.first) { ArrayList() }.add(r)
        }
        return unmatched
    }

    private suspend fun persistDbscanNewClusters(
        subset: List<FaceProcessor.FaceRecord>,
        eps: Float,
        minPts: Int,
        createdAt: Long,
    ): List<Cluster> {
        if (subset.isEmpty()) {
            Log.i(TAG, "persistDbscanNewClusters: empty subset → skip DBSCAN")
            return emptyList()
        }
        val dbscanStarted = SystemClock.elapsedRealtime()
        val labels = Dbscan(eps = eps, minPts = minPts).run(subset.map { it.embedding })
        Log.i(
            TAG,
            "persistDbscanNewClusters: subset=${subset.size} DBSCAN in " +
                "${SystemClock.elapsedRealtime() - dbscanStarted}ms " +
                "noise=${labels.count { it == Dbscan.NOISE }}",
        )
        val grouped = LinkedHashMap<Int, MutableList<FaceProcessor.FaceRecord>>()
        labels.forEachIndexed { idx, label ->
            if (label == Dbscan.NOISE) return@forEachIndexed
            grouped.getOrPut(label) { ArrayList() }.add(subset[idx])
        }
        if (grouped.isEmpty()) {
            Log.i(TAG, "persistDbscanNewClusters: only noise after DBSCAN → 0 new clusters")
            return emptyList()
        }
        val resultClusters = ArrayList<Cluster>(grouped.size)
        for ((label, members) in grouped) {
            Log.i(
                TAG,
                "persistDbscanNewClusters: cluster label=$label members=${members.size} " +
                    "sourceUris=${members.map { it.sourceUri }}",
            )
            val centroid = EmbeddingMath.meanAndNormalize(members.map { it.embedding })
            val rep = pickRepresentative(members)
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
                "persistDbscanNewClusters: persisted label=$label id=$clusterId members=${members.size} " +
                    "thumbFromCrop=${rep.displayCrop != null} thumbUri=$savedThumbUri",
            )
        }
        return resultClusters
    }

    /**
     * Pick the most cluster-representative face. Heuristic for v1: prefer the largest
     * available natural display crop (largest face = closest to the camera = best chance of a
     * recognisable thumbnail), falling back to the first member if none have a crop.
     */
    private fun pickRepresentative(members: List<FaceProcessor.FaceRecord>): FaceProcessor.FaceRecord {
        val withCrop = members.filter { it.displayCrop != null }
        if (withCrop.isEmpty()) {
            Log.w(
                TAG,
                "pickRepresentative: no members have displayCrop; falling back to members[0] " +
                    "sourceUri=${members.first().sourceUri}",
            )
            return members.first()
        }
        val winner = withCrop.maxByOrNull {
            (it.displayCrop?.width ?: 0) * (it.displayCrop?.height ?: 0)
        }!!
        val area = (winner.displayCrop?.width ?: 0) * (winner.displayCrop?.height ?: 0)
        Log.i(
            TAG,
            "pickRepresentative: chose ${winner.sourceUri} faceIdx=${winner.faceIndex} " +
                "cropArea=$area (${withCrop.size}/${members.size} candidates with crops)",
        )
        return winner
    }

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
