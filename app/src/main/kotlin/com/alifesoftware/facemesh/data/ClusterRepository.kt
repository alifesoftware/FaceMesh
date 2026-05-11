package com.alifesoftware.facemesh.data

import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.domain.model.Cluster
import com.alifesoftware.facemesh.ml.cluster.EmbeddingMath
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onEach

/**
 * Repository surface for clusters. Translates between the DB-layer [ClusterEntity] and the
 * UI-layer [Cluster] domain model.
 */
class ClusterRepository(private val dao: ClusterDao) {

    fun observeClusters(): Flow<List<Cluster>> = dao.observeClusters()
        .onEach { Log.i(TAG, "observeClusters: emit n=${it.size} ids=${it.map { c -> c.id }}") }
        .map { list -> list.map { it.toDomain() } }

    suspend fun loadClusters(): List<Cluster> {
        val started = SystemClock.elapsedRealtime()
        val rows = dao.getAllClusters()
        val mapped = rows.map { it.toDomain() }
        Log.i(
            TAG,
            "loadClusters: read ${rows.size} row(s) in ${SystemClock.elapsedRealtime() - started}ms " +
                "ids=${rows.map { it.id }}",
        )
        return mapped
    }

    suspend fun loadCentroidsForIds(ids: Set<String>): List<Pair<String, FloatArray>> {
        if (ids.isEmpty()) {
            Log.i(TAG, "loadCentroidsForIds: empty id set -> early return empty")
            return emptyList()
        }
        val started = SystemClock.elapsedRealtime()
        val all = dao.getAllClusters()
        val matched = all.filter { it.id in ids }.map { it.id to it.centroid }
        Log.i(
            TAG,
            "loadCentroidsForIds: requested=${ids.size} totalInDb=${all.size} " +
                "matched=${matched.size} took=${SystemClock.elapsedRealtime() - started}ms " +
                "centroidDim=${matched.firstOrNull()?.second?.size ?: 0}",
        )
        if (matched.size < ids.size) {
            val missing = ids - matched.map { it.first }.toSet()
            Log.w(TAG, "loadCentroidsForIds: ${missing.size} requested id(s) NOT in DB: $missing")
        }
        return matched
    }

    suspend fun saveCluster(
        cluster: Cluster,
        centroid: FloatArray,
        contributingFaces: List<ClusterImageEntity>,
        createdAt: Long,
    ) {
        val started = SystemClock.elapsedRealtime()
        Log.i(
            TAG,
            "saveCluster: id=${cluster.id} faceCount=${cluster.faceCount} " +
                "centroidDim=${centroid.size} contributingFaces=${contributingFaces.size} " +
                "thumbUri=${cluster.representativeImageUri} createdAt=$createdAt",
        )
        dao.saveClusterWithImages(
            cluster = ClusterEntity(
                id = cluster.id,
                centroid = centroid,
                representativeImageUri = cluster.representativeImageUri.toString(),
                faceCount = cluster.faceCount,
                createdAt = createdAt,
                name = cluster.name,
            ),
            images = contributingFaces,
        )
        Log.i(TAG, "saveCluster: id=${cluster.id} persisted in ${SystemClock.elapsedRealtime() - started}ms")
    }

    /**
     * Returns the distinct source-photo URIs that contributed faces to [clusterId], in
     * insertion order. A single source photo can contribute multiple faces (group shots) and
     * we de-duplicate so the gallery doesn't show the same photo twice.
     *
     * Empty list when the cluster id isn't known or has no images yet.
     */
    suspend fun loadImagesForCluster(clusterId: String): List<Uri> {
        val started = SystemClock.elapsedRealtime()
        val rows = dao.getImagesForCluster(clusterId)
        // De-dupe by imageUri, preserving insertion order.
        val unique = LinkedHashSet<String>(rows.size)
        for (row in rows) unique += row.imageUri
        val uris = unique.map(Uri::parse)
        Log.i(
            TAG,
            "loadImagesForCluster: id=$clusterId rows=${rows.size} uniquePhotos=${uris.size} " +
                "took=${SystemClock.elapsedRealtime() - started}ms",
        )
        return uris
    }

    suspend fun deleteCluster(id: String) {
        val started = SystemClock.elapsedRealtime()
        Log.i(TAG, "deleteCluster: id=$id")
        dao.deleteCluster(id)
        Log.i(TAG, "deleteCluster: id=$id done in ${SystemClock.elapsedRealtime() - started}ms")
    }

    suspend fun deleteAll() {
        val started = SystemClock.elapsedRealtime()
        Log.w(TAG, "deleteAll: wiping ALL clusters (Reset flow)")
        dao.deleteAllClusters()
        Log.i(TAG, "deleteAll: completed in ${SystemClock.elapsedRealtime() - started}ms")
    }

    /**
     * Returns persisted clusters with centroids, ordered like [ClusterDao.getAllClusters]
     * (oldest-first by [ClusterEntity.createdAt]) for deterministic merge logs.
     */
    suspend fun loadPersistedClustersForIncrementalMerge(): List<Pair<String, FloatArray>> {
        val started = SystemClock.elapsedRealtime()
        val rows = dao.getAllClusters()
        val out = rows.map { it.id to it.centroid.clone() }
        Log.i(
            TAG,
            "loadPersistedClustersForIncrementalMerge: n=${out.size} " +
                "took=${SystemClock.elapsedRealtime() - started}ms dim=${rows.firstOrNull()?.centroid?.size ?: 0}",
        )
        return out
    }

    /**
     * Appends [newFaces] to [clusterId], recomputes the centroid over **all** stored embeddings,
     * bumps [ClusterEntity.faceCount], and persists. Representative URI and name unchanged.
     *
     * @return Domain [Cluster] after update, or `null` if [clusterId] is unknown.
     */
    suspend fun appendFacesAndRecomputeCentroid(
        clusterId: String,
        newFaces: List<ClusterImageEntity>,
    ): Cluster? {
        if (newFaces.isEmpty()) {
            Log.w(TAG, "appendFacesAndRecomputeCentroid: id=$clusterId early return (newFaces empty)")
            return null
        }
        val started = SystemClock.elapsedRealtime()
        val header = dao.findById(clusterId)
        if (header == null) {
            Log.w(TAG, "appendFacesAndRecomputeCentroid: id=$clusterId NOT FOUND -> null")
            return null
        }
        val existing = dao.getImagesForCluster(clusterId)
        val embeddings = ArrayList<FloatArray>(existing.size + newFaces.size)
        for (r in existing) embeddings += r.embedding
        for (r in newFaces) embeddings += r.embedding
        val centroid = EmbeddingMath.meanAndNormalize(embeddings)
        Log.i(
            TAG,
            "appendFacesAndRecomputeCentroid: id=$clusterId prevFaces=${existing.size} " +
                "adding=${newFaces.size} total=${embeddings.size} dim=${centroid.size}",
        )
        val updated = ClusterEntity(
            id = header.id,
            centroid = centroid,
            representativeImageUri = header.representativeImageUri,
            faceCount = embeddings.size,
            createdAt = header.createdAt,
            name = header.name,
        )
        dao.saveClusterWithImages(cluster = updated, images = newFaces)
        val domain = Cluster(
            id = updated.id,
            representativeImageUri = Uri.parse(updated.representativeImageUri),
            faceCount = updated.faceCount,
            name = updated.name,
        )
        Log.i(
            TAG,
            "appendFacesAndRecomputeCentroid: id=$clusterId done in ${SystemClock.elapsedRealtime() - started}ms",
        )
        return domain
    }

    private fun ClusterEntity.toDomain(): Cluster = Cluster(
        id = id,
        representativeImageUri = Uri.parse(representativeImageUri),
        faceCount = faceCount,
        name = name,
    )

    companion object {
        private const val TAG: String = "FaceMesh.Repo"
    }
}
