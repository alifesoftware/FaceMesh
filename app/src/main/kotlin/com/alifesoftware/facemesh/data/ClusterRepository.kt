package com.alifesoftware.facemesh.data

import android.net.Uri
import com.alifesoftware.facemesh.domain.model.Cluster
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Repository surface for clusters. Translates between the DB-layer [ClusterEntity] and the
 * UI-layer [Cluster] domain model.
 */
class ClusterRepository(private val dao: ClusterDao) {

    fun observeClusters(): Flow<List<Cluster>> = dao.observeClusters().map { list ->
        list.map { it.toDomain() }
    }

    suspend fun loadClusters(): List<Cluster> = dao.getAllClusters().map { it.toDomain() }

    suspend fun loadCentroidsForIds(ids: Set<String>): List<Pair<String, FloatArray>> {
        if (ids.isEmpty()) return emptyList()
        return dao.getAllClusters()
            .filter { it.id in ids }
            .map { it.id to it.centroid }
    }

    suspend fun saveCluster(
        cluster: Cluster,
        centroid: FloatArray,
        contributingFaces: List<ClusterImageEntity>,
        createdAt: Long,
    ) {
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
    }

    suspend fun deleteCluster(id: String) = dao.deleteCluster(id)

    suspend fun deleteAll() = dao.deleteAllClusters()

    private fun ClusterEntity.toDomain(): Cluster = Cluster(
        id = id,
        representativeImageUri = Uri.parse(representativeImageUri),
        faceCount = faceCount,
        name = name,
    )
}
