package com.alifesoftware.facemesh.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Persistence layer entity for a face cluster.
 *
 * The 128-d centroid is stored as a little-endian `FloatArray` blob via [FloatArrayConverter].
 * UI-facing model lives in [com.alifesoftware.facemesh.domain.model.Cluster].
 */
@Entity(tableName = "cluster")
data class ClusterEntity(
    @PrimaryKey val id: String,
    val centroid: FloatArray,
    val representativeImageUri: String,
    val faceCount: Int,
    val createdAt: Long,
    val name: String? = null,
) {
    // Generated equals/hashCode for FloatArray-bearing data classes (Room/Kotlin recommendation).
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ClusterEntity) return false
        if (id != other.id) return false
        if (!centroid.contentEquals(other.centroid)) return false
        if (representativeImageUri != other.representativeImageUri) return false
        if (faceCount != other.faceCount) return false
        if (createdAt != other.createdAt) return false
        if (name != other.name) return false
        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + centroid.contentHashCode()
        result = 31 * result + representativeImageUri.hashCode()
        result = 31 * result + faceCount
        result = 31 * result + createdAt.hashCode()
        result = 31 * result + (name?.hashCode() ?: 0)
        return result
    }
}

/**
 * Persistence layer entity for a single face that contributed to a cluster.
 *
 * Composite primary key allows the same source image URI to legitimately contribute multiple
 * faces (group shots) to the same or different clusters.
 */
@Entity(
    tableName = "cluster_image",
    primaryKeys = ["clusterId", "imageUri", "faceIndex"],
    foreignKeys = [
        ForeignKey(
            entity = ClusterEntity::class,
            parentColumns = ["id"],
            childColumns = ["clusterId"],
            onDelete = ForeignKey.CASCADE,
        )
    ],
    indices = [Index("clusterId")],
)
data class ClusterImageEntity(
    val clusterId: String,
    val imageUri: String,
    val faceIndex: Int,
    val embedding: FloatArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ClusterImageEntity) return false
        if (clusterId != other.clusterId) return false
        if (imageUri != other.imageUri) return false
        if (faceIndex != other.faceIndex) return false
        if (!embedding.contentEquals(other.embedding)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = clusterId.hashCode()
        result = 31 * result + imageUri.hashCode()
        result = 31 * result + faceIndex
        result = 31 * result + embedding.contentHashCode()
        return result
    }
}
