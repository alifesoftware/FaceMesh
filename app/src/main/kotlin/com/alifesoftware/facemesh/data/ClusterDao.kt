package com.alifesoftware.facemesh.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import kotlinx.coroutines.flow.Flow

@Dao
interface ClusterDao {

    @Query("SELECT * FROM cluster ORDER BY createdAt ASC")
    fun observeClusters(): Flow<List<ClusterEntity>>

    @Query("SELECT * FROM cluster ORDER BY createdAt ASC")
    suspend fun getAllClusters(): List<ClusterEntity>

    @Query("SELECT * FROM cluster WHERE id = :id LIMIT 1")
    suspend fun findById(id: String): ClusterEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertCluster(cluster: ClusterEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertClusterImages(images: List<ClusterImageEntity>)

    @Transaction
    suspend fun saveClusterWithImages(cluster: ClusterEntity, images: List<ClusterImageEntity>) {
        upsertCluster(cluster)
        if (images.isNotEmpty()) upsertClusterImages(images)
    }

    @Query("DELETE FROM cluster WHERE id = :id")
    suspend fun deleteCluster(id: String)

    @Query("DELETE FROM cluster")
    suspend fun deleteAllClusters()

    @Query("SELECT * FROM cluster_image WHERE clusterId = :clusterId")
    suspend fun getImagesForCluster(clusterId: String): List<ClusterImageEntity>
}
