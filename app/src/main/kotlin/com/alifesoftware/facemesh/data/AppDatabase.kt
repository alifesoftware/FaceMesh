package com.alifesoftware.facemesh.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

@Database(
    entities = [ClusterEntity::class, ClusterImageEntity::class],
    version = 1,
    exportSchema = true,
)
@TypeConverters(FloatArrayConverter::class)
abstract class AppDatabase : RoomDatabase() {

    abstract fun clusterDao(): ClusterDao

    companion object {
        const val NAME: String = "facemesh.db"

        fun build(context: Context): AppDatabase = Room
            .databaseBuilder(context.applicationContext, AppDatabase::class.java, NAME)
            .fallbackToDestructiveMigration()
            .build()
    }
}
