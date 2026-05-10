package com.alifesoftware.facemesh.data

import android.content.Context
import android.os.SystemClock
import android.util.Log
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
        private const val TAG: String = "FaceMesh.Db"
        const val NAME: String = "facemesh.db"

        fun build(context: Context): AppDatabase {
            val started = SystemClock.elapsedRealtime()
            val dbFile = context.getDatabasePath(NAME)
            Log.i(
                TAG,
                "build: opening Room database name=$NAME path=${dbFile.absolutePath} " +
                    "exists=${dbFile.exists()} sizeBeforeOpen=${if (dbFile.exists()) dbFile.length() else 0L}B",
            )
            val db = Room
                .databaseBuilder(context.applicationContext, AppDatabase::class.java, NAME)
                .fallbackToDestructiveMigration()
                .build()
            Log.i(TAG, "build: Room database ready in ${SystemClock.elapsedRealtime() - started}ms")
            return db
        }
    }
}
