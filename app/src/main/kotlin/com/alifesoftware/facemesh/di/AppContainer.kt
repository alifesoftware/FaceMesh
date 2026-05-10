package com.alifesoftware.facemesh.di

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.BuildConfig
import com.alifesoftware.facemesh.data.AppDatabase
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.data.ClusterRepository
import com.alifesoftware.facemesh.ml.MlPipelineProvider
import com.alifesoftware.facemesh.ml.download.ModelDownloadManager
import com.alifesoftware.facemesh.ml.download.ModelStore

/**
 * Hand-rolled object graph. Instantiated once in [com.alifesoftware.facemesh.FaceMeshApplication]
 * and exposed through it. We intentionally avoid Hilt / Koin to stay inside the SPEC \u00a79.1
 * APK budget.
 *
 * Phases 4\u20136 will add: FaceDetector, FaceAligner, FaceEmbedder, ClusterifyUseCase,
 * FilterAgainstClustersUseCase. The TFLite runtime itself is initialised lazily by the
 * [com.alifesoftware.facemesh.viewmodel.HomeViewModel] on the first Clusterify tap.
 */
class AppContainer(applicationContext: Context) {

    val appCtx: Context = applicationContext.applicationContext

    val database: AppDatabase by lazy {
        val started = SystemClock.elapsedRealtime()
        Log.i(TAG, "lazy: building AppDatabase '${AppDatabase.NAME}' (Room)")
        AppDatabase.build(appCtx).also {
            Log.i(TAG, "lazy: AppDatabase ready in ${SystemClock.elapsedRealtime() - started}ms")
        }
    }
    val preferences: AppPreferences by lazy {
        Log.i(TAG, "lazy: constructing AppPreferences (DataStore 'facemesh_prefs')")
        AppPreferences(appCtx)
    }
    val clusterRepository: ClusterRepository by lazy {
        Log.i(TAG, "lazy: constructing ClusterRepository (will trigger AppDatabase build if not yet built)")
        ClusterRepository(database.clusterDao())
    }

    val modelStore: ModelStore by lazy {
        Log.i(TAG, "lazy: constructing ModelStore at filesDir/models")
        ModelStore(appCtx)
    }
    val modelDownloadManager: ModelDownloadManager by lazy {
        Log.i(TAG, "lazy: constructing ModelDownloadManager baseUrl=${BuildConfig.MODEL_BASE_URL}")
        ModelDownloadManager(
            store = modelStore,
            preferences = preferences,
            baseUrl = BuildConfig.MODEL_BASE_URL,
        )
    }

    val mlPipelineProvider: MlPipelineProvider by lazy {
        Log.i(TAG, "lazy: constructing MlPipelineProvider (TFLite graph deferred to first ensureProcessor)")
        MlPipelineProvider(
            context = appCtx,
            store = modelStore,
            clusterRepository = clusterRepository,
            preferences = preferences,
        )
    }

    init {
        Log.i(
            TAG,
            "init: AppContainer wiring complete (all components LAZY) baseUrl=${BuildConfig.MODEL_BASE_URL}",
        )
    }

    companion object {
        private const val TAG: String = "FaceMesh.DI"
    }
}
