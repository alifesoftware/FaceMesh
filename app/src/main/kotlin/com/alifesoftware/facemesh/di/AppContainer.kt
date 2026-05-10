package com.alifesoftware.facemesh.di

import android.content.Context
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

    val database: AppDatabase by lazy { AppDatabase.build(appCtx) }
    val preferences: AppPreferences by lazy { AppPreferences(appCtx) }
    val clusterRepository: ClusterRepository by lazy { ClusterRepository(database.clusterDao()) }

    val modelStore: ModelStore by lazy { ModelStore(appCtx) }
    val modelDownloadManager: ModelDownloadManager by lazy {
        ModelDownloadManager(
            store = modelStore,
            preferences = preferences,
            baseUrl = BuildConfig.MODEL_BASE_URL,
        )
    }

    val mlPipelineProvider: MlPipelineProvider by lazy {
        MlPipelineProvider(
            context = appCtx,
            store = modelStore,
            clusterRepository = clusterRepository,
            preferences = preferences,
        )
    }
}
