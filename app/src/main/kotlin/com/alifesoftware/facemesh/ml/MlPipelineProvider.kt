package com.alifesoftware.facemesh.ml

import android.content.Context
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.data.ClusterRepository
import com.alifesoftware.facemesh.domain.ClusterifyUseCase
import com.alifesoftware.facemesh.domain.FilterAgainstClustersUseCase
import com.alifesoftware.facemesh.ml.download.ModelDescriptor
import com.alifesoftware.facemesh.ml.download.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Constructs and caches the heavy ML graph (TFLite runtime + interpreters + use cases) once
 * the model bundle is on disk. Safe to call concurrently; first caller pays the init cost.
 *
 * Held by the [com.alifesoftware.facemesh.di.AppContainer].
 */
class MlPipelineProvider(
    private val context: Context,
    private val store: ModelStore,
    private val clusterRepository: ClusterRepository,
    private val preferences: AppPreferences,
) {

    private val mutex = Mutex()
    @Volatile private var runtime: TfLiteRuntime? = null
    @Volatile private var detector: BlazeFaceDetector? = null
    @Volatile private var embedder: FaceEmbedder? = null
    @Volatile private var aligner: FaceAligner? = null

    /** Once the runtime is initialised, this reflects the active delegate; otherwise null. */
    val activeDelegate: TfLiteRuntime.Delegate?
        get() = runtime?.activeDelegate

    suspend fun clusterifyUseCase(): ClusterifyUseCase = withContext(Dispatchers.IO) {
        val processor = ensureProcessor()
        ClusterifyUseCase(
            context = context,
            processor = processor,
            clusterRepository = clusterRepository,
            preferences = preferences,
        )
    }

    suspend fun filterUseCase(): FilterAgainstClustersUseCase = withContext(Dispatchers.IO) {
        val processor = ensureProcessor()
        FilterAgainstClustersUseCase(
            processor = processor,
            clusterRepository = clusterRepository,
            preferences = preferences,
        )
    }

    private suspend fun ensureProcessor(): FaceProcessor {
        mutex.withLock {
            if (detector == null || embedder == null || aligner == null) {
                val rt = runtime ?: TfLiteRuntime.initialise(context).also { runtime = it }
                val manifest = store.readManifest()
                    ?: error("Model manifest missing; ensureAvailable() must succeed first")
                val det = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_DETECTOR }
                    ?: error("Manifest missing entry of type=${ModelDescriptor.TYPE_DETECTOR}")
                val emb = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_EMBEDDER }
                    ?: error("Manifest missing entry of type=${ModelDescriptor.TYPE_EMBEDDER}")
                detector = BlazeFaceDetector(
                    runtime = rt,
                    modelFile = store.fileFor(det),
                    inputSize = manifest.config.detectorInput[0],
                )
                embedder = FaceEmbedder(
                    runtime = rt,
                    modelFile = store.fileFor(emb),
                    inputSize = manifest.config.embedderInput[0],
                )
                aligner = FaceAligner(outputSize = manifest.config.embedderInput[0])
            }
        }
        return FaceProcessor(
            resolver = context.contentResolver,
            detector = detector!!,
            aligner = aligner!!,
            embedder = embedder!!,
        )
    }

    /** Used by [com.alifesoftware.facemesh.domain.ResetAppUseCase] when the user wipes everything. */
    fun close() {
        runCatching { detector?.close() }
        runCatching { embedder?.close() }
        detector = null
        embedder = null
        aligner = null
        runtime = null
    }
}
