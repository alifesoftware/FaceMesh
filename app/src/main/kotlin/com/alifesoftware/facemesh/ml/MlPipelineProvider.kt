package com.alifesoftware.facemesh.ml

import android.content.Context
import android.os.SystemClock
import android.util.Log
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
        // Capture local refs INSIDE the mutex so a concurrent close() that nulls the fields
        // can't sneak in between the lock release and the FaceProcessor constructor reads.
        // Volatile alone wouldn't help here - only the mutex serialises build vs. teardown.
        val (detLocal, alignLocal, embedLocal) = mutex.withLock {
            if (detector == null || embedder == null || aligner == null) {
                val started = SystemClock.elapsedRealtime()
                Log.i(TAG, "ensureProcessor: building ML graph (first call)")
                val rt = runtime ?: TfLiteRuntime.initialise(context).also { runtime = it }
                val manifest = store.readManifest()
                    ?: error("Model manifest missing; ensureAvailable() must succeed first")
                val det = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_DETECTOR }
                    ?: error("Manifest missing entry of type=${ModelDescriptor.TYPE_DETECTOR}")
                val emb = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_EMBEDDER }
                    ?: error("Manifest missing entry of type=${ModelDescriptor.TYPE_EMBEDDER}")
                Log.i(
                    TAG,
                    "ensureProcessor: manifest v${manifest.version} delegate=${rt.activeDelegate} " +
                        "detector=${det.name} embedder=${emb.name}",
                )
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
                Log.i(
                    TAG,
                    "ensureProcessor: graph ready in ${SystemClock.elapsedRealtime() - started}ms",
                )
            }
            Triple(detector!!, aligner!!, embedder!!)
        }
        return FaceProcessor(
            resolver = context.contentResolver,
            detector = detLocal,
            aligner = alignLocal,
            embedder = embedLocal,
        )
    }

    /** Used by [com.alifesoftware.facemesh.domain.ResetAppUseCase] when the user wipes everything. */
    fun close() {
        Log.i(TAG, "close: tearing down ML graph (detector + embedder + aligner)")
        runCatching { detector?.close() }
            .onFailure { Log.w(TAG, "close: detector.close() threw", it) }
        runCatching { embedder?.close() }
            .onFailure { Log.w(TAG, "close: embedder.close() threw", it) }
        detector = null
        embedder = null
        aligner = null
        runtime = null
    }

    companion object {
        private const val TAG: String = "FaceMesh.Pipeline"
    }
}
