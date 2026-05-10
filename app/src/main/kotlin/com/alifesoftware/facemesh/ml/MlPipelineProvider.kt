package com.alifesoftware.facemesh.ml

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig.Detector.DetectorVariant
import com.alifesoftware.facemesh.data.AppPreferences
import com.alifesoftware.facemesh.data.ClusterRepository
import com.alifesoftware.facemesh.domain.ClusterifyUseCase
import com.alifesoftware.facemesh.domain.FilterAgainstClustersUseCase
import com.alifesoftware.facemesh.ml.download.ModelDescriptor
import com.alifesoftware.facemesh.ml.download.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Constructs and caches the heavy ML graph (TFLite runtime + interpreters + use cases) once
 * the model bundle is on disk. Safe to call concurrently; first caller pays the init cost.
 *
 * Held by the [com.alifesoftware.facemesh.di.AppContainer].
 *
 * ## Detector variant dispatch
 *
 * The face detector is one of two BlazeFace variants - short-range or full-range - chosen
 * by the user in Settings (see
 * [com.alifesoftware.facemesh.config.PipelineConfig.Detector.DetectorVariant]). On every
 * call to [clusterifyUseCase] / [filterUseCase] this provider reads the current variant
 * preference; if it differs from the cached detector's variant the cached detector is
 * closed and a new one is built. The aligner, embedder, and TfLite runtime are reused
 * across variant switches because they are detector-independent.
 */
class MlPipelineProvider(
    private val context: Context,
    private val store: ModelStore,
    private val clusterRepository: ClusterRepository,
    private val preferences: AppPreferences,
) {

    private val mutex = Mutex()
    @Volatile private var runtime: TfLiteRuntime? = null
    @Volatile private var detector: FaceDetector? = null
    @Volatile private var detectorVariant: DetectorVariant? = null
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
        // Read the requested variant outside the lock so concurrent callers don't all block
        // on the DataStore read; the inside-lock check below detects any drift.
        val requestedVariant = preferences.detectorVariant.first()
        val (detLocal, alignLocal, embedLocal) = mutex.withLock {
            // Build (or rebuild) detector if it's missing OR the requested variant differs
            // from what was cached. Aligner + embedder are variant-independent so they only
            // need to be built once.
            val variantChanged = detectorVariant != null && detectorVariant != requestedVariant
            if (detector == null || variantChanged || embedder == null || aligner == null) {
                val started = SystemClock.elapsedRealtime()
                if (variantChanged) {
                    Log.i(
                        TAG,
                        "ensureProcessor: detector variant change " +
                            "$detectorVariant -> $requestedVariant; closing old detector",
                    )
                    runCatching { detector?.close() }
                        .onFailure { Log.w(TAG, "ensureProcessor: old detector.close() threw", it) }
                    detector = null
                } else if (detector == null) {
                    Log.i(TAG, "ensureProcessor: building ML graph (first call) variant=$requestedVariant")
                }
                val rt = runtime ?: TfLiteRuntime.initialise(context).also { runtime = it }
                val manifest = store.readManifest()
                    ?: error("Model manifest missing; ensureAvailable() must succeed first")

                if (detector == null) {
                    detector = buildDetector(rt, manifest, requestedVariant)
                    detectorVariant = requestedVariant
                }
                if (embedder == null) {
                    val emb = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_EMBEDDER }
                        ?: error("Manifest missing entry of type=${ModelDescriptor.TYPE_EMBEDDER}")
                    embedder = FaceEmbedder(
                        runtime = rt,
                        modelFile = store.fileFor(emb),
                        inputSize = manifest.config.embedderInput[0],
                    )
                }
                if (aligner == null) {
                    aligner = FaceAligner(outputSize = manifest.config.embedderInput[0])
                }
                Log.i(
                    TAG,
                    "ensureProcessor: graph ready in ${SystemClock.elapsedRealtime() - started}ms " +
                        "delegate=${rt.activeDelegate} variant=$requestedVariant",
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

    private fun buildDetector(
        rt: TfLiteRuntime,
        manifest: com.alifesoftware.facemesh.ml.download.ModelManifest,
        variant: DetectorVariant,
    ): FaceDetector = when (variant) {
        DetectorVariant.SHORT_RANGE -> {
            val det = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_DETECTOR_SHORT_RANGE }
                ?: error("Manifest missing entry of type=${ModelDescriptor.TYPE_DETECTOR_SHORT_RANGE}")
            Log.i(
                TAG,
                "buildDetector: variant=SHORT_RANGE manifest=${det.name} input=${manifest.config.shortRangeDetectorInput.toList()}",
            )
            BlazeFaceShortRangeDetector(
                runtime = rt,
                modelFile = store.fileFor(det),
                inputSize = manifest.config.shortRangeDetectorInput[0],
            )
        }
        DetectorVariant.FULL_RANGE -> {
            val det = manifest.models.firstOrNull { it.type == ModelDescriptor.TYPE_DETECTOR_FULL_RANGE }
                ?: error(
                    "Manifest missing entry of type=${ModelDescriptor.TYPE_DETECTOR_FULL_RANGE}; " +
                        "this manifest is from a previous app version that didn't ship the full-range " +
                        "detector. Upgrade the manifest (run ensureAvailable on a fresh URL) or fall " +
                        "back to SHORT_RANGE in Settings.",
                )
            Log.i(
                TAG,
                "buildDetector: variant=FULL_RANGE manifest=${det.name} input=${manifest.config.fullRangeDetectorInput.toList()}",
            )
            BlazeFaceFullRangeDetector(
                runtime = rt,
                modelFile = store.fileFor(det),
                inputSize = manifest.config.fullRangeDetectorInput[0],
            )
        }
    }

    /** Used by [com.alifesoftware.facemesh.domain.ResetAppUseCase] when the user wipes everything. */
    fun close() {
        Log.i(TAG, "close: tearing down ML graph (detector + embedder + aligner)")
        runCatching { detector?.close() }
            .onFailure { Log.w(TAG, "close: detector.close() threw", it) }
        runCatching { embedder?.close() }
            .onFailure { Log.w(TAG, "close: embedder.close() threw", it) }
        detector = null
        detectorVariant = null
        embedder = null
        aligner = null
        runtime = null
    }

    companion object {
        private const val TAG: String = "FaceMesh.Pipeline"
    }
}
