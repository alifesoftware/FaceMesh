package com.alifesoftware.facemesh.ml

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import com.google.android.gms.tasks.Tasks
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.File
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wraps the Play-Services-provided TFLite runtime (SPEC \u00a78.3 / NFR-02).
 *
 *   1. Initialises Play Services TFLite once.
 *   2. Detects GPU delegate availability; falls back to XNNPACK CPU when unavailable.
 *   3. Builds [InterpreterApi] instances from on-disk model files (memory-mapped for zero-copy
 *      load).
 *
 * Phase 4 / 5 use this to construct the BlazeFace and GhostFaceNet interpreters.
 */
class TfLiteRuntime private constructor(
    private val gpuAvailable: Boolean,
) {

    enum class Delegate { GPU, XNNPACK }

    val activeDelegate: Delegate = if (gpuAvailable) Delegate.GPU else Delegate.XNNPACK

    /** Memory-map a `.tflite` file into a [MappedByteBuffer] required by [InterpreterApi.create]. */
    fun loadModel(file: File): MappedByteBuffer {
        require(file.exists() && file.length() > 0) { "Model file missing or empty: ${file.absolutePath}" }
        Log.i(TAG, "loadModel: mmap ${file.absolutePath} (${file.length()} bytes)")
        return file.inputStream().channel.use { channel ->
            channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
    }

    /**
     * Build an interpreter for the given model bytes.
     *
     * Caller is responsible for closing the returned [InterpreterApi] (typically wrapped in a
     * `use` block or held by a long-lived component such as [FaceDetector] / [FaceEmbedder]).
     */
    fun newInterpreter(model: ByteBuffer, numThreads: Int = 2): InterpreterApi {
        val options = InterpreterApi.Options().apply {
            setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            if (gpuAvailable) {
                addDelegateFactory(GpuDelegateFactory())
            } else {
                setNumThreads(numThreads)
            }
        }
        val started = SystemClock.elapsedRealtime()
        val interpreter = InterpreterApi.create(model, options)
        Log.i(
            TAG,
            "newInterpreter: created delegate=$activeDelegate " +
                "${if (!gpuAvailable) "threads=$numThreads " else ""}in ${SystemClock.elapsedRealtime() - started}ms",
        )
        return interpreter
    }

    companion object {
        private const val TAG: String = "FaceMesh.TfLite"

        @Volatile private var instance: TfLiteRuntime? = null

        /**
         * Synchronously initialises Play Services TFLite. MUST be called from a background
         * dispatcher (typically [kotlinx.coroutines.Dispatchers.IO]); blocks on the underlying
         * [com.google.android.gms.tasks.Task].
         */
        suspend fun initialise(context: Context): TfLiteRuntime {
            instance?.let {
                Log.i(TAG, "initialise: returning cached runtime delegate=${it.activeDelegate}")
                return it
            }
            val started = SystemClock.elapsedRealtime()
            Log.i(TAG, "initialise: probing GPU delegate availability...")
            val gpuAvailable = checkGpuSupport(context)
            Log.i(TAG, "initialise: gpuDelegateAvailable=$gpuAvailable")
            val options = TfLiteInitializationOptions.builder()
                .setEnableGpuDelegateSupport(gpuAvailable)
                .build()
            Tasks.await(TfLite.initialize(context.applicationContext, options))
            val rt = TfLiteRuntime(gpuAvailable).also { instance = it }
            Log.i(
                TAG,
                "initialise: Play Services TFLite ready delegate=${rt.activeDelegate} " +
                    "took=${SystemClock.elapsedRealtime() - started}ms",
            )
            return rt
        }

        /** Best-effort check; if Play Services GPU module isn't available we fall back. */
        private suspend fun checkGpuSupport(context: Context): Boolean = runCatching {
            Tasks.await(TfLiteGpu.isGpuDelegateAvailable(context.applicationContext))
        }.onFailure {
            Log.w(TAG, "checkGpuSupport: probe failed, falling back to CPU/XNNPACK", it)
        }.getOrDefault(false)
    }
}
