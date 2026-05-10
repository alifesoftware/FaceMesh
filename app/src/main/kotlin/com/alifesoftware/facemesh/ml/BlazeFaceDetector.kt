package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.InterpreterApi
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * BlazeFace front-camera detector running on the [TfLiteRuntime].
 *
 * Lifecycle: build once, reuse across many frames; call [close] when done.
 */
class BlazeFaceDetector(
    private val runtime: TfLiteRuntime,
    modelFile: File,
    private val decoder: BlazeFaceDecoder = BlazeFaceDecoder(),
    private val inputSize: Int = 128,
) : Closeable {

    private val interpreter: InterpreterApi = runtime.newInterpreter(runtime.loadModel(modelFile))

    // BlazeFace exports list two outputs in different orders depending on the conversion
    // pipeline. We look them up by name to stay bulletproof against re-exports. Name match is
    // case-insensitive and tolerates the trailing colon-zero TF/Keras naming suffix.
    private val regOutputIndex: Int
    private val clsOutputIndex: Int

    init {
        Log.i(
            TAG,
            "init: building interpreter for ${modelFile.name} (${modelFile.length()}B) " +
                "path=${modelFile.absolutePath}",
        )
        val inputCount = interpreter.inputTensorCount
        val outputCount = interpreter.outputTensorCount
        Log.i(TAG, "init: tensor counts inputs=$inputCount outputs=$outputCount")
        for (i in 0 until inputCount) {
            val t = interpreter.getInputTensor(i)
            Log.i(
                TAG,
                "init: input[$i] name='${t.name()}' shape=${t.shape().toList()} dtype=${t.dataType()} " +
                    "bytes=${t.numBytes()}",
            )
        }
        val names = (0 until outputCount).map { i ->
            val t = interpreter.getOutputTensor(i)
            Log.i(
                TAG,
                "init: output[$i] name='${t.name()}' shape=${t.shape().toList()} dtype=${t.dataType()} " +
                    "bytes=${t.numBytes()}",
            )
            t.name().lowercase()
        }
        regOutputIndex = names.indexOfFirst { it.contains("regressor") }
            .also { require(it >= 0) { "BlazeFace 'regressors' output not found; got $names" } }
        clsOutputIndex = names.indexOfFirst { it.contains("classificator") || it.contains("classif") }
            .also { require(it >= 0) { "BlazeFace 'classificators' output not found; got $names" } }
        Log.i(
            TAG,
            "init: resolved regressors -> output[$regOutputIndex] '${names[regOutputIndex]}', " +
                "classifications -> output[$clsOutputIndex] '${names[clsOutputIndex]}'",
        )
        Log.i(
            TAG,
            "init: ready inputSize=${inputSize}x${inputSize}x3 anchors=$NUM_ANCHORS " +
                "regStride=${BlazeFaceDecoder.REG_STRIDE} delegate=${runtime.activeDelegate}",
        )
    }

    // Reused between calls to avoid GC churn during clustering of 50+ images.
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(inputSize * inputSize * 3 * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val regOutput: Array<Array<FloatArray>> = Array(1) { Array(NUM_ANCHORS) { FloatArray(BlazeFaceDecoder.REG_STRIDE) } }
    private val clsOutput: Array<Array<FloatArray>> = Array(1) { Array(NUM_ANCHORS) { FloatArray(1) } }
    private val outputs: MutableMap<Int, Any> = mutableMapOf()
    private val resizePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    /**
     * Run detection on [source] and return faces in source-pixel coordinates.
     * Caller must NOT recycle [source] until this call returns.
     */
    fun detect(source: Bitmap): List<DetectedFace> {
        Log.i(
            TAG,
            "detect: start source=${source.width}x${source.height} config=${source.config} " +
                "byteCount=${source.byteCount} -> input=${inputSize}x${inputSize}x3 " +
                "(letterbox + [-1,1] norm)",
        )
        if (source.isRecycled) {
            Log.e(TAG, "detect: source bitmap is RECYCLED; cannot run detection")
            return emptyList()
        }
        val prepStart = SystemClock.elapsedRealtime()
        prepareInput(source)
        val prepMs = SystemClock.elapsedRealtime() - prepStart
        Log.i(TAG, "detect: prepareInput done in ${prepMs}ms; bound output buffers and submitting to interpreter")

        outputs.clear()
        outputs[regOutputIndex] = regOutput
        outputs[clsOutputIndex] = clsOutput
        val inferStart = SystemClock.elapsedRealtime()
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputs)
        } catch (t: Throwable) {
            Log.e(TAG, "detect: interpreter.run threw after ${SystemClock.elapsedRealtime() - inferStart}ms", t)
            throw t
        }
        val inferMs = SystemClock.elapsedRealtime() - inferStart

        val regs = FloatArray(NUM_ANCHORS * BlazeFaceDecoder.REG_STRIDE)
        for (i in 0 until NUM_ANCHORS) {
            System.arraycopy(regOutput[0][i], 0, regs, i * BlazeFaceDecoder.REG_STRIDE, BlazeFaceDecoder.REG_STRIDE)
        }
        val cls = FloatArray(NUM_ANCHORS) { clsOutput[0][it][0] }

        // Logit distribution stats so we can spot dead/saturated outputs without spamming per-anchor lines.
        var logitMin = Float.POSITIVE_INFINITY
        var logitMax = Float.NEGATIVE_INFINITY
        var logitSum = 0.0
        var posCount = 0
        var aboveZeroSig = 0     // sigmoid(0)=0.5, i.e. logit > 0
        var aboveHalfSig = 0     // sigmoid > 0.75 (typical detector threshold), i.e. logit > ~1.0986
        var topLogit = Float.NEGATIVE_INFINITY
        var topLogitIndex = -1
        for (i in 0 until NUM_ANCHORS) {
            val v = cls[i]
            if (v < logitMin) logitMin = v
            if (v > logitMax) {
                logitMax = v
                topLogit = v
                topLogitIndex = i
            }
            logitSum += v
            if (v > 0f) { posCount++; aboveZeroSig++ }
            if (v > 1.0986123f) aboveHalfSig++
        }
        val logitMean = (logitSum / NUM_ANCHORS).toFloat()
        Log.i(
            TAG,
            "detect: inference done prep=${prepMs}ms infer=${inferMs}ms anchors=$NUM_ANCHORS " +
                "logits[min/mean/max]=${"%.3f".format(logitMin)}/${"%.3f".format(logitMean)}/" +
                "${"%.3f".format(logitMax)} positive=$posCount sigmoidGt0.5=$aboveZeroSig " +
                "sigmoidGt0.75=$aboveHalfSig topAnchor=$topLogitIndex(logit=${"%.3f".format(topLogit)})",
        )
        if (topLogitIndex >= 0) {
            val base = topLogitIndex * BlazeFaceDecoder.REG_STRIDE
            Log.i(
                TAG,
                "detect: topAnchor[$topLogitIndex] regressor dx=${"%.2f".format(regs[base])} " +
                    "dy=${"%.2f".format(regs[base + 1])} w=${"%.2f".format(regs[base + 2])} " +
                    "h=${"%.2f".format(regs[base + 3])}",
            )
        }

        val decodeStart = SystemClock.elapsedRealtime()
        val faces = decoder.decode(regs, cls, source.width, source.height)
        val decodeMs = SystemClock.elapsedRealtime() - decodeStart
        Log.i(
            TAG,
            "detect: done returning ${faces.size} face(s) decode=${decodeMs}ms " +
                "topScores=${faces.take(3).map { "%.3f".format(it.score) }} " +
                "topBboxes=${faces.take(3).map { it.boundingBox }}",
        )
        return faces
    }

    private fun prepareInput(source: Bitmap) {
        // Letterbox-resize to inputSize\u00d7inputSize on a temporary bitmap.
        val scale = minOf(
            inputSize.toFloat() / source.width,
            inputSize.toFloat() / source.height,
        )
        val scaledW = source.width * scale
        val scaledH = source.height * scale
        val padX = (inputSize - scaledW) / 2f
        val padY = (inputSize - scaledH) / 2f
        Log.i(
            TAG,
            "prepareInput: letterbox scale=${"%.4f".format(scale)} " +
                "scaledContent=${"%.1f".format(scaledW)}x${"%.1f".format(scaledH)} " +
                "padding=(x=${"%.1f".format(padX)}, y=${"%.1f".format(padY)}) " +
                "into ${inputSize}x${inputSize}",
        )

        val resizeStart = SystemClock.elapsedRealtime()
        val tmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(tmp)
        val matrix = Matrix().apply {
            postScale(scale, scale)
            postTranslate(padX, padY)
        }
        canvas.drawBitmap(source, matrix, resizePaint)
        val resizeMs = SystemClock.elapsedRealtime() - resizeStart

        val readStart = SystemClock.elapsedRealtime()
        val px = IntArray(inputSize * inputSize)
        tmp.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)
        tmp.recycle()
        val readMs = SystemClock.elapsedRealtime() - readStart

        // Track per-channel min/max post-normalisation so we can confirm the input distribution is
        // sane (a wholly-black or wholly-white frame would collapse into a tiny range).
        var rMin = Float.POSITIVE_INFINITY; var rMax = Float.NEGATIVE_INFINITY; var rSum = 0.0
        var gMin = Float.POSITIVE_INFINITY; var gMax = Float.NEGATIVE_INFINITY; var gSum = 0.0
        var bMin = Float.POSITIVE_INFINITY; var bMax = Float.NEGATIVE_INFINITY; var bSum = 0.0

        val normStart = SystemClock.elapsedRealtime()
        inputBuffer.rewind()
        for (color in px) {
            // BlazeFace expects [-1, 1] normalisation per channel.
            val r = ((color shr 16) and 0xFF) / 127.5f - 1f
            val g = ((color shr 8) and 0xFF) / 127.5f - 1f
            val b = (color and 0xFF) / 127.5f - 1f
            if (r < rMin) rMin = r; if (r > rMax) rMax = r; rSum += r
            if (g < gMin) gMin = g; if (g > gMax) gMax = g; gSum += g
            if (b < bMin) bMin = b; if (b > bMax) bMax = b; bSum += b
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        inputBuffer.rewind()
        val normMs = SystemClock.elapsedRealtime() - normStart
        val pixelCount = px.size.toDouble()
        Log.i(
            TAG,
            "prepareInput: pixels=${px.size} resize=${resizeMs}ms readPixels=${readMs}ms norm=${normMs}ms " +
                "bufferBytes=${inputBuffer.capacity()}",
        )
        Log.i(
            TAG,
            "prepareInput: normalised channel stats " +
                "R[min=${"%.3f".format(rMin)} mean=${"%.3f".format(rSum / pixelCount)} max=${"%.3f".format(rMax)}] " +
                "G[min=${"%.3f".format(gMin)} mean=${"%.3f".format(gSum / pixelCount)} max=${"%.3f".format(gMax)}] " +
                "B[min=${"%.3f".format(bMin)} mean=${"%.3f".format(bSum / pixelCount)} max=${"%.3f".format(bMax)}]",
        )
        if (rMax - rMin < 0.05f && gMax - gMin < 0.05f && bMax - bMin < 0.05f) {
            Log.w(
                TAG,
                "prepareInput: very low pixel variance across all channels; source bitmap may be " +
                    "blank/solid colour and detection is unlikely to find faces",
            )
        }
    }

    override fun close() {
        Log.i(TAG, "close: releasing BlazeFace interpreter")
        runCatching { interpreter.close() }
            .onFailure { Log.w(TAG, "close: interpreter.close() threw", it) }
    }

    companion object {
        private const val TAG: String = "FaceMesh.Detector"
        const val NUM_ANCHORS: Int = 896
    }
}
