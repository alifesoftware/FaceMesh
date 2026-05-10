package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
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
        val names = (0 until interpreter.outputTensorCount).map { i ->
            interpreter.getOutputTensor(i).name().lowercase()
        }
        regOutputIndex = names.indexOfFirst { it.contains("regressor") }
            .also { require(it >= 0) { "BlazeFace 'regressors' output not found; got $names" } }
        clsOutputIndex = names.indexOfFirst { it.contains("classificator") || it.contains("classif") }
            .also { require(it >= 0) { "BlazeFace 'classificators' output not found; got $names" } }
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
        prepareInput(source)
        outputs.clear()
        outputs[regOutputIndex] = regOutput
        outputs[clsOutputIndex] = clsOutput
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputs)

        val regs = FloatArray(NUM_ANCHORS * BlazeFaceDecoder.REG_STRIDE)
        for (i in 0 until NUM_ANCHORS) {
            System.arraycopy(regOutput[0][i], 0, regs, i * BlazeFaceDecoder.REG_STRIDE, BlazeFaceDecoder.REG_STRIDE)
        }
        val cls = FloatArray(NUM_ANCHORS) { clsOutput[0][it][0] }
        return decoder.decode(regs, cls, source.width, source.height)
    }

    private fun prepareInput(source: Bitmap) {
        // Letterbox-resize to inputSize\u00d7inputSize on a temporary bitmap.
        val tmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(tmp)
        val matrix = Matrix().apply {
            val scale = minOf(
                inputSize.toFloat() / source.width,
                inputSize.toFloat() / source.height,
            )
            postScale(scale, scale)
            postTranslate(
                (inputSize - source.width * scale) / 2f,
                (inputSize - source.height * scale) / 2f,
            )
        }
        canvas.drawBitmap(source, matrix, resizePaint)

        val px = IntArray(inputSize * inputSize)
        tmp.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)
        tmp.recycle()

        inputBuffer.rewind()
        for (color in px) {
            // BlazeFace expects [-1, 1] normalisation per channel.
            val r = ((color shr 16) and 0xFF) / 127.5f - 1f
            val g = ((color shr 8) and 0xFF) / 127.5f - 1f
            val b = (color and 0xFF) / 127.5f - 1f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        inputBuffer.rewind()
    }

    override fun close() {
        runCatching { interpreter.close() }
    }

    companion object {
        const val NUM_ANCHORS: Int = 896
    }
}
