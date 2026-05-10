package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import com.alifesoftware.facemesh.ml.cluster.EmbeddingMath
import org.tensorflow.lite.InterpreterApi
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GhostFaceNet-V1 (FP16) embedder running on the [TfLiteRuntime].
 *
 * Input:  112\u00d7112\u00d73 RGB float (NHWC), normalised to [-1, 1].
 * Output: 512-d embedding, L2-normalised by [embed].
 *
 * Confirmed against the ONNX FP32 reference at
 * https://github.com/alifesoftware/ModelZoo/blob/master/GhostFaceNet/Model/ghostface_fp32.onnx
 *   input  : float32[1, 112, 112, 3]
 *   output : float32[1, 512]
 *
 * Re-uses input/output buffers across calls to keep allocations down (SPEC NFR-04).
 */
class FaceEmbedder(
    private val runtime: TfLiteRuntime,
    modelFile: File,
    private val inputSize: Int = INPUT_SIZE,
    private val embeddingSize: Int = EMBEDDING_SIZE,
) : Closeable {

    private val interpreter: InterpreterApi = runtime.newInterpreter(runtime.loadModel(modelFile))

    private val input: ByteBuffer = ByteBuffer
        .allocateDirect(inputSize * inputSize * 3 * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val output: Array<FloatArray> = arrayOf(FloatArray(embeddingSize))

    /**
     * @param alignedFace must already be [inputSize] x [inputSize]. Returned embedding is owned
     * by the caller (a fresh copy is allocated each call so callers can store it without
     * coordination).
     */
    fun embed(alignedFace: Bitmap): FloatArray {
        require(alignedFace.width == inputSize && alignedFace.height == inputSize) {
            "FaceEmbedder expects ${inputSize}x${inputSize}, got ${alignedFace.width}x${alignedFace.height}"
        }
        prepareInput(alignedFace)
        interpreter.run(input, output)
        val copy = output[0].copyOf()
        return EmbeddingMath.l2NormalizeInPlace(copy)
    }

    private fun prepareInput(bitmap: Bitmap) {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        input.rewind()
        for (color in pixels) {
            val r = ((color shr 16) and 0xFF) / 127.5f - 1f
            val g = ((color shr 8) and 0xFF) / 127.5f - 1f
            val b = (color and 0xFF) / 127.5f - 1f
            input.putFloat(r)
            input.putFloat(g)
            input.putFloat(b)
        }
        input.rewind()
    }

    override fun close() {
        runCatching { interpreter.close() }
    }

    companion object {
        const val INPUT_SIZE: Int = 112
        const val EMBEDDING_SIZE: Int = 512
    }
}
