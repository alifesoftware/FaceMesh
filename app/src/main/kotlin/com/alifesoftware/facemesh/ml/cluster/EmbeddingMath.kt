package com.alifesoftware.facemesh.ml.cluster

import android.util.Log
import kotlin.math.sqrt

/**
 * Hand-rolled vector math for face embeddings (SPEC \u00a76.4 / \u00a76.6).
 *
 * Pure-Kotlin, allocation-free where it matters. Inputs are expected to be 128-dimensional but
 * the helpers work for any equal-length pair.
 */
object EmbeddingMath {

    private const val TAG: String = "FaceMesh.EmbedMath"

    /**
     * Mutates [vec] in place to have unit L2 norm. Returns [vec] for chaining. Vectors with
     * essentially-zero norm (e.g., dead embeddings from a noise crop) are left untouched and
     * filled with zeros to avoid NaNs downstream.
     */
    fun l2NormalizeInPlace(vec: FloatArray): FloatArray {
        var sum = 0.0
        for (v in vec) sum += v.toDouble() * v
        val norm = sqrt(sum)
        if (norm < 1e-8) {
            Log.w(TAG, "l2NormalizeInPlace: norm=$norm (<1e-8); zero-filling ${vec.size}-d vector")
            vec.fill(0f)
            return vec
        }
        val inv = (1.0 / norm).toFloat()
        for (i in vec.indices) vec[i] *= inv
        Log.i(TAG, "l2NormalizeInPlace: dim=${vec.size} norm=${"%.4f".format(norm)} -> unit (scale=${"%.4f".format(inv.toDouble())})")
        return vec
    }

    /** Dot product. Equivalent to cosine similarity when both inputs are L2-normalised. */
    fun dot(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "size mismatch: ${a.size} vs ${b.size}" }
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }

    /** Cosine **distance** = 1 - cosine similarity (= 1 - dot for L2-normalised vectors). */
    fun cosineDistance(a: FloatArray, b: FloatArray): Float = (1f - dot(a, b)).coerceIn(0f, 2f)

    /**
     * Component-wise mean of [vectors] followed by L2 normalisation. Returns a new array.
     * Used to compute cluster centroids (SPEC \u00a75.2).
     */
    fun meanAndNormalize(vectors: List<FloatArray>): FloatArray {
        require(vectors.isNotEmpty()) { "Cannot average an empty list" }
        val dim = vectors.first().size
        Log.i(TAG, "meanAndNormalize: averaging ${vectors.size} vector(s) of dim=$dim")
        val out = FloatArray(dim)
        for (v in vectors) {
            require(v.size == dim) { "dimension mismatch in meanAndNormalize" }
            for (i in 0 until dim) out[i] += v[i]
        }
        val n = vectors.size.toFloat()
        for (i in 0 until dim) out[i] /= n
        return l2NormalizeInPlace(out)
    }
}
