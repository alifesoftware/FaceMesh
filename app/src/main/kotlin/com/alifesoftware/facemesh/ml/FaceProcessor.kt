package com.alifesoftware.facemesh.ml

import android.content.ContentResolver
import android.graphics.Bitmap
import android.net.Uri
import com.alifesoftware.facemesh.media.BitmapDecoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Per-image face pipeline used by both Clusterify and Filter:
 *   `Uri \u2192 decode+downsample \u2192 BlazeFace \u2192 FaceFilters \u2192 align \u2192 GhostFaceNet \u2192 L2`
 *
 * Returns one [FaceRecord] per validated face; the bitmap is recycled internally.
 */
class FaceProcessor(
    private val resolver: ContentResolver,
    private val detector: BlazeFaceDetector,
    private val aligner: FaceAligner,
    private val embedder: FaceEmbedder,
) {

    data class FaceRecord(
        val sourceUri: Uri,
        val faceIndex: Int,
        val embedding: FloatArray,
        val representativeCrop: Bitmap?,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is FaceRecord) return false
            if (sourceUri != other.sourceUri) return false
            if (faceIndex != other.faceIndex) return false
            if (!embedding.contentEquals(other.embedding)) return false
            // representativeCrop intentionally not part of equality
            return true
        }
        override fun hashCode(): Int {
            var result = sourceUri.hashCode()
            result = 31 * result + faceIndex
            result = 31 * result + embedding.contentHashCode()
            return result
        }
    }

    /**
     * @param keepRepresentativeCrop when true, the first face per image keeps a 112\u00d7112 crop so
     * the cluster can pick a representative thumbnail without re-running alignment later.
     */
    suspend fun process(uri: Uri, keepRepresentativeCrop: Boolean = false): List<FaceRecord> =
        withContext(Dispatchers.Default) {
            val bitmap = withContext(Dispatchers.IO) { BitmapDecoder.decode(resolver, uri) }
            try {
                val candidates = detector.detect(bitmap)
                val accepted = FaceFilters.apply(candidates)
                accepted.mapIndexed { index, face ->
                    val aligned = aligner.align(bitmap, face)
                    val embedding = embedder.embed(aligned)
                    val keep = keepRepresentativeCrop && index == 0
                    if (!keep) aligned.recycle()
                    FaceRecord(
                        sourceUri = uri,
                        faceIndex = index,
                        embedding = embedding,
                        representativeCrop = if (keep) aligned else null,
                    )
                }
            } finally {
                bitmap.recycle()
            }
        }
}
