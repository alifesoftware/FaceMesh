package com.alifesoftware.facemesh.ml

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.RectF
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.alifesoftware.facemesh.config.PipelineConfig
import com.alifesoftware.facemesh.media.BitmapDecoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

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
        /**
         * A *natural* (un-warped) square crop of the source bitmap, centred on the face
         * bounding box with [PipelineConfig.DisplayCrop.paddingFraction] padding on each side. Suitable for
         * use as the cluster's avatar thumbnail. Distinct from the affine-aligned 112\u00d7112 crop
         * that goes into the embedder \u2014 that crop is recycled before this record is returned.
         *
         * Null when the caller didn't request a display crop (e.g. the Filter pipeline) or when
         * the source bitmap was too small to provide a usable region.
         */
        val displayCrop: Bitmap?,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is FaceRecord) return false
            if (sourceUri != other.sourceUri) return false
            if (faceIndex != other.faceIndex) return false
            if (!embedding.contentEquals(other.embedding)) return false
            // displayCrop intentionally not part of equality
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
     * @param keepDisplayCrop when true, every accepted face gets a [FaceRecord.displayCrop]
     * suitable for use as a cluster avatar. Producing one per face (not just `faceIndex == 0`)
     * means multi-face photos still surface a usable thumbnail when a person is detected as
     * the second / third face in a group shot.
     */
    suspend fun process(uri: Uri, keepDisplayCrop: Boolean = false): List<FaceRecord> =
        withContext(Dispatchers.Default) {
            val totalStart = SystemClock.elapsedRealtime()
            Log.i(TAG, "process: start uri=$uri keepDisplayCrop=$keepDisplayCrop")
            val decodeStart = SystemClock.elapsedRealtime()
            val bitmap = withContext(Dispatchers.IO) { BitmapDecoder.decode(resolver, uri) }
            val decodeMs = SystemClock.elapsedRealtime() - decodeStart
            Log.i(TAG, "process: decoded ${bitmap.width}x${bitmap.height} in ${decodeMs}ms")
            try {
                val detectStart = SystemClock.elapsedRealtime()
                val candidates = detector.detect(bitmap)
                val detectMs = SystemClock.elapsedRealtime() - detectStart
                Log.i(TAG, "process: detector returned ${candidates.size} candidate(s) in ${detectMs}ms")
                val accepted = FaceFilters.apply(candidates)
                Log.i(
                    TAG,
                    "process: ${accepted.size}/${candidates.size} face(s) survived FaceFilters; " +
                        "starting align+embed loop",
                )
                val records = accepted.mapIndexed { index, face ->
                    val faceStart = SystemClock.elapsedRealtime()
                    val aligned = aligner.align(bitmap, face)
                    val embedding = embedder.embed(aligned)
                    aligned.recycle()  // alignment crop is for the embedder only; never reused for display
                    val display = if (keepDisplayCrop) cropDisplayThumbnail(bitmap, face.boundingBox) else null
                    val faceMs = SystemClock.elapsedRealtime() - faceStart
                    Log.i(
                        TAG,
                        "process: face[$index] align+embed=${faceMs}ms " +
                            "displayCrop=${display?.let { "${it.width}x${it.height}" } ?: "none"} " +
                            "score=${"%.3f".format(face.score)} bbox=${face.boundingBox}",
                    )
                    FaceRecord(
                        sourceUri = uri,
                        faceIndex = index,
                        embedding = embedding,
                        displayCrop = display,
                    )
                }
                val totalMs = SystemClock.elapsedRealtime() - totalStart
                Log.i(
                    TAG,
                    "process: done uri=$uri faces=${records.size} totalTime=${totalMs}ms " +
                        "(decode=${decodeMs}ms detect=${detectMs}ms)",
                )
                records
            } finally {
                bitmap.recycle()
            }
        }

    companion object {
        private const val TAG: String = "FaceMesh.Processor"

        /**
         * Produce a square, padded, *natural-pose* crop of [source] around [bbox]. The crop is
         * shifted (rather than letterboxed) when the padded square would clip past the edge of
         * the source, so the result is always strictly square and visually full.
         *
         * Internal so the unit tests can exercise the geometry directly.
         */
        internal fun cropDisplayThumbnail(
            source: Bitmap,
            bbox: RectF,
            paddingFraction: Float = PipelineConfig.DisplayCrop.paddingFraction,
            maxOutputDim: Int = PipelineConfig.DisplayCrop.maxOutputDim,
        ): Bitmap? {
            if (source.width < 4 || source.height < 4) {
                Log.w(TAG, "cropDisplayThumbnail: source too small (${source.width}x${source.height}); returning null")
                return null
            }
            val cx = (bbox.left + bbox.right) / 2f
            val cy = (bbox.top + bbox.bottom) / 2f
            val faceMaxSide = max(bbox.width(), bbox.height())
            if (faceMaxSide < 1f) {
                Log.w(TAG, "cropDisplayThumbnail: degenerate bbox=$bbox; returning null")
                return null
            }

            val paddedSquare = faceMaxSide * (1f + 2f * paddingFraction)
            val sourceMinSide = min(source.width, source.height).toFloat()
            val cropSize = paddedSquare.coerceAtMost(sourceMinSide)
            if (paddedSquare > sourceMinSide) {
                Log.i(
                    TAG,
                    "cropDisplayThumbnail: paddedSquare=${"%.1f".format(paddedSquare)} > " +
                        "sourceMinSide=$sourceMinSide; CLAMP cropSize=${"%.1f".format(cropSize)}",
                )
            }
            val half = cropSize / 2f

            var left = cx - half
            var top = cy - half
            var right = cx + half
            var bottom = cy + half

            // Shift the crop to stay square against source edges.
            if (left < 0f) {
                Log.i(TAG, "cropDisplayThumbnail: SHIFT left underflow=${"%.1f".format(-left)}")
                right -= left; left = 0f
            }
            if (top < 0f) {
                Log.i(TAG, "cropDisplayThumbnail: SHIFT top underflow=${"%.1f".format(-top)}")
                bottom -= top; top = 0f
            }
            if (right > source.width) {
                Log.i(
                    TAG,
                    "cropDisplayThumbnail: SHIFT right overflow=${"%.1f".format(right - source.width)}",
                )
                left -= (right - source.width); right = source.width.toFloat()
            }
            if (bottom > source.height) {
                Log.i(
                    TAG,
                    "cropDisplayThumbnail: SHIFT bottom overflow=${"%.1f".format(bottom - source.height)}",
                )
                top -= (bottom - source.height); bottom = source.height.toFloat()
            }

            // Final clamp guards against pathological inputs.
            val preClampL = left; val preClampT = top; val preClampR = right; val preClampB = bottom
            left = left.coerceAtLeast(0f)
            top = top.coerceAtLeast(0f)
            right = right.coerceAtMost(source.width.toFloat())
            bottom = bottom.coerceAtMost(source.height.toFloat())
            if (left != preClampL || top != preClampT || right != preClampR || bottom != preClampB) {
                Log.i(
                    TAG,
                    "cropDisplayThumbnail: FINAL CLAMP " +
                        "from=[${"%.1f".format(preClampL)},${"%.1f".format(preClampT)}," +
                        "${"%.1f".format(preClampR)},${"%.1f".format(preClampB)}] " +
                        "to=[${"%.1f".format(left)},${"%.1f".format(top)}," +
                        "${"%.1f".format(right)},${"%.1f".format(bottom)}]",
                )
            }

            val w = (right - left).toInt().coerceAtLeast(1)
            val h = (bottom - top).toInt().coerceAtLeast(1)
            val cropped = Bitmap.createBitmap(source, left.toInt(), top.toInt(), w, h)

            // Down-scale if the natural crop is bigger than the on-disk cap.
            val maxDim = max(w, h)
            if (maxDim <= maxOutputDim) {
                Log.i(
                    TAG,
                    "cropDisplayThumbnail: bbox=$bbox source=${source.width}x${source.height} " +
                        "natural=${w}x${h} (no rescale)",
                )
                return cropped
            }
            val scale = maxOutputDim.toFloat() / maxDim
            val scaled = Bitmap.createScaledBitmap(
                cropped,
                (w * scale).toInt().coerceAtLeast(1),
                (h * scale).toInt().coerceAtLeast(1),
                /* filter = */ true,
            )
            if (scaled !== cropped) cropped.recycle()
            Log.i(
                TAG,
                "cropDisplayThumbnail: bbox=$bbox source=${source.width}x${source.height} " +
                    "natural=${w}x${h} -> scaled=${scaled.width}x${scaled.height} (cap=$maxOutputDim)",
            )
            return scaled
        }

        private fun min(a: Int, b: Int): Int = if (a < b) a else b
    }
}
