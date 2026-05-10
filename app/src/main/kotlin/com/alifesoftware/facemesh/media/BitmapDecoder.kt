package com.alifesoftware.facemesh.media

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import com.alifesoftware.facemesh.config.PipelineConfig
import java.io.IOException
import kotlin.math.max

/**
 * Loads images from `content://` URIs with controlled memory usage:
 *   \u2022 Two-pass decode (bounds first, then sample-sized).
 *   \u2022 EXIF rotation honored.
 *   \u2022 Largest dimension capped at [PipelineConfig.Decode.maxLongEdgePx] to satisfy
 *     SPEC FR-35 / NFR-04.
 *
 * Phase 2 surface; Phase 4 calls into this for detection input.
 */
object BitmapDecoder {

    private const val TAG = "FaceMesh.Decoder"

    @Throws(IOException::class)
    fun decode(
        resolver: ContentResolver,
        uri: Uri,
        maxDim: Int = PipelineConfig.Decode.maxLongEdgePx,
    ): Bitmap {
        val started = SystemClock.elapsedRealtime()
        Log.i(TAG, "decode: start uri=$uri maxDim=$maxDim")
        val (w, h) = readBounds(resolver, uri)
        val sample = computeInSampleSize(w, h, maxDim)
        Log.i(TAG, "decode: bounds=${w}x${h} sampleSize=$sample (target<=$maxDim)")
        val opts = BitmapFactory.Options().apply {
            inJustDecodeBounds = false
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val raw = resolver.openInputStream(uri).use {
            requireNotNull(it) { "Could not open $uri" }
            BitmapFactory.decodeStream(it, null, opts)
        } ?: run {
            Log.e(TAG, "decode: BitmapFactory returned null for uri=$uri")
            throw IOException("Failed to decode $uri")
        }
        Log.i(TAG, "decode: rawDecoded=${raw.width}x${raw.height} config=${raw.config}")
        val rotated = applyExifRotation(resolver, uri, raw)
        val elapsed = SystemClock.elapsedRealtime() - started
        Log.i(
            TAG,
            "decode: done uri=$uri final=${rotated.width}x${rotated.height} " +
                "rotatedNew=${rotated !== raw} took=${elapsed}ms",
        )
        return rotated
    }

    private fun readBounds(resolver: ContentResolver, uri: Uri): Pair<Int, Int> {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri).use {
            requireNotNull(it) { "Could not open $uri" }
            BitmapFactory.decodeStream(it, null, opts)
        }
        return opts.outWidth to opts.outHeight
    }

    /** Smallest power-of-two sample size that brings the longest side <= maxDim. */
    internal fun computeInSampleSize(width: Int, height: Int, maxDim: Int): Int {
        var sample = 1
        var w = width
        var h = height
        var iter = 0
        while (max(w, h) > maxDim) {
            sample *= 2
            w /= 2
            h /= 2
            iter++
            Log.i(TAG, "computeInSampleSize: halving #$iter -> sample=$sample est=${w}x${h} (target<=$maxDim)")
        }
        if (iter == 0) {
            Log.i(TAG, "computeInSampleSize: ${width}x${height} already fits ${maxDim}; sample=1")
        } else {
            Log.i(TAG, "computeInSampleSize: ${width}x${height} -> sample=$sample after $iter halving(s)")
        }
        return sample
    }

    private fun applyExifRotation(resolver: ContentResolver, uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = try {
            resolver.openInputStream(uri).use { input ->
                input?.let { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
                    ?: ExifInterface.ORIENTATION_NORMAL
            }
        } catch (e: Exception) {
            Log.w(TAG, "applyExifRotation: failed to read EXIF for uri=$uri; assuming NORMAL", e)
            ExifInterface.ORIENTATION_NORMAL
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            // EXIF 5 / 7 are the rare compound orientations (rotate + flip) that some
            // scanners and panorama tools emit. Recipe matches Glide's well-tested
            // initializeMatrixForRotation: rotate first, then post-scale -1 along X.
            ExifInterface.ORIENTATION_TRANSPOSE -> matrix.apply {
                postRotate(90f)
                postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> matrix.apply {
                postRotate(-90f)
                postScale(-1f, 1f)
            }
            else -> {
                Log.i(TAG, "applyExifRotation: orientation=$orientation, no transform applied")
                return bitmap
            }
        }
        Log.i(TAG, "applyExifRotation: orientation=$orientation, applying transform")
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }
}
