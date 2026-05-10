package com.alifesoftware.facemesh.media

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.IOException
import kotlin.math.max

/**
 * Loads images from `content://` URIs with controlled memory usage:
 *   \u2022 Two-pass decode (bounds first, then sample-sized).
 *   \u2022 EXIF rotation honored.
 *   \u2022 Largest dimension capped at [MAX_DIM] to satisfy SPEC FR-35 / NFR-04.
 *
 * Phase 2 surface; Phase 4 calls into this for detection input.
 */
object BitmapDecoder {

    const val MAX_DIM: Int = 1280

    @Throws(IOException::class)
    fun decode(resolver: ContentResolver, uri: Uri, maxDim: Int = MAX_DIM): Bitmap {
        val (w, h) = readBounds(resolver, uri)
        val sample = computeInSampleSize(w, h, maxDim)
        val opts = BitmapFactory.Options().apply {
            inJustDecodeBounds = false
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val raw = resolver.openInputStream(uri).use {
            requireNotNull(it) { "Could not open $uri" }
            BitmapFactory.decodeStream(it, null, opts)
        } ?: throw IOException("Failed to decode $uri")
        return applyExifRotation(resolver, uri, raw)
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
        while (max(w, h) > maxDim) {
            sample *= 2
            w /= 2
            h /= 2
        }
        return sample
    }

    private fun applyExifRotation(resolver: ContentResolver, uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = try {
            resolver.openInputStream(uri).use { input ->
                input?.let { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
                    ?: ExifInterface.ORIENTATION_NORMAL
            }
        } catch (_: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            else -> return bitmap
        }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }
}
