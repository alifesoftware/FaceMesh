package com.alifesoftware.facemesh.ml

import android.graphics.Bitmap
import java.io.Closeable

/**
 * Common surface for the two BlazeFace variants the app ships:
 *
 *   - [BlazeFaceShortRangeDetector] (input 128 x 128, 896 anchors over 16x16x2 + 8x8x6 grid).
 *     Trained for selfie-distance frontal faces filling 30..80% of the frame.
 *   - [BlazeFaceFullRangeDetector]  (input 192 x 192, 2304 anchors over a single 48x48 grid).
 *     Trained for a wider face-size and pose distribution; better on group photos / distant faces.
 *
 * The user picks between the two via Settings (see
 * [com.alifesoftware.facemesh.config.PipelineConfig.Detector.DetectorVariant] and the
 * `detectorVariant` flow on [com.alifesoftware.facemesh.data.AppPreferences]).
 *
 * Implementations are deliberately **not** sharing code beyond this interface and the output
 * data classes ([DetectedFace], [FaceLandmarks]) - each variant has its own anchor table,
 * decoder, regressor unpacking buffers, and TFLite output tensor name lookup. This keeps the
 * two paths independently auditable and means a tweak to one model's calibration cannot
 * silently regress the other.
 */
interface FaceDetector : Closeable {

    /**
     * Run detection on [source] and return faces in source-pixel coordinates.
     * The caller must NOT recycle [source] until this call returns.
     */
    fun detect(source: Bitmap): List<DetectedFace>
}
