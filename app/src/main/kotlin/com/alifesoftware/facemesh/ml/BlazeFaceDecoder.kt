package com.alifesoftware.facemesh.ml

import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Decodes BlazeFace raw outputs (regressors + classifications) into [DetectedFace]s.
 *
 * Regressor layout per anchor: 16 floats
 *   [0] dx, [1] dy        \u2014 box center offset in pixels (input space)
 *   [2] w,  [3] h         \u2014 box size in pixels (input space)
 *   [4..15]               \u2014 6 landmark (x,y) pairs in input space
 *
 * Classifications per anchor: 1 logit (sigmoid'd to score).
 */
class BlazeFaceDecoder(
    private val inputSize: Int = 128,
    private val anchors: Array<BlazeFaceAnchors.Anchor> = BlazeFaceAnchors.FRONT_128,
    private val scoreThreshold: Float = 0.75f,
    private val iouThreshold: Float = 0.30f,
) {

    /**
     * @param regressors flat float array, shape `[anchors, 16]`.
     * @param classifications flat float array, shape `[anchors, 1]` of raw logits.
     * @param sourceWidth pixel width of the source bitmap (for unprojecting from [inputSize]).
     * @param sourceHeight pixel height of the source bitmap.
     */
    fun decode(
        regressors: FloatArray,
        classifications: FloatArray,
        sourceWidth: Int,
        sourceHeight: Int,
    ): List<DetectedFace> {
        require(regressors.size == anchors.size * REG_STRIDE) {
            "regressors size ${regressors.size} != ${anchors.size * REG_STRIDE}"
        }
        require(classifications.size == anchors.size) {
            "classifications size ${classifications.size} != ${anchors.size}"
        }

        val candidates = ArrayList<DetectedFace>(32)
        val sx = sourceWidth.toFloat() / inputSize
        val sy = sourceHeight.toFloat() / inputSize

        var droppedLowScore = 0
        var droppedDegenerate = 0
        var topLogit = Float.NEGATIVE_INFINITY
        var topScore = 0f

        for (i in anchors.indices) {
            val score = sigmoid(classifications[i])
            if (classifications[i] > topLogit) {
                topLogit = classifications[i]
                topScore = score
            }
            if (score < scoreThreshold) {
                droppedLowScore++
                continue
            }
            val anchor = anchors[i]
            val base = i * REG_STRIDE

            // Box center is anchor center + offset/inputSize in normalised coords, then * size.
            val cx = (anchor.cx + regressors[base + 0] / inputSize) * inputSize
            val cy = (anchor.cy + regressors[base + 1] / inputSize) * inputSize
            val w = regressors[base + 2]
            val h = regressors[base + 3]

            val left = (cx - w / 2f) * sx
            val top = (cy - h / 2f) * sy
            val right = (cx + w / 2f) * sx
            val bottom = (cy + h / 2f) * sy

            val rect = RectF(
                left.coerceIn(0f, sourceWidth.toFloat()),
                top.coerceIn(0f, sourceHeight.toFloat()),
                right.coerceIn(0f, sourceWidth.toFloat()),
                bottom.coerceIn(0f, sourceHeight.toFloat()),
            )
            if (rect.width() <= 0f || rect.height() <= 0f) {
                droppedDegenerate++
                Log.i(
                    TAG,
                    "decode: anchor[$i] dropped degenerate rect=$rect (raw l=$left,t=$top,r=$right,b=$bottom) " +
                        "score=${"%.3f".format(score)}",
                )
                continue
            }

            val landmarks = FaceLandmarks(
                rightEye = readLandmark(regressors, base + 4, anchor, sx, sy, sourceWidth, sourceHeight),
                leftEye = readLandmark(regressors, base + 6, anchor, sx, sy, sourceWidth, sourceHeight),
                noseTip = readLandmark(regressors, base + 8, anchor, sx, sy, sourceWidth, sourceHeight),
                mouthCenter = readLandmark(regressors, base + 10, anchor, sx, sy, sourceWidth, sourceHeight),
                rightEarTragion = readLandmark(regressors, base + 12, anchor, sx, sy, sourceWidth, sourceHeight),
                leftEarTragion = readLandmark(regressors, base + 14, anchor, sx, sy, sourceWidth, sourceHeight),
            )
            Log.i(
                TAG,
                "decode: anchor[$i] kept score=${"%.3f".format(score)} bbox=$rect " +
                    "anchorCenter=(${anchor.cx},${anchor.cy}) regOffset=(${regressors[base]},${regressors[base + 1]}) " +
                    "regSize=(${w},${h})",
            )
            candidates += DetectedFace(rect, landmarks, score)
        }

        Log.i(
            TAG,
            "decode: scanned ${anchors.size} anchors topLogit=${"%.3f".format(topLogit)} " +
                "topScore=${"%.3f".format(topScore)} droppedLowScore=$droppedLowScore " +
                "droppedDegenerate=$droppedDegenerate kept=${candidates.size} (threshold=$scoreThreshold)",
        )
        val kept = weightedNms(candidates, iouThreshold)
        Log.i(
            TAG,
            "decode: source=${sourceWidth}x${sourceHeight} threshold=$scoreThreshold " +
                "candidatesAboveThreshold=${candidates.size} keptAfterNms=${kept.size} iou=$iouThreshold",
        )
        return kept
    }

    private fun readLandmark(
        reg: FloatArray,
        offset: Int,
        anchor: BlazeFaceAnchors.Anchor,
        sx: Float,
        sy: Float,
        srcW: Int,
        srcH: Int,
    ): PointF {
        val px = (anchor.cx + reg[offset] / inputSize) * inputSize * sx
        val py = (anchor.cy + reg[offset + 1] / inputSize) * inputSize * sy
        return PointF(px.coerceIn(0f, srcW.toFloat()), py.coerceIn(0f, srcH.toFloat()))
    }

    /** Two-pass IoU-based weighted NMS (per MediaPipe's default "weighted" suppression). */
    private fun weightedNms(
        candidates: List<DetectedFace>,
        iou: Float,
    ): List<DetectedFace> {
        if (candidates.isEmpty()) {
            Log.i(TAG, "weightedNms: no candidates -> early return empty")
            return emptyList()
        }
        val sorted = candidates.sortedByDescending { it.score }.toMutableList()
        val kept = ArrayList<DetectedFace>()
        var groupIndex = 0
        while (sorted.isNotEmpty()) {
            val anchor = sorted.removeAt(0)
            val overlapping = ArrayList<DetectedFace>()
            overlapping += anchor
            val iter = sorted.iterator()
            while (iter.hasNext()) {
                val other = iter.next()
                val iouValue = intersectionOverUnion(anchor.boundingBox, other.boundingBox)
                if (iouValue > iou) {
                    Log.i(
                        TAG,
                        "weightedNms: group[$groupIndex] absorbing overlap " +
                            "anchorScore=${"%.3f".format(anchor.score)} " +
                            "otherScore=${"%.3f".format(other.score)} iou=${"%.3f".format(iouValue)}",
                    )
                    overlapping += other
                    iter.remove()
                }
            }
            val merged = weightedAverage(overlapping)
            Log.i(
                TAG,
                "weightedNms: group[$groupIndex] anchorScore=${"%.3f".format(anchor.score)} " +
                    "merged=${overlapping.size} bbox=${merged.boundingBox}",
            )
            kept += merged
            groupIndex++
        }
        Log.i(TAG, "weightedNms: produced ${kept.size} group(s) from ${candidates.size} candidate(s)")
        return kept
    }

    private fun weightedAverage(group: List<DetectedFace>): DetectedFace {
        if (group.size == 1) return group[0]
        var sum = 0f
        for (f in group) sum += f.score
        if (sum == 0f) return group[0]

        var l = 0f; var t = 0f; var r = 0f; var b = 0f
        var ex1 = 0f; var ey1 = 0f; var ex2 = 0f; var ey2 = 0f
        var nx = 0f; var ny = 0f; var mx = 0f; var my = 0f
        var ear1x = 0f; var ear1y = 0f; var ear2x = 0f; var ear2y = 0f

        for (f in group) {
            val w = f.score / sum
            l += f.boundingBox.left * w
            t += f.boundingBox.top * w
            r += f.boundingBox.right * w
            b += f.boundingBox.bottom * w
            ex1 += f.landmarks.rightEye.x * w; ey1 += f.landmarks.rightEye.y * w
            ex2 += f.landmarks.leftEye.x * w; ey2 += f.landmarks.leftEye.y * w
            nx += f.landmarks.noseTip.x * w; ny += f.landmarks.noseTip.y * w
            mx += f.landmarks.mouthCenter.x * w; my += f.landmarks.mouthCenter.y * w
            ear1x += f.landmarks.rightEarTragion.x * w; ear1y += f.landmarks.rightEarTragion.y * w
            ear2x += f.landmarks.leftEarTragion.x * w; ear2y += f.landmarks.leftEarTragion.y * w
        }
        return DetectedFace(
            boundingBox = RectF(l, t, r, b),
            landmarks = FaceLandmarks(
                rightEye = PointF(ex1, ey1),
                leftEye = PointF(ex2, ey2),
                noseTip = PointF(nx, ny),
                mouthCenter = PointF(mx, my),
                rightEarTragion = PointF(ear1x, ear1y),
                leftEarTragion = PointF(ear2x, ear2y),
            ),
            score = group.first().score,
        )
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        if (interRight <= interLeft || interBottom <= interTop) return 0f
        val inter = (interRight - interLeft) * (interBottom - interTop)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    companion object {
        private const val TAG: String = "FaceMesh.Decoder.Anchor"
        const val REG_STRIDE: Int = 16
        fun sigmoid(x: Float): Float = 1f / (1f + exp((-x).toDouble())).toFloat()
    }
}
