package com.alifesoftware.facemesh.ml.download

import org.json.JSONObject

/**
 * Pure-Kotlin representation of the sidecar bundle manifest (SPEC §8.1).
 *
 * Hand-rolled JSON parsing avoids pulling in Gson / kotlinx-serialization which would push us
 * outside the 500 KB APK budget.
 *
 * ## Schema versions
 *
 *   - **v1** shipped one detector model and config keys `detector_input` + `embedder_input`.
 *   - **v2** adds the BlazeFace full-range detector as a second model entry. The config
 *     adds `detector_short_range_input` and `detector_full_range_input`. v1 keys are still
 *     accepted as fall-backs so cached v1 manifests on devices upgrading from older builds
 *     continue to load (with the full-range variant disabled).
 *
 * Parser tolerates both versions; the writer always emits v2 format.
 */
data class ModelManifest(
    val version: Int,
    val models: List<ModelDescriptor>,
    val config: ModelConfig,
) {
    companion object {
        fun parse(json: String): ModelManifest {
            val root = JSONObject(json)
            val modelsArray = root.getJSONArray("models")
            val models = (0 until modelsArray.length()).map { i ->
                val obj = modelsArray.getJSONObject(i)
                ModelDescriptor(
                    type = obj.getString("type"),
                    name = obj.getString("name"),
                    relativeUrl = obj.getString("url"),
                    sha256 = obj.getString("sha256"),
                )
            }
            val configObj = root.getJSONObject("config")
            // Schema-tolerant detector input parsing: prefer v2 keys, fall back to legacy v1
            // `detector_input` (which always meant the short-range model). Full-range is
            // optional in v1 manifests; if absent we default to the canonical 192x192.
            val shortRange = parseInputDims(
                configObj,
                preferredKey = "detector_short_range_input",
                fallbackKey = "detector_input",
                default = intArrayOf(128, 128),
            )
            val fullRange = parseInputDims(
                configObj,
                preferredKey = "detector_full_range_input",
                fallbackKey = null,
                default = intArrayOf(192, 192),
            )
            val inputEmb = configObj.getJSONArray("embedder_input")
            val config = ModelConfig(
                dbscanEps = configObj.getDouble("dbscan_eps").toFloat(),
                dbscanMinPts = configObj.getInt("dbscan_min_pts"),
                matchThreshold = configObj.getDouble("match_threshold").toFloat(),
                shortRangeDetectorInput = shortRange,
                fullRangeDetectorInput = fullRange,
                embedderInput = intArrayOf(inputEmb.getInt(0), inputEmb.getInt(1)),
            )
            return ModelManifest(
                version = root.getInt("version"),
                models = models,
                config = config,
            )
        }

        private fun parseInputDims(
            configObj: JSONObject,
            preferredKey: String,
            fallbackKey: String?,
            default: IntArray,
        ): IntArray {
            val arr = when {
                configObj.has(preferredKey) -> configObj.getJSONArray(preferredKey)
                fallbackKey != null && configObj.has(fallbackKey) -> configObj.getJSONArray(fallbackKey)
                else -> return default
            }
            return intArrayOf(arr.getInt(0), arr.getInt(1))
        }
    }
}

data class ModelDescriptor(
    /**
     * Semantic role this artifact fulfils. Stable across re-exports / re-naming so that
     * [com.alifesoftware.facemesh.ml.MlPipelineProvider] can resolve the right file even if
     * the upstream filename changes. Known values: [TYPE_DETECTOR_SHORT_RANGE],
     * [TYPE_DETECTOR_FULL_RANGE], [TYPE_EMBEDDER].
     */
    val type: String,
    /** On-disk filename; used by [com.alifesoftware.facemesh.ml.download.ModelStore]. */
    val name: String,
    /** URL relative to `BuildConfig.MODEL_BASE_URL`. */
    val relativeUrl: String,
    val sha256: String,
) {
    companion object {
        const val TYPE_DETECTOR_SHORT_RANGE: String = "detector_blazeface_short_range"
        const val TYPE_DETECTOR_FULL_RANGE: String = "detector_blazeface_full_range"
        const val TYPE_EMBEDDER: String = "embedder_ghostfacenet_fp16"

        /**
         * Backwards-compatibility alias for callers / tests still using the old name. Equal to
         * [TYPE_DETECTOR_SHORT_RANGE]. Prefer the explicit names in new code.
         */
        @Deprecated(
            "Use TYPE_DETECTOR_SHORT_RANGE for the original detector or TYPE_DETECTOR_FULL_RANGE for the new one.",
            ReplaceWith("TYPE_DETECTOR_SHORT_RANGE"),
        )
        const val TYPE_DETECTOR: String = TYPE_DETECTOR_SHORT_RANGE
    }
}

data class ModelConfig(
    val dbscanEps: Float,
    val dbscanMinPts: Int,
    val matchThreshold: Float,
    val shortRangeDetectorInput: IntArray,
    val fullRangeDetectorInput: IntArray,
    val embedderInput: IntArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ModelConfig) return false
        if (dbscanEps != other.dbscanEps) return false
        if (dbscanMinPts != other.dbscanMinPts) return false
        if (matchThreshold != other.matchThreshold) return false
        if (!shortRangeDetectorInput.contentEquals(other.shortRangeDetectorInput)) return false
        if (!fullRangeDetectorInput.contentEquals(other.fullRangeDetectorInput)) return false
        if (!embedderInput.contentEquals(other.embedderInput)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = dbscanEps.hashCode()
        result = 31 * result + dbscanMinPts
        result = 31 * result + matchThreshold.hashCode()
        result = 31 * result + shortRangeDetectorInput.contentHashCode()
        result = 31 * result + fullRangeDetectorInput.contentHashCode()
        result = 31 * result + embedderInput.contentHashCode()
        return result
    }
}
